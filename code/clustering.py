import numpy as np
from nibabel import trackvis as tv
from dipy.segment.quickbundles import QuickBundles
from dipy.data import get_data
from nibabel import trackvis


fname = 'csa_streamlines.trk'
streams, hdr = trackvis.read(fname)

streams, hdr = trackvis.read(fname)
streamlines = [s[0] for s in streams]

qb = QuickBundles(streamlines, dist_thr=25., pts=12)

centroids = qb.centroids
colormap = np.random.rand(len(centroids), 3)

colors = np.ones((len(streamlines), 3))
for i, centroid in enumerate(centroids):
    inds = qb.label2tracksids(i)
    colors[inds] = colormap[i]


def save_streamlines_with_color(streamlines, colors, out_file, hdr):
    trk = []

    for i, streamline in enumerate(streamlines):
        # Color (RBG [0-255]) for each point of the streamline
        scalars = np.array([colors[i]] * len(streamline)) * 255
        properties = None
        trk.append((streamline, scalars, properties))

    tv.write(out_file, trk, hdr)


save_streamlines_with_color(streamlines, colors, 'csa_streamlines_clusters25.trk', hdr)



