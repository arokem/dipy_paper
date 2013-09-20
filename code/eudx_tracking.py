import nibabel as nib
from nibabel import trackvis as tv
from dipy.viz import fvtk

dname = "/home/eleftherios/Data/Dave_St-Amour/results/"

"""
T1
"""
fT1 = dname + "t1_brain_on_b0_brain_nonlinear.nii.gz"
img = nib.load(fT1)
data_T1 = img.get_data()
affine_T1 = img.get_affine()

"""
Load Streamlines
"""

# ftrk = dname + "whole_brain_wm_ants.trk"
ftrk = dname + "whole_brain_wm_ants_cing.trk"
ren = fvtk.ren()

from dipy.reconst.shm import sh_to_sf
from dipy.reconst.odf import peaks_from_model

streams, hdr = tv.read(ftrk)

streamlines = [s[0] for s in streams]

print(len(streamlines))

from dipy.tracking.vox2track import track_counts

"""
Create intersection roi
"""

def create_roi(data_shape, pos, rw, value=1):
    i, j, k = pos
    roi = np.zeros(data_shape)
    roi[i-rw:i+rw, j-rw:j+rw, k-rw:k+rw] = value
    return roi


def filter_roi_tracks(tracks, roi, value=1):
    """ bring the tracks from the roi region and their indices
    """
    tcs,tes = track_counts(streamlines, roi.shape, (1, 1, 1), True)
    roiinds=np.where(roi==value)
    roiinds=np.array(roiinds).T

    cnt=0
    sinds=[]
    for vox in roiinds:
        try:
            sinds+=tes[tuple(vox)]
            cnt+=1
        except:
            pass
    return [tracks[i] for i in list(set(sinds))], list(set(sinds))

# splenium

# pos = (90, 122, 66)
# I, J, K = pos

# roi = create_roi(data_T1.shape, pos=pos, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines,roi)

# pos2 = (80, 122, 76)
# roi = create_roi(data_T1.shape, pos=pos2, rw=5)
# streamlines, sinds = filter_roi_tracks(streamlines,roi)

# pos3 = (110, 122, 76)
# roi = create_roi(data_T1.shape, pos=pos3, rw=5)
# streamlines, sinds = filter_roi_tracks(streamlines,roi)

# CC far

# pos = (62, 121, 99)
# I, J, K = pos

# roi = create_roi(data_T1.shape, pos=pos, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines,roi)

# pos2 = (120, 121, 99)

# roi = create_roi(data_T1.shape, pos=pos2, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines,roi)

# cingulum

# pos = (85, 121, 81)
# I, J, K = pos

# roi = create_roi(data_T1.shape, pos=pos, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines, roi)

# pos2 = (85, 131, 70)

# roi = create_roi(data_T1.shape, pos=pos2, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines, roi)

# pos3 = (83, 82, 83)

# roi = create_roi(data_T1.shape, pos=pos3, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines, roi)

# pos4 = (80, 69, 78)

# roi = create_roi(data_T1.shape, pos=pos4, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines, roi)

# pos5 = (81, 62, 63)

# roi = create_roi(data_T1.shape, pos=pos5, rw=10)
# streamlines, sinds = filter_roi_tracks(streamlines, roi)


# streams2 = [(s, None, None) for s in streamlines]

# tv.write(dname + "whole_brain_wm_ants_2.trk", streams2, hdr)

print(len(streamlines))

# from dipy.tracking.distances import approx_polygon_track
# from dipy.tracking.metrics import length

# streamlines = [approx_polygon_track(s, 0.2) for s in streamlines if length(s) > 100]

from dipy.viz.colormap import line_colors

# no = 10000

# from dipy.segment.quickbundles import QuickBundles

# qb = QuickBundles(streamlines[:no], 25, 20)

# print(qb.total_clusters)

# qb.remove_small_clusters(200)

# print(qb.total_clusters)

# streamline_ids = []

# for c in qb.clustering:
#     streamline_ids+=qb.label2tracksids(c)

# final_streamlines = [streamlines[id] for id in streamline_ids]

fvtk.add(ren, fvtk.streamtube(streamlines,
         line_colors(streamlines), linewidth=2 * 0.15))


fvtk.add(ren, fvtk.slicer(data_T1,
                          plane_i=[data_T1.shape[0]/2],
                          plane_j=None,
                          plane_k=None,
                          outline=False))

fvtk.show(ren)
