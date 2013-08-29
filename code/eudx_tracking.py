import nibabel as nib
from nibabel import trackvis as tv

dname = "/home/eleftherios/Data/Dave_St-Amour/results/"
fodf_sh = dname + "fodf.nii.gz"
ftrk = dname + "whole_brain_wm_ants.trk"

img = nib.load(fodf_sh)
data = img.get_data()
affine = img.get_affine()

from dipy.reconst.shm import sh_to_sf
from dipy.reconst.odf import peaks_from_model

streams, hdr = tv.read(ftrk)

streamlines = [s[0] for s in streams]

from dipy.tracking.distances import approx_polygon_track
from dipy.tracking.metrics import length

streamlines = [approx_polygon_track(s, 0.2) for s in streamlines if length(s) > 100]

from dipy.viz import fvtk
from dipy.viz.colormap import line_colors

ren = fvtk.ren()

no = 10000

from dipy.segment.quickbundles import QuickBundles

qb = QuickBundles(streamlines[:no], 25, 20)

print(qb.total_clusters)

qb.remove_small_clusters(200)

print(qb.total_clusters)

streamline_ids = []

for c in qb.clustering:
    streamline_ids+=qb.label2tracksids(c)

final_streamlines = [streamlines[id] for id in streamline_ids]

fvtk.add(ren, fvtk.pretty_line(final_streamlines,
         line_colors(final_streamlines, 'boys_standard'), linewidth=2*0.15))

fvtk.show(ren)



