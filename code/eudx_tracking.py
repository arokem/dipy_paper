import nibabel as nib
from nibabel import trackvis as tv

dname = "/home/eleftherios/Data/Dave_St-Amour/results/"
#fodf_sh = dname + "fodf.nii.gz"
ftrk = dname + "whole_brain_wm_ants.trk"

# img = nib.load(fodf_sh)
# data = img.get_data()
# affine = img.get_affine()

from dipy.viz import fvtk
ren = fvtk.ren()

"""
T1
"""
fT1 = dname + "t1_brain_on_b0_brain_nonlinear.nii.gz"
img = nib.load(fT1)
data_T1 = img.get_data()
affine_T1 = img.get_affine()

I, J, K = data_T1.shape

fvtk.add(ren, fvtk.slicer(data_T1, plane_i=[I / 2], plane_j=[J / 2], plane_k=[K / 2]))

""" Load Streamlines
"""

from dipy.reconst.shm import sh_to_sf
from dipy.reconst.odf import peaks_from_model

streams, hdr = tv.read(ftrk)

streamlines = [s[0] for s in streams]

print(len(streamlines))

from dipy.tracking.vox2track import track_counts


""" Create intersection roi
"""

roi_width = 10

roi = np.zeros(data_T1.shape)
roi[90-roi_width:90+roi_width, 122-roi_width:122+roi_width, 66-roi_width:66+roi_width] = 1

tcs,tes = track_counts(streamlines, roi.shape, (1,1,1), True)

#find volume indices of mask's voxels
roiinds=np.where(roi==1)

#make it a nice 2d numpy array (Nx3)
roiinds=np.array(roiinds).T

def filter_roi_tracks(tracks, roiinds, tes):
    """ bring the tracks from the roi region and their indices
    """
    cnt=0
    sinds=[]
    for vox in roiinds:
        try:
            sinds+=tes[tuple(vox)]
            cnt+=1
        except:
            pass
    return [tracks[i] for i in list(set(sinds))], list(set(sinds))

streamlines, sinds = filter_roi_tracks(streamlines, roiinds, tes)

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
         line_colors(streamlines), linewidth=2*0.15))





fvtk.show(ren)



