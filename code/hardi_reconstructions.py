import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, sh_to_sf
from dipy.viz import fvtk
from dipy.reconst.odf import gfa

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()
sphere = get_sphere('symmetric724')
sphere2 = get_sphere('symmetric362')

data_small  = data[20:50,55:85, 38:39]

order = 4
qballmodel = QballModel(gtab, order, smooth=0.006)
print 'Computing the qball ODFs...'
qballfit = qballmodel.fit(data_small)
qballodfs = qballfit.odf(sphere)

ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(qballodfs, sphere, colormap='jet'))
fvtk.show(ren)
#print('Saving illustration as qballodfs.png')
#fvtk.record(ren, n_frames=1, out_path='qballodfs.png', size=(600, 600))
fvtk.clear(ren)

# min-max normalize ODFs
print 'Min-max normalizing and visualizing...'
minmax_odfs = (qballodfs - np.min(qballodfs, -1)[..., None]) / (np.max(qballodfs, -1)[..., None] - np.min(qballodfs, -1)[..., None])

fvtk.add(ren, fvtk.sphere_funcs(minmax_odfs, sphere, colormap='jet'))
fvtk.show(ren)
#print('Saving illustration as qball_minmax_odfs.png')
#fvtk.record(ren, n_frames=1, out_path='qball_minmax_odfs.png', size=(600, 600))
fvtk.clear(ren)

csamodel = CsaOdfModel(gtab, order, smooth=0.006)
print 'Computing the CSA ODFs...'
csafit = csamodel.fit(data_small)
csaodfs = csafit.odf(sphere)

print 'Visualizing CSA ODFs...'
ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(csaodfs, sphere, colormap='jet'))
fvtk.show(ren)
#print('Saving illustration as csa_odfs.png')
#fvtk.record(ren, n_frames=1, out_path='csa_odfs.png', size=(600, 600))
fvtk.clear(ren)

# notice the negative values produced
print 'Visualizing CSA ODFs without negative values...'
ren = fvtk.ren()
fvtk.add(ren, fvtk.sphere_funcs(np.clip(csaodfs, 0, np.max(csaodfs, -1)[..., None]), sphere, colormap='jet'))
fvtk.show(ren)
#print('Saving illustration as csa_odfs_positive.png')
#fvtk.record(ren, n_frames=1, out_path='csa_odfs_positive.png', size=(600, 600))
fvtk.clear(ren)

from dipy.segment.mask import hist_mask

mask = hist_mask(data[...,0])

csafit = csamodel.fit(data, mask)

nib.save(nib.Nifti1Image(csafit.shm_coeff.astype('float32'), img.get_affine()), 'csa_odf_sh.nii.gz')