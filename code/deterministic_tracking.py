import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.shm import CsaOdfModel, QballModel, normalize_data
from dipy.reconst.odf import peaks_from_model
from dipy.tracking.metrics import length
from dipy.segment.mask import hist_mask

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

mask = hist_mask(data[..., 0])

csamodel = CsaOdfModel(gtab, 4, smooth=0.006)
sphere = get_sphere('symmetric724')

# Peaks_from_model is used to calculate properties of the ODFs 
# and return for example the peaks and their indices, or GFA 
# which is similar to FA but for ODF based models. This function 
# mainly needs a reconstruction model, the data and a sphere as input. 
# The sphere is an object that represents the spherical 
# discrete grid where the ODF values will be evaluated.
print('Computing CSA ODF peaks...')
csapeaks = peaks_from_model(model=csamodel,
                            data=data,
                            sphere=sphere,
                            relative_peak_threshold=.5,
                            min_separation_angle=45,
                            mask=mask,
                            return_odf=False,
                            normalize_peaks=True)

from dipy.tracking.eudx import EuDX


fa_file = 'fa.nii.gz'
fa_img = nib.load(fa_file)
FA = fa_img.get_data()

# Seed only in highly anisotropic areas
FA_masked = FA > 0.3
indices = np.nonzero(FA_masked)
seeds = np.array(indices).T

# Full brain deterministic tracking on the CSA peaks
# mask from FA > 0.1
# step size of 0.5 voxel
# angular threshold of 60 degrees
eu = EuDX(FA.astype(np.float64),
          csapeaks.peak_indices[..., 0],
          #seeds=seeds,
          seeds=100000,
          odf_vertices=sphere.vertices,
          a_low=0.1,
          step_sz=0.2,
          ang_thr=60.)

print('Streamline tracking from %d longer than 10 mm ...'% seeds.shape[0])
csa_streamlines = [streamline for streamline in eu if length(streamline) > 10]

print 'Saving fibers in trk format...'
import nibabel as nib
hdr = nib.trackvis.empty_header()
hdr['voxel_size'] = (2., 2., 2.)
hdr['voxel_order'] = 'LAS'
hdr['dim'] = csapeaks.gfa.shape[:3]

csa_streamlines_trk = ((sl, None, None) for sl in csa_streamlines)
csa_sl_fname = 'csa_streamlines.trk'
nib.trackvis.write(csa_sl_fname, csa_streamlines_trk, hdr, points_space='voxel')




