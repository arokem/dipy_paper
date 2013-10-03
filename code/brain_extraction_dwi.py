"""
===================================
Brain segmentation with median_otsu
===================================

We show how to extract brain information and mask from a b0 image using dipy's
segment.mask module.

First import the necessary modules:
"""

import numpy as np
import nibabel as nib

"""
Download and read the data for this tutorial.

The scil_b0 dataset contains different data from different companies and models.
For this example, the data comes from a 3 tesla GE MRI.
"""

# from dipy.data import fetch_scil_b0, read_scil_b0
# fetch_scil_b0()
# img = read_scil_b0()
# data = np.squeeze(img.get_data())

# from dipy.data import fetch_taiwan_ntu_dsi, read_taiwan_ntu_dsi
# img, gtab = read_taiwan_ntu_dsi()

from dipy.data import fetch_sherbrooke_3shell, read_sherbrooke_3shell
fetch_sherbrooke_3shell()
img, gtab = read_sherbrooke_3shell()
data = img.get_data()[..., 0]

"""
``img`` contains a nibabel Nifti1Image object. Data is the actual brain data as
a numpy ndarray.

Segment the brain using dipy's mask module.

``median_otsu`` returns the segmented brain data and a binary mask of the brain.
It is possible to fine tune the parameters of ``median_otsu`` (``median_radius``
and ``num_pass``) if extraction yields incorrect results but the default
parameters work well on most volumes. For this example, default parameters (4,
4) will be used.
"""

from dipy.segment.mask import median_otsu
b0_mask, mask = median_otsu(data, 2, 4)

"""
Saving the segmentation results is very easy using nibabel. We need the b0_mask,
and the binary mask volumes. The affine matrix which transform the image's
coordinates to the world coordinates is also needed. Here, we choose to save
both images in float32.
"""

mask_img = nib.Nifti1Image(mask.astype(np.float32), img.get_affine())
b0_img = nib.Nifti1Image(b0_mask.astype(np.float32), img.get_affine())

fname = 'ge_3t'
nib.save(mask_img, fname + '_binary_mask.nii.gz')
nib.save(b0_img, fname + '_mask.nii.gz')

"""
Quick view of the results middle slice using matplotlib.
"""

import matplotlib.pyplot as plt

sli = 30 #data.shape[2] / 2
# plt.figure('Brain segmentation')
# plt.subplot(1, 2, 1).set_axis_off()
# plt.imshow(data[:, :, sli], cmap='gray')
# plt.subplot(1, 2, 2).set_axis_off()
# plt.imshow(b0_mask[:, :, sli], cmap='gray')
#plt.savefig('median_otsu.png')

from skimage import exposure
im = data[:, :, sli]

def rescale_intensity(im, low, high):

	p2 = np.percentile(im, low)
	p98 = np.percentile(im, high)
	im2 = exposure.rescale_intensity(im, in_range=(p2, p98))
	return im2

# im2 = rescale_intensity(data[:, :, sli], 0, 100)
# im3 = rescale_intensity(b0_mask[:, :, sli], 0, 100)


im2 = rescale_intensity(data[:, :, sli], 5, 99)
im3 = rescale_intensity(b0_mask[:, :, sli], 5, 99)

origin='upper'
cmap = 'gray'
plt.figure('Brain segmentation')
plt.subplot(1, 2, 1).set_axis_off()
plt.imshow(np.swapaxes(im2, 0, 1), cmap=cmap, origin='upper', interpolation='nearest')
plt.subplot(1, 2, 2).set_axis_off()
plt.imshow(np.swapaxes(im3, 0, 1), cmap=cmap, origin='upper', interpolation='nearest')

"""
.. figure:: median_otsu.png
   :align: center

   **An application of median_otsu for brain segmentation**.

``median_otsu`` can also automatically crop the outputs to remove the largest
possible number of background voxels. This makes outputted data significantly
smaller.  auto cropping in ``median_otsu`` is activated by setting the
``autocrop`` parameter to True.
"""

b0_mask_crop, mask_crop = median_otsu(data, 4, 4, autocrop=True)

"""
Saving cropped data using nibabel as demonstrated previously.
"""

mask_img_crop = nib.Nifti1Image(mask_crop.astype(np.float32), img.get_affine())
b0_img_crop = nib.Nifti1Image(
    b0_mask_crop.astype(np.float32), img.get_affine())
nib.save(mask_img_crop, fname + '_binary_mask_crop.nii.gz')
nib.save(b0_img_crop, fname + '_mask_crop.nii.gz')


