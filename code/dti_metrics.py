import numpy as np
import nibabel as nib
from dipy.data import fetch_stanford_hardi, read_stanford_hardi, get_sphere
from dipy.reconst.dti import TensorModel
from dipy.viz import fvtk

fetch_stanford_hardi()
img, gtab = read_stanford_hardi()
data = img.get_data()

from dipy.segment.mask import median_otsu
data, mask = median_otsu(data, 6, 5)

data_small = data[:, :, 38] #data[20:50-10, 55+10:85, 38]

dtimodel = TensorModel(gtab)

dtifit = dtimodel.fit(data_small)

fa = dtifit.fa

md = dtifit.md

from dipy.reconst.dti import color_fa

cfa = color_fa(fa, dtifit.evecs)

plan = dtifit.planarity

sph = dtifit.sphericity

sph[mask[:, :, 38]==0] = 0

rd = dtifit.rd

ad = dtifit.ad

trace = dtifit.trace

import matplotlib.pyplot as plt

plt.figure('Brain segmentation')
plt.subplot(2, 3, 1).set_axis_off()
plt.imshow(fa, cmap='gray')
plt.title('FA')
plt.subplot(2, 3, 2).set_axis_off()
plt.imshow(md, cmap='gray')
plt.title('MD')
plt.subplot(2, 3, 3).set_axis_off()
plt.imshow(cfa)
plt.title('DEC')
plt.subplot(2, 3, 4).set_axis_off()
plt.imshow(rd, cmap='gray')
plt.title('Radial diffusivity')
plt.subplot(2, 3, 5).set_axis_off()
plt.imshow(sph, cmap='gray')
plt.title('Sphericity')
plt.subplot(2, 3, 6).set_axis_off()
plt.imshow(plan, cmap='gray')
plt.title('Planarity')






