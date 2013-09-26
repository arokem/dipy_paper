import numpy as np
import matplotlib.pyplot as plt
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.data import get_data, dsi_voxels
from dipy.reconst.dsi import (
    DiffusionSpectrumModel, DiffusionSpectrumDeconvModel)
import dipy.reconst.dti as dti
from dipy.reconst.odf import gfa, peak_directions
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from dipy.sims.voxel import (
    SingleTensor, MultiTensor, multi_tensor_odf, all_tensor_evecs,
    add_noise, single_tensor, sticks_and_ball)
from dipy.viz import fvtk

sphere = get_sphere('symmetric724')
mevals0 = np.array(([0.0017, 0.0003, 0.0003],
                    [0.0017, 0.0003, 0.0003]))

mevals1 = np.array(([0.0017, 0.0003, 0.0003],
                    [0.0017, 0.0003, 0.0003],
                    [0.0017, 0.0003, 0.0003]))


table_path = get_data('dsi515btable')
gtab = gradient_table(table_path)

#####################################


S_0, sticks_0 = MultiTensor(
    gtab, mevals0, S0=100, angles=[(0, 0), (60, 0)], fractions=[50, 50], snr=None)
S_1, sticks_0 = MultiTensor(
    gtab, mevals1, S0=100, angles=[(0, 0), (90, 0), (90, 90)], fractions=[33, 33, 34], snr=None)

gridsize=35
c=gridsize//2

sphere = sphere.subdivide(3)

dsmodel_dec = DiffusionSpectrumDeconvModel(gtab, qgrid_size=gridsize, r_start=0.4*c, r_end=0.7*c)
dsmodel = DiffusionSpectrumModel(gtab, qgrid_size=gridsize, r_start=0.4*c, r_end=0.7*c)
pdf0 = dsmodel.fit(S_0).pdf()
pdf1 = dsmodel.fit(S_1).pdf()
pdf0=pdf0/pdf0.sum()
pdf1=pdf1/pdf1.sum()
odf0=dsmodel.fit(S_0).odf(sphere)
odf1=dsmodel.fit(S_1).odf(sphere)

pdf0_dec = dsmodel_dec.fit(S_0).pdf()
pdf1_dec = dsmodel_dec.fit(S_1).pdf()
pdf0_dec=pdf0_dec/pdf0_dec.sum()
pdf1_dec=pdf1_dec/pdf1_dec.sum()
odf0_dec=dsmodel_dec.fit(S_0).odf(sphere)
odf1_dec=dsmodel_dec.fit(S_1).odf(sphere)


np.save('odf_dsi.npy',odf0)
np.save('odf_dsi_deconv.npy',odf0_dec)

ODF=np.vstack((odf0,odf1))
ODF_dec=np.vstack((odf0_dec,odf1_dec))

ODFs=np.vstack((ODF,ODF_dec))

r=fvtk.ren()
fvtk.add(r, fvtk.sphere_funcs(ODFs,sphere,colormap='jet'))
fvtk.show(r)


# fig=plt.figure()
# ax1=fig.add_subplot(2,2,1)
# ax1.set_title('first')
# ind=ax1.imshow(pdf0[c,:,:],interpolation='nearest') #Needs to be in row,col order
# plt.colorbar(ind)
# ax2=fig.add_subplot(2,2,2)
# ax2.set_title('second')
# ind=ax2.imshow(pdf0[:,c,:],interpolation='nearest') #Needs to be in row,col order
# plt.colorbar(ind)
# ax3=fig.add_subplot(2,2,3)
# ax3.set_title('third')
# ind=ax3.imshow(pdf0[:,:,c],interpolation='nearest') #Needs to be in row,col order
# plt.colorbar(ind)
# plt.show()
# fig=plt.figure()
# ax1=fig.add_subplot(2,2,1)
# ax1.set_title('first')
# ind=ax1.imshow(pdf1[c,:,:],interpolation='nearest') #Needs to be in row,col order
# plt.colorbar(ind)
# ax2=fig.add_subplot(2,2,2)
# ax2.set_title('second')
# ind=ax2.imshow(pdf1[:,c,:],interpolation='nearest') #Needs to be in row,col order
# plt.colorbar(ind)
# ax3=fig.add_subplot(2,2,3)
# ax3.set_title('third')
# ind=ax3.imshow(pdf1[:,:,c],interpolation='nearest') #Needs to be in row,col order
# plt.colorbar(ind)
# plt.show()