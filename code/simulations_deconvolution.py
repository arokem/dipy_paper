import warnings
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           run_module_suite)

from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel,
                                   odf_sh_to_sharp)
from dipy.reconst.odf import peak_directions
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.shm import sf_to_sh, sh_to_sf, QballModel, CsaOdfModel

angle = [90, 60, 50, 45]
SNRs = [100, 30, 10]

for snr in SNRs:
    for ang in angle:
        print("[Crossing angle, SNR]", ang, snr)
        SNR = snr
        S0 = 100

        _, fbvals, fbvecs = get_data('small_64D')

        bvals = np.load(fbvals)
        bvecs = np.load(fbvecs)

        gtab = gradient_table(bvals, bvecs)
        mevals = np.array(([0.0015, 0.0003, 0.0003],
                           [0.0015, 0.0003, 0.0003]))

        S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (ang, 0)],
                                 fractions=[50, 50], snr=SNR)

        sphere = get_sphere('symmetric724')
        sphere2 = get_sphere('symmetric362')

        mevecs = [all_tensor_evecs(sticks[0]).T,
                  all_tensor_evecs(sticks[1]).T]

        odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

        response = (np.array([0.0015, 0.0003, 0.0003]), S0)

        csd = ConstrainedSphericalDeconvModel(gtab, response)

        csd_fit = csd.fit(S)

        fod = csd_fit.odf(sphere)

        e1 = 15.0
        e2 = 3.0
        ratio = e2 / e1

        csd = ConstrainedSDTModel(gtab, ratio, None)

        csd_fit = csd.fit(S)
        fodf = csd_fit.odf(sphere)

        csa = CsaOdfModel(gtab, sh_order=8)
        csafit = csa.fit(S)
        csa_odf = csafit.odf(sphere)
        csa_sh = sf_to_sh(csa_odf, sphere, sh_order=8, basis_type=None)    
        fodf_sh = odf_sh_to_sharp(csa_sh, sphere2, basis=None, ratio=3 / 15.,
                                  sh_order=8, lambda_=0.1, tau=0.1)
        fodf_csa = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type=None) 

        odf_sh = sf_to_sh(odf_gt, sphere, sh_order=8, basis_type=None)    
        # Because the ground truth ODF and the CSA ODFs are actually properly normalized ODF
        # with the r^2 in the PDF integral
        # the default q-ball sharp parameters are not optimal. One must then decrease the
        # constrained-regularization weight lambda_.
        fodf_sh = odf_sh_to_sharp(odf_sh, sphere2, basis=None, ratio=3 / 15.,
                                  sh_order=8, lambda_=0.1, tau=0.1)

        fodf_gt = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type=None)

        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add( r, fvtk.sphere_funcs( np.vstack((fod, fodf, odf_gt, fodf_gt, csa_odf, fodf_csa)), sphere ) )
        fvtk.show( r )

