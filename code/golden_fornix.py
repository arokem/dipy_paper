import numpy as np
from nibabel import trackvis as tv
from dipy.tracking import metrics as tm
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_data
from dipy.viz import fvtk

fname = get_data('fornix')
streams, hdr = tv.read(fname)

streamlines = [i[0] for i in streams]

ren = fvtk.ren()

ren.SetBackground(1., 1., 1)

fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.banana))

fvtk.show(ren)
