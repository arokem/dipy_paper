import numpy as np
from nibabel import trackvis as tv
from dipy.tracking import metrics as tm
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_data
from dipy.viz import fvtk
from dipy.tracking.distances import approx_polygon_track

fname = get_data('fornix')
dname = '/home/eleftherios/Data/fornixForElef/'
fname1 = dname + 'Track0209L.trk'
fname2 = dname + 'Track0209R.trk'
fname3 = dname + 'Track0326L.trk'
fname4 = dname + 'Track0326R.trk'
fname5 = dname + 'boo.trk'
# Track0326L.trk
# Track0326R.trk

def trk2streamlines(fname, thr=0.2):
    streams, hdr = tv.read(fname)
    streamlines = [i[0] for i in streams]
    streamlines = [approx_polygon_track(s, thr) for s in streamlines]
    return streamlines

T1 = trk2streamlines(fname1)
T2 = trk2streamlines(fname2)
T3 = trk2streamlines(fname3)
T4 = trk2streamlines(fname4)
T5 = trk2streamlines(fname5)

ren = fvtk.ren()

ren.SetBackground(1., 1., 1)

fvtk.add(ren, fvtk.streamtube(T1, fvtk.colors.red))
fvtk.add(ren, fvtk.streamtube(T2, fvtk.colors.blue))
# fvtk.add(ren, fvtk.streamtube(T3, fvtk.colors.red))
# fvtk.add(ren, fvtk.streamtube(T4, fvtk.colors.green))
# fvtk.add(ren, fvtk.streamtube(T5, fvtk.colors.blue_light))

fvtk.show(ren)
