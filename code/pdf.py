
dname = "/home/eleftherios/Data/"
fname = dname + "crossing60.npy"
fname2 = dname + "crossing60deconv.npy"

import numpy as np
from dipy.viz import fvtk

pdf = np.load(fname)
pdf2 = np.load(fname2)

from scipy.ndimage import zoom

pdf = zoom(pdf, 2, order=3)

pdf2 = zoom(pdf2, 2, order=3)

ren = fvtk.ren()

ren.SetBackground(1, 1, 1.)

def opac_color(pdf):

    vol = pdf.copy()

    vol = np.interp(vol, [vol.min(), vol.max()], [0, 255])
    vol = vol.astype('uint8')

    bin, res = np.histogram(vol.ravel())
    res2 = np.interp(res, [vol.max()/15, vol.max()], [0, 1])
    opacitymap = np.vstack((res, res2)).T
    opacitymap = opacitymap.astype('float32')

    zer = np.zeros(res2.shape)
    colormap = np.vstack((res, res2, zer, zer+0.5)).T #res2[::-1]
    colormap = colormap.astype('float32')

    return opacitymap, colormap


opacitymap, colormap = opac_color(pdf)

vol = fvtk.volume(pdf, maptype=1, opacitymap=opacitymap, colormap=colormap)

opacitymap2, colormap2 = opac_color(pdf2)

vol2 = fvtk.volume(pdf2, maptype=1, opacitymap=opacitymap, colormap=colormap)

vol2.SetPosition(25 * 2, 0, 0)

fvtk.add(ren, vol)
fvtk.add(ren, vol2)

fvtk.show(ren)
