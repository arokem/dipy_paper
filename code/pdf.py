
dname = "/home/eleftherios/Data/"
fname = dname + "crossing60.npy"
fname2 = dname + "crossing60deconv.npy"
fname3 = "odf_dsi.npy"
fname4 = "odf_dsi_deconv.npy"


import numpy as np
from dipy.viz import fvtk

pdf = np.load(fname)
pdf2 = np.load(fname2)

odf = np.load(fname3)
odf2 = np.load(fname4)


from scipy.ndimage import zoom

pdf = zoom(pdf, 2, order=3)

pdf2 = zoom(pdf2, 2, order=3)

ren = fvtk.ren()

ren.SetBackground(1, 1, 1.)

def volume_new(data_matrix):
    from dipy.viz.fvtk import vtk

    data_matrix = data_matrix.copy()

    data_matrix = np.interp(data_matrix, [data_matrix.min(), data_matrix.max()], [0, 255])
    data_matrix = data_matrix.astype('uint8')

    # For VTK to be able to use the data, it must be stored as a VTK-image. This can be done by the vtkImageImport-class which
    # imports raw data and stores it.
    dataImporter = vtk.vtkImageImport()
    # The preaviusly created array is converted to a string of chars and imported.
    data_string = data_matrix.tostring()
    dataImporter.CopyImportVoidPointer(data_string, len(data_string))
    # The type of the newly imported data is set to unsigned char (uint8)
    dataImporter.SetDataScalarTypeToUnsignedChar()
    # Because the data that is imported only contains an intensity value (it isnt RGB-coded or someting similar), the importer
    # must be told this is the case.
    dataImporter.SetNumberOfScalarComponents(1)
    # The following two functions describe how the data is stored and the dimensions of the array it is stored in. For this
    # simple case, all axes are of length 75 and begins with the first element. For other data, this is probably not the case.
    # I have to admit however, that I honestly dont know the difference between SetDataExtent() and SetWholeExtent() although
    # VTK complains if not both are used.
    I, J, K = data_matrix.shape
    print(data_matrix.shape)

    dataImporter.SetDataExtent(0, I-1, 0, J-1, 0, K-1)
    dataImporter.SetWholeExtent(0, I-1, 0, J-1, 0, K-1)

    # The following class is used to store transparencyv-values for later retrival. In our case, we want the value 0 to be
    # completly opaque whereas the three different cubes are given different transperancy-values to show how it works.
    alphaChannelFunc = vtk.vtkPiecewiseFunction()
    alphaChannelFunc.AddPoint(0, 0.0)
    alphaChannelFunc.AddPoint(30, 0.02)
    alphaChannelFunc.AddPoint(60, 0.06)
    alphaChannelFunc.AddPoint(100, 0.18)
    alphaChannelFunc.AddPoint(255, 0.20)

    # This class stores color data and can create color tables from a few color points. For this demo, we want the three cubes
    # to be of the colors red green and blue.
    colorFunc = vtk.vtkColorTransferFunction()
    colorFunc.AddRGBPoint(30,  0.0, 0.0, 1.0)
    colorFunc.AddRGBPoint(60, 0.0, 1.0, 0.0)
    colorFunc.AddRGBPoint(100, 1.0, 0.0, 0.0)
    colorFunc.AddRGBPoint(255, 1.0, 1.0, 0.0)

    # The preavius two classes stored properties. Because we want to apply these properties to the volume we want to render,
    # we have to store them in a class that stores volume prpoperties.
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(colorFunc)
    volumeProperty.SetScalarOpacity(alphaChannelFunc)
    volumeProperty.SetInterpolationTypeToLinear()

    # This class describes how the volume is rendered (through ray tracing).
    compositeFunction = vtk.vtkVolumeRayCastCompositeFunction()
    # We can finally create our volume. We also have to specify the data for it, as well as how the data will be rendered.
    volumeMapper = vtk.vtkVolumeRayCastMapper()
    volumeMapper.SetVolumeRayCastFunction(compositeFunction)
    volumeMapper.SetInputConnection(dataImporter.GetOutputPort())

    # The class vtkVolume is used to pair the preaviusly declared volume as well as the properties to be used when rendering that volume.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)
    return volume


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


# opacitymap, colormap = opac_color(pdf)

# vol = fvtk.volume(pdf, maptype=0, opacitymap=opacitymap, colormap=colormap)

# opacitymap2, colormap2 = opac_color(pdf2)

# vol2 = fvtk.volume(pdf2, maptype=0, opacitymap=opacitymap, colormap=colormap)

# vol2.SetPosition(25 * 4, 0, 0)

# fvtk.add(ren, vol)
# fvtk.add(ren, vol2)


vol = volume_new(pdf)

vol2 = volume_new(pdf2)

vol2.SetPosition(25 * 4, 0, 0)

# fvtk.add(ren, vol)
# fvtk.add(ren, vol2)

from dipy.data import get_sphere

sphere = get_sphere('symmetric724').subdivide(3)

fvtk.add(ren, fvtk.sphere_funcs(odf, sphere))

# fvtk.add(ren, fvtk.sphere_funcs(odf2, sphere))

fvtk.show(ren, size=(800, 800))
