import numpy as np
import scipy.fftpack
import os
import h5py
import pylab


def getSampleTopLeftCorner(iMin,iMax,jMin,jMax):
    """ Function that genereates randomly a position between i,j intervals [iMin,iMax], [jMin,jMax]
    Args:
        iMin (int): the i minimum coordinate (i is the column-position of an array)
        iMax (int): the i maximum coordinate (i is the column-position of an array)
        jMin (int): the j minimum coordinate (j is the row-position of an array)
        jMax (int): the j maximum coordinate (j is the row-position of an array)
    Returns:
        [i,j] (tuple(int,int)): random integers such iMin<=i<iMax,jMin<=j<jMax,
    """ 
    ###write your function here
    x = np.random.randint(iMin,iMax)
    y = np.random.randint(jMin,jMax)
    return (x,y)

def getSampleImage(image, sampleSize, topLeftCorner):
    """ Function that extracts a sample of an image with a given size and a given position
    Args:
        image (numpy.array) : input image to be sampled
        sampleSize (tuple(int,int)): size of the sample
        topLeftCorner (tuple(int,int)): positon of the top left corner of the sample within the image
    Returns:
        sample (numpy.array): image sample
    """ 
    ###write your function here
    sample = image[topLeftCorner[0]:topLeftCorner[0]+sampleSize[0],topLeftCorner[1]:topLeftCorner[1]+sampleSize[1]]
    return sample

def getSamplePS(sample):
    """ Function that calculates the power spectrum of a image sample
    Args:
        sample (numpy.array): image sample
    Returns:
        samplePS (numpy.array): power spectrum of the sample. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """ 
    ###write your function here
    sampleFourier = np.fft.fft2(sample)
    translation = np.fft.fftshift(sampleFourier)
    samplePs = list(map(lambda num: num ** 2, np.abs(translation)))
    return samplePs

def getAveragePS(inputFileName, sampleSize, numberOfSamples):
    """ Function that estimates the average power spectrum of a image database
    Args:
        inputFileName (str) : Absolute pathway to the image database stored in the hdf5
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
        numberOfSamples
    Returns:
        averagePS (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see scipy.ffpack.fftshift)
    """ 
    ###write your function here
    averagePs = np.zeros((sampleSize[0],sampleSize[1]))

    with h5py.File(inputFileName,'r')as f:
        dataset = f['images']
        iMin = 0
        iMax = dataset.shape[1] - sampleSize[0]
        jMin = 0
        jMax = dataset.shape[2] - sampleSize[1]

        for i in range(numberOfSamples):
            img = f.get('images')[i%dataset.shape[0]]
            topLeftCorner = getSampleTopLeftCorner(iMin, iMax, jMin, jMax)
            sample = getSampleImage(img, sampleSize, topLeftCorner)
            imgPS = getSamplePS(sample)
            averagePs = np.array(averagePs) +np.array(imgPS)

    averagePs= averagePs / numberOfSamples
    return averagePs
   
def getRadialFreq(PSSize):
    """ Function that returns the Discrete Fourier Transform radial frequencies
    Args:
        psSize (tuple(int,int)): the size of the window to calculate the frequencies
    Returns:
        radialFreq (numpy.array): radial frequencies in crescent order
    """
    fx = np.fft.fftshift(np.fft.fftfreq(PSSize[0], 1./PSSize[0]))
    fy = np.fft.fftshift(np.fft.fftfreq(PSSize[1], 1./PSSize[1]))
    [X,Y] = np.meshgrid(fx,fy)
    R = np.sqrt(X**2+Y**2)
    return R

def getRadialPS(averagePS):
    """ Function that estimates the average power radial spectrum of a image database
    Args:
        averagePS (numpy.array) : average power spectrum of the database samples.
    Returns:
        averagePSRadial (numpy.array): average radial power spectrum of the database samples.
    """
    APS = np.array(averagePS)
    dimMax = max(APS.shape[0],APS.shape[1])
    PX = np.arange(-dimMax/2,dimMax/2)
    X,Y = np.meshgrid(PX,PX)
    rho,theta = cartToPolar(X,Y)
    rhoUnique = np.unique(rho)
    averagePSRadial = np.zeros(int(rhoUnique.shape[0]))

    for i,r in enumerate(rhoUnique):
        pixelX, pixelY = (rho==r).nonzero()
        samplesPs= APS[pixelX,pixelY]
        averagePSRadial[i] = np.nanmean(samplesPs)

    return averagePSRadial


def cartToPolar(X,Y):
    """ Function that convert Cartesian coordinate system to polar coordinates
    :param X:
    :param Y:
    :return:
    """
    rho = np.sqrt(X**2+Y**2)
    theta = np.arctan2(Y,X)
    return (rho,theta)

def getAveragePSLocal(inputFileName, sampleSize, gridSize):
    """ Function that estimates the local average power spectrum of a image database
    Args:
        inputFileName (str) : Absolute pathway to the image database stored in the hdf5
        sampleSize (tuple(int,int)): size of the samples that are extrated from the images
        gridSize (tuple(int,int)): size of the grid that define the borders of each local region
    Returns:
        averagePSLocal (numpy.array): average power spectrum of the database samples. The axis are shifted such the low frequencies are in the center of the array (see numpy.fft.fftshift)
    """ 
    ###write your function here
    averagePS = np.zeros((32*gridSize[0],32*gridSize[1]))
    averagePSLocal = np.reshape(averagePS,(gridSize[0],gridSize[1],32,32))
    numberOfSamples = 10000
    # print(averagePS.shape)
    # print(averagePSLocal.shape)
    # iMin,iMax = np.zeros(gridSize[0])
    # jMin,jMax = np.zeros(gridSize[1])

    with h5py.File(inputFileName, 'r')as f:
        dataset = f['images']

        for i in range(gridSize[0]):
            for j in range(gridSize[1]):
                iMin = np.floor(dataset.shape[1]/gridSize[0])*i
                iMax = np.floor(dataset.shape[1]/gridSize[0])*(i+1)-sampleSize[0]
                jMin = np.floor(dataset.shape[2]/gridSize[1])*j
                jMax = np.floor(dataset.shape[2]/gridSize[1])*(j+1)-sampleSize[1]

                for nums in range(numberOfSamples):
                    img = f.get('images')[nums % dataset.shape[0]]
                    topLeftCorner = getSampleTopLeftCorner(iMin, iMax, jMin, jMax)
                    sample = getSampleImage(img, sampleSize, topLeftCorner)
                    imgPS = getSamplePS(sample)
                    averagePSLocal[i,j] = np.array(averagePSLocal[i,j]) + np.array(imgPS)

                averagePSLocal[i, j] = averagePSLocal[i, j] / numberOfSamples

    return averagePSLocal
    
def makeAveragePSFigure(averagePS, figureFileName):
    """ Function that makes and save the figure with the power spectrum
    Args:
        averagePSLocal (numpy.array): the average power spectrum in an array of shape [sampleShape[0],sampleShape[1]
        figureFileName (str): absolute path where the figure will be saved
    """ 
    pylab.imshow(np.log(averagePS),cmap = "gray")
    pylab.contour(np.log(averagePS))
    pylab.axis("off")
    pylab.savefig(figureFileName)

def makeAveragePSRadialFigure(radialFreq,averagePSRadial,figureFileName):
    """ Function that makes and save the figure with the power spectrum
    Args:
        averagePS (numpy.array) : the average power spectrum
        averagePSRadial (numpy.array): the average radial power spectrum
        figureFileName (str): absolute path where the figure will be saved
    """ 
    pylab.figure()
    pylab.loglog(radialFreq,averagePSRadial,'.')
    pylab.xlabel("Frequecy")
    pylab.ylabel("Radial Power Spectrum")
    pylab.savefig(figureFileName)


def makeAveragePSLocalFigure(averagePSLocal,figureFileName,gridSize):
    """ Function that makes and save the figure with the local power spectrum
    Args:
        averagePSLocal (numpy.array): the average power spectrum in an array of shape [gridSize[0],gridSize[1],sampleShape[0],sampleShape[1]
        figureFileName (str): absolute path where the figure will be saved
        gridSize (tuple): size of the grid
    """ 
    pylab.figure()
    for i in range(gridSize[0]):
        for j in range(gridSize[1]):
            pylab.subplot(gridSize[0],gridSize[1],i*gridSize[1]+j+1)
            pylab.imshow(np.log(averagePSLocal[i,j]),cmap = "gray")
            pylab.contour(np.log(averagePSLocal[i,j]))
            pylab.axis("off")
    pylab.savefig(figureFileName)

def saveH5(fileName,dataName,numpyArray):
    """ Function that saves numpy arrays in a binary file h5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataName (str): the dataset name 
        numpyArray (numpy.array): the data to be saved
    """

    f = h5py.File(fileName, "w")
    f.create_dataset(dataName,data =numpyArray);
    f.close()


def readH5(fileName, dataName):
    """ Function that reads numpy arrays in a binary file hdf5
    Args:
        fileName (str): the path where the numpy array will be saved. The absolute path should be given. It should finish with '.hdf5'
        dataName (str): the dataset name 
    Return:
        numpyArray (numpy.array): the read data
    """
    f = h5py.File(fileName, "r")
    numpyArray = f[dataName][:]
    f.close()
    return numpyArray




