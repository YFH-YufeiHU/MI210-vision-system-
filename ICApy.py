import numpy as np
import pylab
from sklearn.decomposition import FastICA
import PSpy
import h5py


def getICAInputData(inputFileName, sampleSize,nSamples):
    """ Function that samples the input directory for later to be used by FastICA
    Args:
    inputFileName(str):: Absolute pathway to the image database hdf5 file
    sampleSize (tuple(int,int)): size of the samples that are extrated from the images
    nSamples(int): number of samples that should be taken from the database
    Returns:
    X(numpy.array)nSamples, sampleSize
    """
    #Write your function here
    source = np.zeros((nSamples,sampleSize[0],sampleSize[1]))
    print(source.shape)
    with h5py.File(inputFileName,'r') as f:
        dataset = f['images']
        iMin = 0
        iMax = dataset.shape[1] - sampleSize[0]
        jMin = 0
        jMax = dataset.shape[2] - sampleSize[1]

        for i in range(nSamples):
            img = f.get('images')[i%dataset.shape[0]]
            topLeftCorner = PSpy.getSampleTopLeftCorner(iMin,iMax,jMin,jMax)
            newImg = PSpy.getSampleImage(img,sampleSize,topLeftCorner)
            source[i] = newImg
    print(source.shape)

    return source

def preprocess(X):
    """Function that preprocess the data to be fed to the ICA algorithm
    Args:
    X(numpy array): input to be preprocessed
    Returns:
    X(numpy.array)
    """
    for i in range(X.shape[0]):
        X[i,:,:]=X[i,:,:]-np.mean(X[i])
    X = np.reshape(X,(X.shape[0], X.shape[1] * X.shape[2]))
    print(X.shape)
    return X


def getIC(X):
    """Function that estimates the principal components of the data
    Args:
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the independent sources of the data
    """
    ICA = FastICA(algorithm='parallel',whiten=True,tol=1e-1,max_iter=2000)
    # S_ = ICA.fit_transform(X) #Fit the model and recover the sources from X
    ICA.fit(X)
    S_ = ICA.components_ #The linear operator to apply to the data to get the independent sources.
    return S_


def estimateSources(W,X):
    """Function that estimates the independent sources of the data
    Args:
    W(numpy.array):The matrix of the independent components
    X(numpy.array):preprocessed data
    Returns:
    S(numpy.array) the matrix of the sources of X
    """



def estimateSourcesKurtosis(S):
    """Function  that estimates the kurtosis of a set of multivariate random variables
    Args: 
    S(numpy array): random variable (n-data points of k-size)
    Returns:
    K (numpy.array): kurtosis of each data point (size n,1) 
    """
    #Write your function here
    print("You should define the functione stimateSourcesKurtosis")

def makeKurtosisFigure(S,figureFileName):
    #Write your function here
    print("You should define the functione makeKurtosisFigure")
        
def makeIdependentComponentsFigure(W, sampleSize,figureFileName): 
    W = W.reshape([-1,]+sampleSize)
    pylab.figure()
    for i in range(W.shape[0]):
        pylab.subplot(sampleSize[0],sampleSize[1],i+1)
        pylab.imshow(W[i],cmap = 'gray')
        pylab.axis("off")
    pylab.savefig(figureFileName)
