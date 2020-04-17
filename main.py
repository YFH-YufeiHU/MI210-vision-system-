import PSpy
import WhiteningFilterspy
import ICApy
import numpy as np

#defining read and write directories


inputFileName = "G:/MI210_TD_2/autonomousVehicle.hdf5"
resultsDirectory = "G:/MI210_TD_2/result/"



#defining images and output file names
averagePSResultsFileName = resultsDirectory + "averagePS.hdf5"
averagePSFigureFileName = resultsDirectory + "averagePS.png"

averagePSLocalResultsFileName = resultsDirectory + "averagePSLocal.hdf5"
averagePSLocalFigureFileName = resultsDirectory + "averagePSLocal.png"

whiteningFiltersFigureFileName = resultsDirectory + "whiteningFilters.png"
whiteningFiltersResultsFileName = resultsDirectory + "whiteningFilters.hdf5"

averagePSRadialResultsFileName = resultsDirectory + "averageOSRadial.hdf5"
averagePSRadialFigureFileName = resultsDirectory + "averagePSRadial.png"

ICResultsFileName = resultsDirectory + "IC.hdf5"
ICFigureFileName = resultsDirectory + "IC.png"

ICAActivationsResultsFileName = resultsDirectory +"ActivationsICA.hdf5"
ICAActivationsSparsenessFigureFileName = resultsDirectory +"ActivationsICA.png"

#defining some parameters
sampleSizePS = [32,32] #image sample size for the power spectrum
gridSize = [3,3]
numberOfSamplesPS = 10000 #number of samples from the dataset for estimating PS



sampleSizeICA = [12,12] #image sample size for the ICA
numberOfSamplesICA = 50000 #number of samples from the dataset for making ICA


# #Question 2
# averagePS = PSpy.getAveragePS(inputFileName, sampleSizePS,numberOfSamplesPS)
# PSpy.saveH5(averagePSResultsFileName,'averagePS',averagePS)
# PSpy.makeAveragePSFigure(averagePS, averagePSFigureFileName)
#
# ##Question 3
# averagePSRadial = PSpy.getRadialPS(averagePS)
# radialFreq = PSpy.getRadialFreq(averagePS.shape)
# PSpy.saveH5(averagePSRadialResultsFileName,'averagePSRadial',averagePSRadial)
# PSpy.makeAveragePSRadialFigure(np.unique(radialFreq),averagePSRadial, averagePSRadialFigureFileName)
#
# #Question 4
# averagePSLocal = PSpy.getAveragePSLocal(inputFileName, sampleSizePS, gridSize)
# PSpy.saveH5(averagePSLocalResultsFileName,'averagePS',averagePSLocal)
# PSpy.makeAveragePSLocalFigure(averagePSLocal, averagePSLocalFigureFileName,gridSize)

#Question 5
# averagePS = PSpy.readH5(averagePSResultsFileName,'averagePS')
#
# maxPS = np.max(averagePS)
# noiseVarianceList = [maxPS*10**(-9),maxPS*10**(-8),maxPS*10**(-7),maxPS*10**(-6)] #if you do not see anything interesting you can change this values
#
# whiteningFilters = []
# for noiseVariance in noiseVarianceList:
#    whiteningFilters.append(WhiteningFilterspy.getPowerSpectrumWhiteningFilter(averagePS,noiseVariance))
#
# PSpy.saveH5(whiteningFiltersResultsFileName,'whiteningFilters',np.array(whiteningFilters))
# WhiteningFilterspy.makeWhiteningFiltersFigure(whiteningFilters,whiteningFiltersFigureFileName)

#Question 6
# X = ICApy.getICAInputData(inputFileName, sampleSizeICA, numberOfSamplesICA)
# X = ICApy.preprocess(X)
# W = ICApy.getIC(X)
#
# PSpy.saveH5(ICResultsFileName,'IC',W)
# ICApy.makeIdependentComponentsFigure(W,sampleSizeICA, ICFigureFileName)

#Question 7
A = ICApy.estimateActivations(W)
sparsenessMeasure = ICApy.estimateSparseness(A)
PSpy.saveH5(ICAActivationsResultsFileName,'A',A)
ICApy.makeSparsenessMeasureFigure(A, ICAActivationsSparsenessFigureFileName)
