"""
Needs to be run inside ipython!
"""

import preSVM as pre
import os as os
import nifti as nf

## Globals needed to create the SVMLIB data for a run
## =================================================
svmBaseName = 'rPRC_block_svmData.txt'

roiVmr = 'rPRC_block.nii'

trainLab1 = 'TESTSTU'
trainLab2 = 'TESTNSU'
testLab1 = 'TESTSTU'
testLab2 = 'TESTNSU'

from wordSceneFiles import *
	# Creates 4 lists containing the nii and labelList files:
	# - wordLab, wordNii, sceneNii,sceneLab

os.chdir('/Users/type/Lab/RWCR/bv_AR/mvpa_data/')
niiFiles, labFiles = pre.nii_LabelListMatch('.')

trainLabsFiles = sceneLab
trainNiiFiles  = sceneNii
testLabsFiles  = wordLab
testNiiFiles   = wordNii
## =================================================
## 

from time import localtime, strftime
svmBaseName = strftime("%a%d%b%Y_%H-%M-%S", localtime()) + svmBaseName  
	# Add a timestamp to the SVM file name, preventing accidental
	# overwriting or modification.

print('Creating training data:')
for labF, niiF in zip(trainLabsFiles, trainNiiFiles):
	print(niiF,labF)

	## Get the needed datword_a
	vtc = nf.NiftiImage(niiF)
	roi = nf.NiftiImage(roiVmr)
	vols, labels = pre.readLabList(labF)

	## Preprocess the data
	reducedRoi = pre.roiReduce(roi,vtc)
	maskedVtc  = pre.maskVtc(vtc,reducedRoi)
	reference  = pre.createRefVtc(maskedVtc)

	## Filter labels and vols by trainLab1, trainLab2
	## then change recode the labels as 1 and 2
	l1mask   = labels == trainLab1
	l2mask   = labels == trainLab2
	l1l2mask = l1mask != l2mask
	labels[l1mask] = 1
	labels[l2mask] = 2
	vols     = vols[l1l2mask]
	labels   = labels[l1l2mask]

	svmName = 'train_scene_'+ trainLab1 + 'x' + trainLab2 + '_' + svmBaseName

	pre.writeSVM(maskedVtc,reference,labels,vols,svmName)
	

print('Creating testing data:')
for labF, niiF in zip(testLabsFiles, testNiiFiles):
	print(niiF,labF)
	
	## Get the needed data
	vtc = nf.NiftiImage(niiF)
	roi = nf.NiftiImage(roiVmr)
	vols, labels = pre.readLabList(labF)

	## Preprocess the data
	reducedRoi = pre.roiReduce(roi,vtc)
	maskedVtc  = pre.maskVtc(vtc,reducedRoi)
	reference  = pre.createRefVtc(maskedVtc)

	## Filter labels and vols by trainLab1, trainLab2
	## then change recode the labels as 1 and 2
	l1mask   = labels == testLab1
	l2mask   = labels == testLab2
	l1l2mask = l1mask != l2mask
	labels[l1mask] = 1
	labels[l2mask] = 2
	vols     = vols[l1l2mask]
	labels   = labels[l1l2mask]

	svmName = 'train_word_'+ testLab1 + 'x' + testLab2 + '_' + svmBaseName

	pre.writeSVM(maskedVtc,reference,labels,vols,svmName)

os.chdir('/Users/type/Code/mvpa')

