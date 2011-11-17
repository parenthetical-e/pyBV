"""
Create PyML datasets from Brainvoyager data.

The location of the Brainvoyager data is assumed to be 
in the current directory and can't be otherwise specified.

Requires preSVM.py, assumed to be in 'Users/type/Code/mvpa'
(which you may need to change, see around line 18).

These have been tested and run inside an ipython session 
using the macports install of python26.
"""

import nifti as nf
import PyML as ml
import os as os

## source preSVM.py
oldpth = os.getcwd()
os.chdir('/Users/type/Code/mvpa/')
import preSVM as pre
#reload(pre)
	# excessive reloading for easy debugging
os.chdir(oldpth)


def createDataset(svmName='',roiName='',labFiles=[],niiFiles=[],lab1='',lab2=''):
	"""
	Workhorse...
	"""
	if os.path.exists(svmName):
		print('Overwriting {0}.'.format(svmName))
		os.remove(svmName)

	for labF, niiF in zip(labFiles, niiFiles):
		print(niiF,labF)
	 
		## Get the needed data
		vtc = nf.NiftiImage(niiF)
		roi = nf.NiftiImage(roiName)
		vols, labels = pre.readLabList(labF)

		## Preprocess the data
		reducedRoi = pre.roiReduce(roi,vtc)
		maskedVtc  = pre.maskVtc(vtc,reducedRoi)
		reference  = pre.createRefVtc(maskedVtc)

		## Filter labels and vols by trainLab1, trainLab2
		## then change recode the labels as 1 and 2
		l1mask   = labels == lab1
		l2mask   = labels == lab2
		l1l2mask = l1mask != l2mask
		labels[l1mask] = 1
		labels[l2mask] = 2
		vols     = vols[l1l2mask]
		labels   = labels[l1l2mask]

		pre.writeSVM(maskedVtc,reference,labels,vols,svmName)
	else:
		print('Z-normalizing the data.')
		pre.zSparse(svmName)
			## z-norm the data, requires sparse conversion


def diffLabels(roi='',train=('STI','STU'),test=('TESTSTI','TESTSTU')):
	"""
	This function creates an PyML dataset from vtc amd roi (Brainvoyager) 
	formatted data.  It should be used when the training and testing labels
	are different but all files (relevant) in the PWD are of interest.
	
	Returns:
		- a pair of z-normalized BOLD datasets, one for svm training
		and one for testing; BOLD data was extracted from the full
		(vtc) timecourse via the given roi (as .vmr)
		- data from identical runs is silently overwritten, but
		user is informed of this
	"""

	niiFiles, labFiles = pre.nii_LabelListMatch('.')
		## Create lists of all the
		## needed files

	svmBaseName = (str.split(roi,'.'))[0] + '.txt'
		## suffix of the file to be written

	## TRAIN SET: ###################################################
	svmName = 'train_'+ train[0] + 'x' + train[1] + '_' + svmBaseName	
	print('Creating training data: {0}'.format(svmName))
	if os.path.exists(svmName):
		print('Overwriting {0}.'.format(svmName))
		os.remove(svmName)

	createDataset(svmName,roi,labFiles,niiFiles,train[0],train[1])
		## createDataset(svmName='',roiName='',labFiles,niiFiles,lab1='',lab2='')

	## TEST SET: ###################################################
	svmName = 'test_'+ test[0] + 'x' + test[1] + '_' + svmBaseName	
	print('Creating testing data: {0}'.format(svmName))
	if os.path.exists(svmName):
		print('Overwriting {0}.'.format(svmName))
		os.remove(svmName)

	createDataset(svmName,roi,labFiles,niiFiles,test[0],test[1])
		## createDataset(svmName='',roiName='',labFiles,niiFiles,lab1='',lab2='')


def testOnly(roi='',test=('TESTSTI','TESTSTU')):
	"""
	This function creates an PyML dataset from vtc amd roi (Brainvoyager) 
	formatted data.  It should be used when the ONLY TEST
	labels are needed but different labels will be used for each 
	(as in diffLabels()).
	
	Returns:
		- a pair of z-normalized BOLD datasets, one for svm training
		and one for testing; BOLD data was extracted from the full
		(vtc) timecourse via the given roi (as .vmr)
		- data from identical runs is silently overwritten, but
		user is informed of this
	"""
	niiFiles, labFiles = pre.nii_LabelListMatch('.')
		## Create lists of all the needed files

	svmBaseName = (str.split(roi,'.'))[0] + '.txt'
		## suffix of the file to be written

	## TEST SET: ###################################################
	svmName = 'test_'+ test[0] + 'x' + test[1] + '_' + svmBaseName	
	print('Creating testing data: {0}'.format(svmName))
	if os.path.exists(svmName):
		print('Overwriting {0}.'.format(svmName))
		os.remove(svmName)

	createDataset(svmName,roi,labFiles,niiFiles,test[0],test[1])
		## createDataset(name='',roi='nof of roi.vmr',labFiles,niiFiles,lab1='',lab2='')


def trainOnly(roi='',train=('TESTSTU','TESTSTI')):
	"""
	This function creates an PyML dataset from vtc amd roi (Brainvoyager) 
	formatted data.  It should be used when the ONLY TRAINING
	labels are needed but different labels will be used for each 
	(as in diffLabels()).
	
	Returns:
		- a pair of z-normalized BOLD datasets, one for svm training
		and one for testing; BOLD data was extracted from the full
		(vtc) timecourse via the given roi (as .vmr)
		- data from identical runs is silently overwritten, but
		user is informed of this
	"""
	niiFiles, labFiles = pre.nii_LabelListMatch('.')
		## Create lists of all the needed files

	svmBaseName = (str.split(roi,'.'))[0] + '.txt'
		## suffix of the file to be written

	## TRAIN SET: ###################################################
	svmName = 'train_'+ train[0] + 'x' + train[1] + '_' + svmBaseName	
	print('Creating training data: {0}'.format(svmName))
	if os.path.exists(svmName):
		print('Overwriting {0}.'.format(svmName))
		os.remove(svmName)

	createDataset(svmName,roi,labFiles,niiFiles,train[0],train[1])
		## createDataset(svmName='',roiName='',labFiles,niiFiles,lab1='',lab2='')


def sameLabels(roi='',fracTrain=0.3,labels=('STI','STU')):
	"""
	Creates a unmodified and z-normalized PyML datasets from BV data
	dividing part (as specified by percentTrain) into 
	a training set and part into a testing set.  
	
	This is to be used when the same labels are applied to training 
	and testing sets AND the training and testing are pulled from 
	the same Brainvoyager data files.
	"""
	niiFiles, labFiles = pre.nii_LabelListMatch('.')

	svmBaseName = (str.split(roi,'.'))[0] + '.txt'
	svmName = labels[0] + 'x' + labels[1] + '_' + svmBaseName
	
	createDataset(svmName,roi,labFiles,niiFiles,labels[0],labels[1])
	pre.vSplit(vecName='vec_'+svmName,fracTrain=0.3)


def diffFilesSameLabels(roi='',trainFiles=('name',[],[]),testFiles=('name',[],[]),labels=('','')):
	"""
	Creates a unmodified and z-normalized PyML datasets from BV data
	specified in trainFiles and testFiles using the same labels for 
	each.  This is to be used when you want to split up the BV into known 
	groups but the labels for each group are the same.

	IMPORTANT: trainFiles and testFiles are tuples of a name and two 
	file lists.  The later needed to enusre that file output does 
	not collide with unrelated files and cause sensible names are nice.
	The list containing labels should be the first entry in each, 
	a list of nii files should be the second.
	"""
	svmBaseName = (str.split(roi,'.'))[0] + '.txt'
	svmName_trn = 'train_{0}_{1}x{2}_{3}'.format(trainFiles[0],labels[0],labels[1],svmBaseName)
	svmName_tst = 'test_{0}_{1}x{2}_{3}'.format(testFiles[0],labels[0],labels[1],svmBaseName)
	
	createDataset(svmName_trn,roi,trainFiles[1],trainFiles[2],labels[0],labels[1])
	createDataset(svmName_tst,roi,testFiles[1],testFiles[2],labels[0],labels[1])


def diffFilesDiffLabels(roi='',trainFiles=('name',[],[]),testFiles=('name',[],[]),train=('',''),test=('','')):
	"""
	Creates a unmodified and z-normalized PyML datasets from BV data
	specified in trainFiles and testFiles using the differnt labels for 
	each.  This is to be used when both unqiue files and labels are to be
	employed in the training and testing sets.

	IMPORTANT: trainFiles and testFiles are tuples of file lists.  Files 
	containing labels should be the first entry in each, a list of nii 
	files should be the second.
	"""

	svmBaseName = (str.split(roi,'.'))[0] + '.txt'
	svmName_trn ='train_{0}_{1}x{2}_{3}'.format(trainFiles[0],train[0],train[1],svmBaseName)
	svmName_tst ='test_{0}_{1}x{2}_{3}'.format(testFiles[0],test[0],test[1],svmBaseName)

	createDataset(svmName_trn,roi,trainFiles[1],trainFiles[2],train[0],train[1])
	createDataset(svmName_tst,roi,testFiles[1],testFiles[2],test[0],test[1])
