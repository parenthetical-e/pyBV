
import preSVM as pre

## Globals needed to create the SVMLIB data for a run
## =================================================
svmBaseName = '_giantHippo_svmData.txt'

roiVmr = 'giantHippoTest.nii'

trainLab1 = 'STI'
trainLab2 = 'STU'
testLab1 = ''
testLab2 = ''

os.chdir('/Users/type/Lab/RWCR/bv_AR/mvpa_data/')
niiFiles, labFiles = pre.nii_LabelListMatch('.')

trainLabsFiles = labFiles
trainNiiFiles  = niiFiles
testLabsFiles  = labFiles
testNiiFiles   = niiFiles
## =================================================
## 

from time import localtime, strftime
svmBaseName = svmBaseName + strftime("%a%d%b%Y%H:%M:%S", localtime())
	# Add a timestamp to the SVM file name, preventing accidental
	# overwriting or modification.

for labF, niiF in trainLabsFiles, trainNiiFiles:
	## Get the needed data
	vtc = nf.NiftiImage(niiF)
	roi = nf.NiftiImage(roiVmr)
	vols, labels = pre.readLabList(labF)

	## Preprocess the data
	reducedRoi = pre.reducedRoi(roi,vtc)
	maskedVtc  = pre.maskVtc(vtc,reducedRoi)
	reference  = pre.createRefVtc(maskedVtc)

	## Filter labels and vols by trainLab1, trainLab2
	## then change recode the labels as 1 and 2
	l1mask   = labels == trainLab1
	l2mask   = labels == trainLab2
	l1l2mask = l1mask != l2mask
	vols     = vols[l1l2mask]
	labels   = labels[l1l2mask]
	labels[l1mask] = 1
	labels[l2mask] = 2

	svmName = 'train_'+ trainLab1 + 'x' + trainLab2 + '_' +svmBaseName

	pre.writeSVM(maskedVtc,reference,labels,vols,svmName)
	

for labF, niiF in testLabsFiles, testNiiFiles:
	## Get the needed data
	vtc = nf.NiftiImage(niiF)
	roi = nf.NiftiImage(roiVmr)
	vols, labels = pre.readLabList(labF)

	## Preprocess the data
	reducedRoi = pre.reducedRoi(roi,vtc)
	maskedVtc  = pre.maskVtc(vtc,reducedRoi)
	reference  = pre.createRefVtc(maskedVtc)

	## Filter labels and vols by trainLab1, trainLab2
	## then change recode the labels as 1 and 2
	l1mask   = labels == testLab1
	l2mask   = labels == testLab2
	l1l2mask = l1mask != l2mask
	vols     = vols[l1l2mask]
	labels   = labels[l1l2mask]
	labels[l1mask] = 1
	labels[l2mask] = 2

	svmName = 'test_'+ trainLab1 + 'x' + trainLab2 + '_' +svmBaseName

	pre.writeSVM(maskedVtc,reference,labels,vols,svmName)

os.chdir('/Users/type/Code/mvpa')
