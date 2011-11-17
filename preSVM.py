"""
A set of functions for preprocessing Brainvoyager data (in the .nii format)
in preperation for SVM analysis using PyML.
"""

import numpy as np
import nifti as nf
import Image as im
import os as os
import re as re
import PyML as ml


def readLabList(fname):
	"""
	Read in the labelList files created by prt2Label() from Brainvoyager
	.prt files.
		- Returns two arrays in a tuple, one for volumes (as int) 
		  one for the labels (str).
	"""

	vols,labels = np.loadtxt(fname,delimiter=',',unpack=True,dtype=str)
	vols = np.array(vols,dtype='int')
	return vols, labels

	
def resize_worker(array_2d,dims=()):
	"""
	Does the heavy lifting for resizeVtc() and resizeVmr(); 
	It is not desinged for external use.
	
	- Returns a 2d array, resized to 'dims' by nearest neighbor 
	interpolation.
	"""
	
	dims_swapped = (dims[1],dims[0])
		# for PIL compatibility

	## convert to image, resize, and back to array
	img = im.fromarray(np.uint32(array_2d),"I")
	img_resize = img.resize(dims_swapped, im.NEAREST)
	array_2d_resize = np.array(img_resize)
	
	## Debug:
	print("array_2d size: {0}".format(array_2d.shape))
	print("Before resize as img: {0}".format(img.size))
	print("Target dims: {0}".format(dims))
	print("Target swapped dims: {0}".format(dims_swapped))
	print("After resize as img: {0}".format(img_resize.size))
	print("array_2d_resize size: {0}".format(array_2d_resize.shape))

	return array_2d_resize


def resize3d(array3d,intial_dim,final_dim):
	"""
	Loops throuh z then x resizing a given 3d array 
	via resize_worker(), which is required.  
	
	Internal use only.
	"""

	print("Starting z loop...")
	first_resize = np.zeros((final_dim[0:2]+intial_dim[2:]),dtype="uint32")
	first_dim = final_dim[0:2]
	for z in range(intial_dim[2]):
		first_resize[:,:,z] = resize_worker(array3d[:,:,z],first_dim) 
	
	print("Starting y loop...")
	second_resize = np.zeros(final_dim,dtype="uint32")
	second_dim = final_dim[1:]
	for x in range(final_dim[0]):
		second_resize[x,:,:] = resize_worker(first_resize[x,:,:],second_dim)

	print("[0] => {1}".format(intial_dim,final_dim))
	return second_resize


def downsampleVmr(vmr, by):
	"""
	Uses PIL (python imaging library) to resize the x, y and z dimensions
	of the given vmr to match that of vtc.

	-Returns: a the resized vmr in a NiftiImage object.
	"""
	intial_dim = vmr.data.shape

	# TODO test for int?  How to deal with fractions here...
	final_dim = (by[0]/intial_dim[0],by[1]/intial_dim[1],by[2]/intial_dim[0])
	resized_vmr_data = resize3d(vmr.data,intial_dim,final_dim)	

	return nf.NiftiImage(resized_vmr_data,vmr.header)


def upsampleVtc(vtc, by, vol):
	"""
	[9/1/2011]: This function replaces roiReduce(), which has been depricated; 
	input/output data formats are changed between the two; 
	the volume in the vtc data to act on was added.

	resizeVtc() takes a vmr and vtc NiftiImage() objects (converted to .nii 
	files of course), and alters the x, y, z, dimentions of the vtc to 
	match the vmr by nearest neighbor interpolation.
	
	- Requires resize_worker()

	NOTE: Upscaled vtc data can become VERY large.  For example,
	270 volumes at 256x256x256 occupies over 18 GBs.

	Returns: the resolution altered volume for in vtc stored as a 
	NiftiImage object, the header is dropped.
	"""

	print("Vol: {0}".format(vol))
	intial_dim = vtc.data.shape[1:]
	final_dim = (intial_dim[0]*by[0], intial_dim[1]*by[1],intial_dim[2]*by[2])
	
	resized_vtc_data = resize3d(vtc.data[vol,:,:,:],intial_dim,final_dim)

	return nf.NiftiImage(resized_vtc_data,vtc.header)


def createRef(nii_data):
	"""
	Creates an reference space of sequential intergers simplifying 
	voxel/feature labeling in SVMLIB formatted data.
		- Returns an numpy array populated with an index starting 
		  at 0 at (0,0,0) and ending at N at (N,N,N).
	"""
	
	## Init a vtc.data(x,y,z) shaped array, then fill it with unique
	## random numbers.
	shapeXYZ = nii_data.shape[1:]
	numCoord = np.size(nii_data[1,...])
	refXYZ = np.arange(0,numCoord).reshape(shapeXYZ)
	
	return refXYZ


def maskVtc(roi_vmr,vtc,vol):
	"""
	Creates a bool mask everywhere the roi data is greater than 2.
	Roi needs to a nifti object resulting from imported vmr data. Then
	uses the roiMask to extract that data at the appropriate voxels
	from vtc (a 4D nifti object). It should probably be applied to
	the vtcdata, the labelVtc, and refVtc.
		- roiVmr and vtc(x,y,z) must have indentical dimensions.  
		- 'vol' is the volumne number of the vtc data you wish to mask

		- NOTE: To convert a voi to vmr use the ROI window in Brainvoyager.
		  Select Options, then the 'VOI functions' tab, and select
		  'Create MSK...', give the file a name (it does not matter
		  what as this file is not needed).  Once that is done go to
		  the File menu and select 'Save secondary VMR'.  The result-
		  ing file is wh)t should be read in with NiftiImage().
		
		- Returns a (t=1,x,y,z) nifit object if roi masked vtc data
		  w/ correct header info.
	"""
	
	roi_data = roiVmr.data
	# roiData = np.round(roiData)
		# VMR data contain very small non-zero deicmal entries
		# (e.g. 235.000011 or 0.0000124) where there should be empty
		# decimals (235.0000000).

	masked_vtc = np.zeros(roi_vmr, dtype="uint32")

	## create a ref (later upscaled) to find redundant voxels
	ref = createRef(vtc.data[1,...])
	ref_vtc = nf.NiftiImage(ref[np.newaxis,...]) # vtc NiftiImage obj 
												 # needed for resizeVtc()
	ref_resize = resizeVtc(roi_vmr,ref_vtc,1)

	## Create roi_mask then falsify redundant 
	## entries; keep on the the first
	roi_mask = roi_data > 2
	for uni in np.unique(ref_resize.data):
		ind = np.where(uni == ref_resize.data)
		ind = (ind[0][1:],ind[1][1:],ind[2][1:])
		roi_mask[ind] = False

	## rescale the vtc vol to match the vmr
	## mask that vols data and store in 
	## masked_vtc
	vol_resize = resizeVtc(roi_vmr,vtc,vol)
	masked_vtc = np.where(roi_mask,vol_resize,0)
		# create t, set to 1

	return nf.NiftiImage(masked_vtc, vtc.header)
	

def writePyML(vtc,labels,vol,fname):
	"""
	All incoming vtcs/vmrs should have the same spatial dimensions and
	have been treated by identically (maskVtc)(), roiReduced(),...).
	This script will flatten each of the vtcs and write them to fname
	in the SVMLIB format. If fname exists data will be silently
	APPENDED to it.
		- Any voxels in the vtc data that are zero are not written.

	[6/27/2011]: Added a filter so that as vols is iterated it does not exceed
	vtc.data.shape[0] (i.e. the number of volumes in the vtc data).  This is a 
	concern as vols was offset (during 'prt2Labels.py') to allow for the slow BOLD
	response.  For details see prt2Labels.py.
	"""

	fname    = str(fname)		# just in case...
	outFile  = open(fname,'a')
	if os.path.exists(fname):
		print('*Appending* SVMLIB formatted data to {0}'.format(fname))
	else:
		print('Writing SVMLIB formatted data to {0}'.format(fname))
	
	## remove entries in 'vols' that would lead to 
	## 'vtc' dimensions being exceed as the result 
	## of the offseting for the BOLD delay
	numVolsInVtc = vtc.data.shape[0]
	volMask 	 = vols <= numVolsInVtc
	vols 		 = vols[volMask]

	for vol in range(0,np.size(vols)):
		flatVtcVol = vtc.data[vol,...].flatten()
		flatRefVol = refVtc.data.flatten()
		
		## Remove empty voxels.
		zeroMask   = flatVtcVol != 0
		flatVtcVol = flatVtcVol[zeroMask]
		flatRefVol = flatRefVol[zeroMask]
		
		#print(np.sum(zeroMask))
		#print(np.size(flatVtcVol))
		#print(np.size(flatRefVol))

		## Build up the line for each vol/label.
		## Format: 'label voxID:data voxID:data ...'
		line = str(labels[vol]) + ' '
		for pos in range(0,np.size(flatVtcVol)):
			line = line + str(flatRefVol[pos])+ ':' + \
				str(flatVtcVol[pos]) + ' '

		## write the line and flush it.
		line = line + '\n'
		outFile.write(line)
		outFile.flush()
		
	outFile.close()


def nii_LabelListMatch(direc='.'):
	"""
	Finds all the .nii (i.e. nifti formatted vtc data) and labelList
	files in the current or a specified directory, then sort them; 
	both file types should have the same prefix and thus will correctly 
	aligned.
		- Returns a tuple of lists, nii in one, label files in the other
	"""
	
	files    = os.listdir(direc)
	niiFiles = []
	labFiles = []
	for entry in files:
		if  re.search('THPGLMF2c_TAL.nii$',entry):
			niiFiles.append(entry)

		elif re.search('_labelList.txt$',entry):
			labFiles.append(entry)

	if len(niiFiles) != len(labFiles):
		print('Different number of nii and label files!')
		print(niiFiles)
		print(labFiles)
	
	niiFiles.sort()
	labFiles.sort()

	return niiFiles, labFiles


def zSparse(fname):
	"""
	Converts a sparse formated SVMLIB data file to Vector/CSV format
	and then znomralizes on a feature basis and writes out that file
	as fname.
	"""

	znorm = ml.Standardizer()

	sparse = ml.SparseDataSet(fname)
	sparse.save('temp',format='csv')
	
	vec = ml.VectorDataSet('temp',labelsColumn=1,idColumn=0)
	znorm.train(vec)

	vecName = 'vec_' + fname
	
	# Verbal overwrite of priors
	if os.path.exists(vecName):
		print('Overwriting {0}.'.format(vecName))
		os.remove(vecName)

	vec.save(vecName)


def vecSplit(vecName='',fracTrain=0.3):
	"""
	Splits a vector/csv fotmatted SVMLIB file into training and test sets 
	by row according to numTrain and numTest and writes out the resulting 
	files. Existing files are overwritten.
	q	- numTest + numTrain should equal the number of lines in vecName 
		minus the one line header.
		- Returns 'Done.'

	[08/16/2011]: a major change - instead of taking numTest and numTrain 
	directly fracTrain (the fraction of trials destined for taining) 
	was added as a invocation arg. numTrain/numTest are now discovered.
	
	An unmodified (commented out) version of the old function was left in 
	the source.
	"""
	
	## Calc numbers of features for 
	## training and testing data
	vecData = ml.VectorDataSet(vecName,labelsColumn=1,idColumn=0)
	numTrain = int(vecData.numFeatures * fracTrain)
	numTest = vecData.numFeatures - numTrain

	## Create filenames of train and test data taht will
	## be written soon...
  	## Remove 'vec' from vecName so a more informative,
	## less redundant, names can be created.
	tmpName = str.split(vecName,'vec')
	trainName = open('vec_train_{0}{1}'.format(numTrain,tmpName[-1]), 'w')
	testName = open('vec_test_{0}{1}'.format(numTest,tmpName[-1]), 'w')  

	## Randomly select features for either  
	## training or testing.
	sampler = np.asarray([1] * numTrain + [2] * numTest)
	np.random.shuffle(sampler)  

	## Create indices from 'sampler'
	featureIndex = np.arange(len(sampler))
	invertTrainIndex = featureIndex[sampler == 2]
	invertTestIndex = featureIndex[sampler == 1]

	print('trainIndex: {0}'.format(invertTrainIndex))
	print('testIndex: {0}'.format(invertTestIndex))
    
	## Use trainIndex or testIndex to eliminate features,
	## deepcopy the vecData first; eliminateFeatures()
	## operates in place.
	trainData = ml.VectorDataSet(vecData)
	trainData.eliminateFeatures(invertTrainIndex.tolist())
	trainData.save(trainName)

	testData = ml.VectorDataSet(vecData)
	testData.eliminateFeatures(invertTestIndex.tolist())
	testData.save(testName)


#def vSplit(vecName='',numTrain=30,numTest=70):
	#"""
	#Splits a vector/csv fotmatted SVMLIB file into training and test sets 
	#by row according to numTrain and numTest and writes out the resulting 
	#files. Existing files are overwritten.
		#- numTest + numTrain should equal the number of lines in vecName minus 
		  #the one line header
		#- Returns 'Done.'

	#[08/16/2011]: major change, instead of taking numTest ad numTrain directly
	#fracTrain was added as a invocation arg and numTrain/numTest are now discovered.
	#The unmodified veriosn of the function is commented out below.
	#"""
	#import numpy as np


	### 1 = train, 2 = test
	### shuffle is in place
	#sampler = [1] * numTrain + [2] * numTest
	#sampler = np.array(sampler)
	#np.random.shuffle(sampler)  
	#sampler = list(sampler)
	#print('Sampler: ', sampler)

	#vecFile = open(vecName)
	#header = next(vecFile)

	#trainFile = open('train_{0}_{1}.txt'.format(numTrain,vecName), 'w')
	#testFile = open('test_{0}_{1}.txt'.format(numTest,vecName), 'w')
	#trainFile.write(header)
	#testFile.write(header)

	#for samp in sampler:
		#line = next(vecFile)
		#if samp == 1:
			#trainFile.write(line)
			#trainFile.flush()
		#else:
			#testFile.write(line)
			#testFile.flush()

	#vecFile.close()
	#trainFile.close()
	#testFile.close()

	#return 'Done.'

#def roiReduce(roiVmr, vtc):
	#"""
	#Pads a vmrROI with zeros, 1 in each dimension, so this
	#data can be easily reduced from the vmr's 1mm^3 resolution to the
	#vtc's 3mm^3 resolution. It then crops the excess so the roiVmr x,y,z
	#dimensions are equal to the provided vtc.
		#- Both roiVmr and vtc should be nifti objects.
		#- Returns a roiVmr nifiti object w/ a correct header.

	#"""
	#import numpy as np
	#import nifti as nf
	## start over based on:
	## (denoms are vtc dims x,y,z)
	##In [191]: 256/46; 256/40; 256/58
	##Out[191]: 5
	##Out[191]: 6
	##Out[191]: 4
	## reshape based on these?

	### Pad the roi w zeros
	#padRoi = np.zeros((258,258,258))
	#padRoi[1:257,1:257,1:257] = roiVmr.data
	
	### Reduce roi size by factor of 3:
	### Geedly includes any elements in the reduced ROI
	### just as Brainvoyager does when doing a roi GLM.  
	### That is if only one element in the 1mm resolution
	### is present in the 3mm resolution it is included.
	#roi3mm = padRoi.reshape((258/3,3, 258/3,3, 258/3,3))
	#roi3mm = roi3mm.mean(5).mean(3).mean(1)
	
	### Crop the roi to vtc dimensions
	#roi3mmShapeXYZ = np.array(roi3mm.shape)
	#vtcShapeXYZ    = np.array(vtc.data.shape[1:])
	#excess  = (roi3mmShapeXYZ - vtcShapeXYZ) / 2
	#roiCrop = roi3mm[(excess[0]):(86-excess[0]),(excess[1]):(86-excess[1]),(excess[2]):(86-excess[2])]
																			
	#roiReduced = nf.NiftiImage(roiCrop,roiVmr.header)
	#return roiReduced

