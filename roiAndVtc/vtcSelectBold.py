"""
A set of functions and CL script that reads in vtc formatted BOLD data from 
BrainVoyager ('vtc data export' tab in the VOI ananlysis window) then selects
a fraction (specified by 'keep') of voxels based on largest deviation 
or greatest information.

CL invocation needs:
	arg0: the selection method {'var' or 'info', no quotes}
	arg1: the name of .voi that will be created
	arg2: the name of the vtc data to be processed
"""

def readAndParseVtc(name):
	"""
	Read and parse the vtc data exported from BrainVoyager. 
	
	Returns:
	'bold' a 2d numpy array of the BOLD data and 'locations' 
	(a list of location tuples); each column is a voxel. 
	The order of bold and locations is matched
	"""
	import numpy as np  

	## Get the voxel locations
	f = open(name, mode='r')
	f.seek(1) # drop the first line
	x = [f.readline()][1:]  # drop first element
	y = [f.readline()][1:]  # of each 
	z = [f.readline()][1:]
	locations = zip(x,y,z)
	f.close()

	bold = np.loadtxt(name, skiprows=4, dtype='int')
	bold = bold[...,1:] # drop the first col
	
	return bold, locations


def selectByVar(bold,locations,keep=.05):
	"""
	Select a fraction of voxels with the largest variance; 
	'keep' is the fraction of voxels kept.

	Returns:
	var_select (the st dev per voxel) and locations_select, 
	the selected location data (still as a list of tuples).
	"""

	voxelNum = len(var)
	numToKeep = int(voxelNum * keep)

	var = bold.std(axis=0)
	loc_var = zip(locations, var)
	loc_var_sort = sorted(loc_var, reverse=True, key=itemgetter(1))
		# docs on sorting tuples:
		# http://wiki.python.org/moin/HowTo/Sorting/

	loc_var_select = loc_var_sort[0:numToKeep]
	locations_select, var_select = zip(*loc_var_select)

	return var_select, locations_select



def selectByInfo(bold,locations,keep=.05):
	"""
	Select a fraction of voxels with the largest information content; 
	'keep' is the fraction of voxels kept.

	Returns:
	info_select (the average info per voxel) and locations_select,
	the selected location data (still as a list of tuples).
	"""

	voxelNum = len(locations)
	numToKeep = int(voxelNum * keep)
	
	# DO INFO CALC, produce 'info'
	# info = 
	loc_info = zip(locations, info)
	loc_info_sort = sorted(loc_info, reverse=True, key=itemgetter(1))
		# docs on sorting tuples:
		# http://wiki.python.org/moin/HowTo/Sorting/
	loc_info_select = loc_info_sort[0:numToKeep]
	locations_select, info_select = zip(*loc_info_select)

	return info_select, locations_select



## For CL: 
if __name__ == '__main__':
	import sys
	import numpy as np
	import roiExtract as roi

	argvs = sys.argv[1:]
	method = str(argvs[0])
	vtcName = str(argvs[2])
	outName = str(argvs[1])

	print 'Reading {0}.'.format(vtcName)
	bold, locations = readAndParseVtc(vtcName)
	if method == 'var':
		var_sel, locations_select = selectByVar(bold, locations, .05)
	elif method == 'info':
		info_sel, locations_select = selectByInfo(bold, locations, .05)
	else:
		print '{0} was not a known method. Try "var" or "info".'.format(method)

	print 'Selection complete, writing {0}.'.format(outName)	
	roi.writeVOI(outName,locations_select)
