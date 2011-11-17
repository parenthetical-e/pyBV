#! /opt/local/bin/python

"""
A set of functions and commend line script for selecting a fraction of the largest
t-values from ROI data (Brainvoyager v2.1).  The incoming text file should 
have been produced from the 'VOI Details' button in the options 
section of the VOI analysis.  The resulting t-values are taken from the
current contrast/GLM.

CL version:
	- arg1: name of the .voi file to be created
	- arg2: name of the roi data to be processed
"""

def readAndParseRoi(name,skip=22):
	"""
	Reads and parses the roi data text file, returning: voxelNum
	(a scalar indicating how many voelxs there are), tVals 
	(list of t-values) and locations (a list of tuples).

	**tVals and locations are in the same order**.
	"""

	## read and parse the header 
	## looking for voxelNum
	voxelNum = 0
	f = open(name,mode='r')
	for ii in range(0,skip):
		line = f.readline()
		splitUp = line.split()
		if 'NrOfVoxels:' in splitUp:
 			voxelNum = int(splitUp[1])

	## read and store the rest
	locations = []  # Will be list of tuples containg the x
					# ,y,z voxel cordinates
	tVals = []			
	# f.seek(skip+1) f is an iter so this is not needed, state saved?
	for line in f:
		columns = line.split()
		locations.append((int(columns[0]),int(columns[1]),int(columns[2])))
		tVals.append(columns[3])
	f.close()
	
	if len(tVals) != voxelNum:
		print 'Reported/found voxels were unequal - {0!s}/{1!s}'.format(voxelNum,len(tVals))

	return voxelNum, tVals, locations

def selectByT(tVals,locations,keep=.05):
	"""
	Sort and select t values along with their matching locations.
	
	- 'keep' is the fraction of voxels you wish to keep, default = .05
	- returns two lists, the top 'keep' percent of tVals and matching
	locations (in a tuple).
	"""
	from operator import itemgetter
	
	voxelNum = len(locations)

	## zip, sort based on tVals
	loc_t = zip(locations,tVals)
	loc_t_sorted = sorted(loc_t, reverse=True, key=itemgetter(1))
		# docs on sorting tuples:
		# http://wiki.python.org/moin/HowTo/Sorting/
	
	## select top keep percent
	numToKeep = int(voxelNum * keep)
	loc_t_select = loc_t_sorted[0:numToKeep]
	location_select, tVals_select = zip(*loc_t_select)
										# for some reason the 
										# asterix undoes the zip.
										# I'm confused by this.
	
	return tVals_select, location_select

def writeVOI(name, locations):
	"""
	Takes a locations list (as created by readAndParseRoi() or selectByT()) and 
	writes a BV (v2.1) compatible .voi file (in TAL coordinates).
	"""
	
	voxelNum = len(locations)
	header = []  # appending to list is faster then str.join()...?
	header.append('\n')
	header.append('FileVersion:                3\n')
	header.append('\n')
	header.append('CoordsType:                 TAL\n')
	header.append('\n')
	header.append('SubjectVOINamingConvention: <SUBJ>_<VOI>\n')
	header.append('\n')
	header.append('\n')
	header.append('NrOfVOIs:                   1\n\n')
	header.append('NameOfVOI:  ' + str(name) + '\n')
	header.append('ColorOfVOI: 255 102 102' + '\n\n')
	header.append('NrOfVoxels: ' + str(voxelNum) + '\n')
	
	f = open(name, 'w')
	[f.write(line) for line in header]
	[f.write('{0!s}\t{1!s}\t{2!s}\n'.format(loc[0],loc[1],loc[2])) for loc in locations]
	f.write('\n\nNrOfVOIVTCs: 0')
	f.close()

## For CL invocation 
## first arg should be the final .voi name, 
## second should be the name of the incoming roi data
if __name__ == '__main__':
	import sys
	argvs = sys.argv[1:]
	outName = str(argvs[0])
	roiName = str(argvs[1])
	
	print 'Reading {0}.'.format(roiName)
	voxN, tvals, locs = readAndParseRoi(roiName, 22)
	# print 't-values after intial read:'
	# print tvals

	tvals_sel, locs_sel = selectByT(tvals, locs, .05)
	# print 't-values after selecton:'
	# print tvals_sel
	print 'Selection complete, writing {0}.'.format(outName)
	writeVOI(outName,locs_sel)

