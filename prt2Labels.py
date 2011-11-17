#! /opt/local/bin/python
# Invokes the macports python version.

def prt2Labels(prtFileName='',windowSize=int(),offset=int(),condNames=[]):
	"""
	Reads in a Brainvogager .prt file, returning a tuple of two lists
	(location and label) based on the conditions found in the prt.  
	Length of prt entries is ignored, instead
	window size is used.  Location is the volume number in TR units.

	Changes:
	[6/22/2011]  The 'offset' parameter was introduced to postivily shift
	the numeric entries by its given amount to account for the delay in BOLD response.  
	Vales between 4-6 seconds are typically used in the literature (default is 5).
	"""
	import re
	import numpy as np

	prtFile = open(prtFileName, 'r')
	cVec = []  # Condition vector
	lVec = []  # Location vector
	windowSize = int(windowSize)
	offset 	   = int(offset)

	while True:
		try:
			line = next(prtFile).strip()
			if condNames.count(line) >= 1:
				condName = line

				# loop over lines following
				# a condition match
				while True:
					line = next(prtFile).strip()
					
					# 'COLOR:' indicates end of condition 
					if re.match('COLOR:',line,re.I):
						break

					elif re.search('0+\s+0+',line):
						print('Warning: incorrect (NULL) entry in prt: ' + 
							  prtFileName)
						continue

					# match: <int><whitespace><int>
					# add location and condition data
					# to their vecs, multiple of windowSize.
					elif re.search('^\d+\s+\d+',line):
						line = line.split()
						cVec.extend([condName]*windowSize)
						lVec.extend([ int(line[0]) + ii for 
									ii in range(0,windowSize) ])
		
		except StopIteration:
			break
	
	lVec = np.array(lVec)
	lVec = lVec + offset
	return(lVec, cVec)

# for command line invocation:
if __name__ == '__main__':
	import sys
	argvs = sys.argv[1:]
	lcOut = prt2Labels(argvs[0],argvs[1],argvs[2],argvs[3:])
	lcOut = zip(lcOut[0],lcOut[1])
	
	# There has to be a better way to write a tuple than this
	# but google was of little help.  I miss perl...
	fout = open(argvs[0] + '_labelList'  + '.txt','w')
	for entry in lcOut:
		fout.writelines(str(entry[0]) +
						',' + 
						str(entry[1]) + 
						'\n')
	fout.flush()
	fout.close()
