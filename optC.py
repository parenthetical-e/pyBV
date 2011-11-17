def optC(vecFile=''):
	"""
	A function for optimizing C (the soft margin constatn) 
	for PyML datasets and SVMs.  Optimization proceeds by
	first stepping further inside the range of
	{0.1...1} in increments of 0.1 or outside it, by 
	orders of magnitude.
	
	Max possible C values are 0.01 and 1000; Best possible 
	precision is to the first decimal place.

	PyML is very verbose, so progress in this optmization 
	is recorded in 'optC.log' in the PWD. This log file is 
	appended to across invocations and so may grow without bound.
	"""
	import PyML as ml
	import numpy as np
	from time import strftime
	
	trainData = ml.VectorDataSet(vecFile,labelsColumn=1,idColumn=0)
	log = open('optC.log','a')

	bestC = 1
	stepSize = .1 
		## init w reasonable but in the stop criterion's range

	possibleC = np.array([.1,1])
		## middle to start; possibleC can span no more 
		## than 1 power of 10 otherwise this function will
		## blowup or take an eternity

		## possibleC must be cast as float; int breaks PyML
	log.write('\n\n\n**Begining new optimization.**\n')
	log.write('Dataset: {0}\n'.format(vecFile))
	log.write('First set of possible C values: {0}.\n'.format(possibleC))

	while True:
		log.flush()

		## try all 'possibleC'
		startTime = strftime("%a%d%b%Y_%H:%M:%S")
		s = ml.SVM()
		param = ml.modelSelection.Param(s, 'C', possibleC)
		m = ml.modelSelection.ModelSelector(param)
		m.train(trainData)
		stopTime = strftime("%a%d%b%Y_%H:%M:%S")
		
		log.write('Start/stop times last iteration: 
				{0}/{1}\n'.format(startTime,stopTime))
		
		bestC = m.classifier.C

		## The stop criterion is the 
		## level of precision desired.
		if stepSize < .1:
			log.write('SUCCESS. C is {0}\n'.format(bestC))
			break
		## C can not be greater than 1000
		## or less than .01
		elif bestC > 1000 or bestC < 0.01:
			log.write('WARNING: C is out of range. C is {0}\n'.format(bestC))
			break
		else:
			## Where was best C for last iteration?
			## Use that location to define next set of 
			## possible C values.
			indexC = possibleC.tolist().index(bestC)
			log.write('Best C for last interation: {0}.\n'.format(bestC))
			
			if possibleC[indexC] == possibleC.max():
				stepSize = round(possibleC.max())
				possibleC = np.arange(stepSize,(stepSize*10),stepSize)
				log.write('At max range, new values are: {0}.\n'.format(possibleC))
			elif indexC == 0:
				stepSize = possibleC.min()/10
				possibleC = np.arange(stepSize,possibleC.min(),stepSize)	
				log.write('At min range, new values are: {0}.\n'.format(possibleC))
			else:
				stepSize = stepSize/10
				possibleC = np.arange(possibleC[indexC-1],possibleC[indexC+1],stepSize)
				log.write('Was in range, next values are: {0}.\n'.format(possibleC))
	log.close()
	return bestC
