def runSVM(trainF,testF):
	"""
	[6/27/2011]: This script serves little purpose as it takes to long
	to interate over 0-10,000 to optimize C automatically.
	It is more effiecinetly done by hand (for now at least).

	Go!
	"""
	import PyML as ml
	import numpy as np
	znorm = ml.Standardizer()

	# Init the SVM	
	s=SVM()
	print(s)

	# Reformat the data to csv (aka Vector) so feature based
	# normalization can occur, then normalize that train and 
	# test data
	test     = ml.SparseDataSet(testF)
	vectestF = 'vec_' + testF
	test.save(vectestF,format='csv')
	vecTest  = ml.VectorDataSet(vectestF,labelsColumn=1,idColumn=0)
	znorm.train(vecTest)

	trainedSVM = []  
		# Returns a SVM trained class
	if isinstance(trainF,str):
		train 	  = ml.SparseDataSet(trainF)
		vectrainF = 'vec_' + trainF
		train.save(vectrainF,format='csv')
		vecTrain = ml.VectorDataSet(vectrainF,labelsColumn=1,idColumn=0)
		znorm.train(vecTrain)

		# Optimize C
		param = ml.modelSelection.Param(
						s, 'C', list(np.arange(0,10000,.5)))
		m = ml.modelSelection.ModelSelector(param)
		trainedSVM = m.train(vecTrain)  # Optimize C
		trainedSVM.save('svm_' + testF)
	else:
		trainedSVM = trainF
	

	cross   = trainedSVM.stratifiedCV(trainedSVM, 10)
	results = trainedSVM(vecTest)
	
	return results
