import PyML as ml
import numpy as np
from time import localtime, strftime
# giantHippo (STIvSTU) -- C : 3.540000

## vec_train_STIxSTU_Tue01Mar2011_12-20-10_hippoWordAndScene_svmData.txt
# [0.01,1,10,100] -- C : 100.000000
# [100.0,200,400] -- C : 100.000000
# [20,40,60,80.0] -- C : 60.000000
# [41,45,50,55,60,65,70,75,79.0] -- C : 75.000000
# [72, 73, 74, 75, 76, 77.0] -- C : 74.000000
# list(np.arange(73.5,74.6,.01)) -- C : 74.070000

#'1_30_vec_train_scene_TESTSTUxTESTNSU_Tue01Mar2011_19-08-59_ppa_svmData.txt' 
# [0.01,1,10,100] -- C : 100.000000 
# [100,200,300,400] -- C : 300
# [250,260,270,280,290,300,310,320,330,340,350] -- C : 260
# [255,256,257,258,259,260,261,262,263,264,265,266] -- C : 264
# list(np.arange(263.5,264.6,.01)) -- C : 263.88
# reran it with above and found C : 263.810000

# 1_30_vec_test_TESTSTIxTESTSTU_Tue01Mar2011_12-20-10_hippoWordAndScene_svmData.txt
# [0.01,1,10,100] -- C : 100
# [50, 100, 150, 200, 250, 300, 350, 400] -- C : 300
# [260,270,280,290,300,310,320,330] -- C : 280.000000
# [275,276,277,278,279,280,281,282,283,284,285] -- C : 284
# list(np.arange(283.5,284.6,.01)) -- C : 284.44000

# 1_30_vec_test_TESTSTUxTESTNSU_Tue01Mar2011_12-31-39_hippoWordAndScene_svmData.txt
# [.01,1,10,100,1000] -- C : 10.000000
# [5,10,20,30,40,50,60,70] -- C : 20.00000
# [15,16,17,18,19,20,21,22,23,24,25,26] -- C : 19:00000
# list(np.arange(18.5,19.6,.01)) -- C : 18.880000

# '1_50_vec_test_TESTSTIxTESTSTU_Tue01Mar2011_12-20-10_hippoWordAndScene_svmData.txt'
# [01,1,10,100,1000] -- C : 10.000
# [6,7,8,8,10,20,30,40,50,60]  -- C : 8.00000
# [7.5,8.5,9.5] -- C : 7.50000
# list(np.arange(7.4,8,.01)) -- C : 7.97

# 1_50_vec_train_scene_TESTSTIxTESTSTU_Wed02Mar2011_16-15-50_ppa_svmData.txt
# [0.01,1,10,100,1000] -- C : 100
# [60,80,100,200,300,400] -- C : 400 
# [340,35,360,370,380,390,400,410,420,430,440,450] -- C : 430
# [425,426,227,428,429,430,431,432,433,434,435,436] -- C : 436
# [435,436,437,438,438] -- C : 437
# [436.5,437,437.5] -- C : 437.5
# list(np.arange(437.1,437.99,.01)) -- C : 437.74

# '1_30_vec_test_TESTSTIxTESTSTU_Wed02Mar2011_20-40-13rPRC_block_svmData.txt'
# [0.1,1,10,100,1000] -- C: 1.000
# [0.5,1,2,3,4,5,6] -- C: 4
# list(np.arange(3.51,4.49,.01)) -- C: 4.3

# '1_30_vec_test_TESTNSUxTESTSTU_Wed02Mar2011_20-40-50rPRC_block_svmData.txt'
# [0.1,1,10,100,1000] -- C: 1000
# [200,300,400,500,600,700,800,900,1000,1100] --C: 900
# [860,870,880,890,900,910,920,930,940,950,960] -- C: 930.000000
# [925,926,927,928,929,930,931,932,933,934,936,936] -- C: 932
# list(np.arange(931.5,932.6,.01)) -- C: 931.700000

# 'vec_train_word_TESTSTUxTESTNSU_Wed02Mar2011_20-42-01rPRC_block_svmData.txt'
# [0.01,1,10,100,1000] -- C: 100
# [50,100,200,300,400,500,600] -- C: 600
# [550,600,650,700] -- C: 550
# [510,520,530,540] -- C: 520
# [515,516,517,518,519,520,521,522,523,524,525] -- C: 524.000000
# possibleC =  list(np.arange(523.6,524.6,.01)) -- C: 523.830000
# 

# 'vec_train_scene_TESTSTUxTESTNSU_Wed02Mar2011_20-42-01rPRC_block_svmData.txt'
# [0.01,1,10,100,500,1000] -- C: 1
# [0.1,.2,.3,2,3,4,5,6] -- C: 0.3
# [0.2,.3,.4,.5,.6,.7] -- C: 0.4
# list(np.arange(.3,.5,.01)) -- C: 0.31000 # CV RUNNING

trainFile = 'vec_train_word_TESTSTUxTESTNSU_Wed02Mar2011_20-42-01rPRC_block_svmData.txt'

#possibleC = [515,516,517,518,519,520,521,522,523,524,525]
possibleC =  list(np.arange(523.6,524.6,.01))
trainData = ml.VectorDataSet(trainFile,labelsColumn=1,idColumn=0) 
# assumes data is csv and znormed

startTime = strftime("%a%d%b%Y_%H:%M:%S")

s     = ml.SVM()
param = ml.modelSelection.Param(s, 'C', possibleC)
m     = ml.modelSelection.ModelSelector(param)
m.train(trainData)

stopTime = strftime("%a%d%b%Y_%H:%M:%S")

print(startTime,stopTime)
print(m)
