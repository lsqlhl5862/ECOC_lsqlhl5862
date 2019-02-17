from ecoc import DataDecode
import numpy as np 

test_data=[[1,1,0,1],[2,0,1,4],[3,1,0,2]]
test_labels=[[1,2,3]]
test_data=np.array(test_data)
test_labels=np.array(test_labels)
datadecode=DataDecode.DataDecode()
datadecode.fit(test_data,test_labels.T.ravel(),test_data)