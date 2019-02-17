from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class DataDecode():
    
    def __init__(self):
        self.model=None
        self.performance_matrix=None
    
    def fit(self,data,labels,performance_matrix):
        model=KNeighborsClassifier(n_neighbors=performance_matrix.shape[0])
        self.performance_matrix=performance_matrix
        temp_labels=[]
        for i in range(data.shape[0]):
            temp_labels.append(np.argmax(labels[:, i]))
        temp_labels=np.array(temp_labels)
        model.fit(data,temp_labels)
        self.model=model
    
    def get_pre_labels(self,test_data):
        pre_labels=self.model.predict(test_data)
        print(pre_labels)