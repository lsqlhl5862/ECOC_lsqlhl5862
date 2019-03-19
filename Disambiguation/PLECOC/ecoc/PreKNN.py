from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import libsvm
from sklearn import preprocessing
from FeatureSelection.FeatureSelector import BSSWSS

class PreKNN:
    
    def __init__(self,tr_labels):
        self.model=None
        self.tr_labels=tr_labels
        self.tr_data=None
        self.neighbors=None
        self.pos_cols_list=[]
        self.neg_cols_list=[]
        self.pre_knn_perfomance_matrix=None
        self.pre_knn_labels_matrix=None

    # def fit_predict(self,data,labels,test_data):
    #     model=KNeighborsClassifier(n_neighbors=self.performance_matrix.shape[0])
    #     temp_labels=[]
    #     for i in range(data.shape[0]):
    #         temp_labels.append(np.argmax(labels[:, i]))
    #     temp_labels=np.array(temp_labels)
    #     model.fit(data,temp_labels)
    #     temp=model.kneighbors(test_data)
    #     print(temp[1])
    #     self.model=model
    #     self.tr_data=data
    #     score=self.model.score(data,temp_labels)
    #     print(score)
    #     # self.single_decode(test_data[0,:],self.pre_labels[0])


    # def single_decode(self,data,label):
    #     performance_matrix=self.performance_matrix.copy()
    #     performance_row=performance_matrix[label-1,:]
    #     for i in range(len(performance_row)):
    #         performance_flag=performance_row[i]
    #         temp=[]
    #         p_labels, _, _ = svm_predict([], data, self.svm_models[i])
    #         print(p_labels,performance_flag)
    
    def fit(self,data,labels):
        model=KNeighborsClassifier(n_neighbors=labels.shape[0])
        temp_labels=np.zeros((1,data.shape[0])).T
        self.labels=labels
        model.fit(data,temp_labels.ravel())
        self.model=model

    def predict(self,pre_data):
        self.neighbors=self.model.kneighbors(pre_data)
        for i in range(pre_data.shape[0]):
            temp_distances=self.neighbors[0][i]
            temp_indexs=self.neighbors[1][i]
            temp_pre_labels_matrix=np.zeros((1,self.tr_labels.shape[0])).T
            temp_pre_distances_matrix=np.zeros((1,self.tr_labels.shape[0])).T
            for j in range(len(temp_indexs)):
                for index in np.where(self.tr_labels[:,temp_indexs[j]]==1)[0]:
                    temp_pre_distances_matrix[index][0]+=temp_distances[j]
                    temp_pre_labels_matrix[index][0]+=1
            # print(np.where(temp_pre_labels_matrix==temp_pre_labels_matrix.max()))
            self.pre_knn_perfomance_matrix=temp_pre_distances_matrix if self.pre_knn_perfomance_matrix is None else np.hstack((self.pre_knn_perfomance_matrix,temp_pre_distances_matrix))
            self.pre_knn_labels_matrix=temp_pre_labels_matrix if self.pre_knn_labels_matrix is None else np.hstack((self.pre_knn_labels_matrix,temp_pre_labels_matrix))
        self.pre_knn_perfomance_matrix=preprocessing.MinMaxScaler().fit_transform(self.pre_knn_perfomance_matrix)
        
        self.pre_label_matrix = np.zeros((self.tr_labels.shape[0], pre_data.shape[0]))
        for i in range(pre_data.shape[0]):
            idx = self.pre_knn_labels_matrix[:, i] == max(self.pre_knn_labels_matrix[:, i])
            self.pre_label_matrix[idx, i] = 1
    
    def getPreKnnMatrix(self):
        return self.pre_knn_perfomance_matrix

    def getPredictMatrix(self):
        return self.pre_label_matrix

class PLFeatureSelection:
    
    def __init__(self,num_features):
        self.num_features=num_features
        self.fs_model=None

    def fit(self,data,labels):
        # coding_col=self.coding_col.tolist()
        # for i in range(data.shape[0]):
        #     temp_labels=np.where(labels[i,:]==1)[0]
        #     num_pos=0
        #     num_neg=0
        #     for class_index in temp_labels:
        #         if(coding_col[class_index]==1):
        #             num_pos+=1
        #         elif(coding_col[class_index]==-1):
        #             num_neg+=1
        #     print(num_pos)
        #     print(num_neg)
        self.fs_model = BSSWSS(k=self.num_features)  # remain 2 features.
        self.fs_model.fit(data, labels)
    
    def transform(self,data):
        return self.fs_model.transform(data)
