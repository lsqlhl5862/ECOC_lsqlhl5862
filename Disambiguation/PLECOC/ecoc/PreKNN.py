from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import libsvm
from sklearn import preprocessing
from FeatureSelection.FeatureSelector import BSSWSS
from sklearn.svm import libsvm
from svmutil import *
import sklearn.metrics as metrics

class PreKNN:
    
    def __init__(self,tr_labels,tv_data,tv_labels):
        self.model=None
        self.tr_labels=tr_labels
        self.tv_data=tv_data
        self.tv_labels=tv_labels
        self.tr_data=None
        self.pos_cols_list=[]
        self.neg_cols_list=[]

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

    def predict(self,pre_data,true_labels):
        neighbors=self.model.kneighbors(pre_data)
        pre_knn_labels_matrix=None
        for i in range(pre_data.shape[0]):
            temp_distances=neighbors[0][i]
            temp_indexs=neighbors[1][i]
            distances_sum=np.sum(temp_distances)
            temp_pre_labels_matrix=np.zeros((1,self.tr_labels.shape[0])).T
            temp_pre_distances_matrix=np.zeros((1,self.tr_labels.shape[0])).T
            temp_pre_weight=np.zeros((1,len(temp_indexs))).T
            for j in range(len(temp_indexs)):
                temp_pre_weight[j]=1-temp_distances[j]/distances_sum
                for index in np.where(self.tr_labels[:,temp_indexs[j]]==1)[0]:
                    temp_pre_distances_matrix[index][0]+=temp_distances[j]
                    temp_pre_labels_matrix[index][0]+=temp_pre_weight[j]
            # print(np.where(temp_pre_labels_matrix==temp_pre_labels_matrix.max()))
            # self.pre_knn_perfomance_matrix=temp_pre_distances_matrix if self.pre_knn_perfomance_matrix is None else np.hstack((self.pre_knn_perfomance_matrix,temp_pre_distances_matrix))
            pre_knn_labels_matrix=temp_pre_labels_matrix if pre_knn_labels_matrix is None else np.hstack((pre_knn_labels_matrix,temp_pre_labels_matrix))
        # self.pre_knn_perfomance_matrix=preprocessing.MinMaxScaler().fit_transform(self.pre_knn_perfomance_matrix)
        # self.pre_knn_perfomance_matrix=preprocessing.MinMaxScaler().fit_transform(self.pre_knn_labels_matrix)
        pre_knn_perfomance_matrix=preprocessing.StandardScaler().fit_transform(pre_knn_labels_matrix)
        pre_knn_perfomance_matrix=preprocessing.MinMaxScaler().fit_transform(pre_knn_perfomance_matrix)
        
        pre_label_matrix = np.zeros((self.tr_labels.shape[0], pre_data.shape[0]))
        for i in range(pre_data.shape[0]):
            idx = pre_knn_labels_matrix[:, i] == max(pre_knn_labels_matrix[:, i])
            pre_label_matrix[idx, i] = 1
        
        count = 0
        for i in range(pre_data.shape[0]):
            max_idx1 = np.argmax(pre_label_matrix[:, i])
            max_idx2 = np.argmax(true_labels[:, i])
            if max_idx1 == max_idx2:
                count = count+1
        knn_accuracy = count / pre_data.shape[0]
        
        return pre_label_matrix,knn_accuracy,pre_knn_perfomance_matrix

    
    def getValidationData(self):
        return self.tv_data,self.tv_labels

class PLFeatureSelection:
    
    score_list=[]

    def __init__(self,num_features):
        self.num_features=num_features
        self.fs_model=None

    def code_fit(self,data,labels,tv_data,tv_labels,coding_col,params):
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
        tv_pos_idx=[]
        tv_neg_idx=[]
        tv_data_flag=np.zeros(tv_data.shape[0])
        coding_col[np.where(coding_col==-1)[0]]=0
        #     print(num_neg)
        for j in range(tv_data.shape[0]):
            if np.all((tv_labels[:, j] & coding_col) == tv_labels[:, j]):
                tv_pos_idx.append(j)
                tv_data_flag[j]+=1
            else:
                if np.all((tv_labels[:, j] & np.int8(np.logical_not(coding_col))) == tv_labels[:, j]):
                    tv_neg_idx.append(j)
                    tv_data_flag[j]+=1
        print(len(np.where(tv_data_flag==0)[0]))
        pos_inst = tv_data[tv_pos_idx]
        neg_inst = tv_data[tv_neg_idx]
        tv_inst = np.vstack((pos_inst, neg_inst))
        tv_labels = np.hstack(
            (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))

        acc_list=np.zeros((self.num_features,4))
        for i in range(self.num_features):
            fs_model = BSSWSS(k=self.num_features-i)  # remain 2 features.
            fs_model.fit(data, labels)
            prob = svm_problem(labels.tolist(),
                               fs_model.transform(data).tolist())
            param = svm_parameter(params.get('svm_param'))
            model = svm_train(prob, param)
            tmp_tv_inst=fs_model.transform(tv_inst)
            p1, p2, p3 =svm_predict(
                tv_labels, tmp_tv_inst.tolist(), model)
        #计算混淆矩阵
            # tp=0
            # fp=0
            # tn=0
            # fn=0
            # for j in range(len(p1)):
            #     if p1[j]==1 and tv_labels[j]==1:
            #         tp+=1
            #     elif p1[j]==1 and tv_labels[j]==-1:
            #         fp+=1
            #     elif p1[j]==-1 and tv_labels[j]==-1:
            #         tn+=1
            #     else:
            #         fn+=1
            # p=1
            accuracy=p2[0]
            # precision=tp / (tp + fp) if (tp+fp)!=0 else 0
            # recall=tp/(tp+fn) if (tp+fn)!=0 else 0
            # fscore=(1+p*p)*(precision*recall)/(p*p*(precision+recall)) if (precision+recall)!=0 else 0
            
            acc_list[i][0]=accuracy
            acc_list[i][1]=metrics.precision_score(tv_labels,p1)
            acc_list[i][2]=metrics.recall_score(tv_labels,p1)
            acc_list[i][3]=metrics.f1_score(tv_labels,p1)
        print(str(np.argmax(acc_list[:,0]))+"："+str(acc_list[:,0].max()))
        print(str(np.argmax(acc_list[:,1]))+"："+str(acc_list[:,1].max()))
        print(str(np.argmax(acc_list[:,2]))+"："+str(acc_list[:,2].max()))
        print(str(np.argmax(acc_list[:,3]))+"："+str(acc_list[:,3].max()))
        self.fs_model=BSSWSS(k=self.num_features-np.argmax(acc_list[:,3]))
        self.fs_model.fit(data,labels)

    def fit(self,data,labels,tv_data,tv_labels,coding_col,params):
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
        tv_pos_idx=[]
        tv_neg_idx=[]
        tv_data_flag=np.zeros(tv_data.shape[0])
        coding_col[np.where(coding_col==-1)[0]]=0
        #     print(num_neg)
        for j in range(tv_data.shape[0]):
            if np.all((tv_labels[:, j] & coding_col) == tv_labels[:, j]):
                tv_pos_idx.append(j)
                tv_data_flag[j]+=1
            else:
                if np.all((tv_labels[:, j] & np.int8(np.logical_not(coding_col))) == tv_labels[:, j]):
                    tv_neg_idx.append(j)
                    tv_data_flag[j]+=1
        print(len(np.where(tv_data_flag==0)[0]))
        pos_inst = tv_data[tv_pos_idx]
        neg_inst = tv_data[tv_neg_idx]
        tv_inst = np.vstack((pos_inst, neg_inst))
        tv_labels = np.hstack(
            (np.ones(len(pos_inst)), -np.ones(len(neg_inst))))

        acc_list=np.zeros((self.num_features,4))
        for i in range(self.num_features):
            fs_model = BSSWSS(k=self.num_features-i)  # remain 2 features.
            fs_model.fit(data, labels)
            prob = svm_problem(labels.tolist(),
                               fs_model.transform(data).tolist())
            param = svm_parameter(params.get('svm_param'))
            model = svm_train(prob, param)
            tmp_tv_inst=fs_model.transform(tv_inst)
            p1, p2, p3 =svm_predict(
                tv_labels, tmp_tv_inst.tolist(), model)
        #计算混淆矩阵
            # tp=0
            # fp=0
            # tn=0
            # fn=0
            # for j in range(len(p1)):
            #     if p1[j]==1 and tv_labels[j]==1:
            #         tp+=1
            #     elif p1[j]==1 and tv_labels[j]==-1:
            #         fp+=1
            #     elif p1[j]==-1 and tv_labels[j]==-1:
            #         tn+=1
            #     else:
            #         fn+=1
            # p=1
            accuracy=p2[0]
            # precision=tp / (tp + fp) if (tp+fp)!=0 else 0
            # recall=tp/(tp+fn) if (tp+fn)!=0 else 0
            # fscore=(1+p*p)*(precision*recall)/(p*p*(precision+recall)) if (precision+recall)!=0 else 0
            
            acc_list[i][0]=accuracy
            acc_list[i][1]=metrics.precision_score(tv_labels,p1)
            acc_list[i][2]=metrics.recall_score(tv_labels,p1)
            acc_list[i][3]=metrics.f1_score(tv_labels,p1)
        print(str(np.argmax(acc_list[:,0]))+"："+str(acc_list[:,0].max()))
        print(str(np.argmax(acc_list[:,1]))+"："+str(acc_list[:,1].max()))
        print(str(np.argmax(acc_list[:,2]))+"："+str(acc_list[:,2].max()))
        print(str(np.argmax(acc_list[:,3]))+"："+str(acc_list[:,3].max()))
        self.fs_model=BSSWSS(k=self.num_features-np.argmax(acc_list[:,3]))
        self.fs_model.fit(data,labels)
    
    def transform(self,data):
        return self.fs_model.transform(data)
