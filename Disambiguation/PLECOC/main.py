from ecoc import Rand
from ecoc import BottomUp
import numpy as np
import random
import time
from scipy import io
from svmutil import *
from tools import Tools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import preprocessing
from matplotlib import pyplot as plt
from ecoc.PreKNN import PreKNN


def read_mat(filepath, tr_key='train_data', tr_label_key='train_p_target',
             ts_key='test_data', ts_label_key='test_target'):
    if not filepath.endswith('mat'):
        raise NameError('file %s not a mat file', filepath)
    mat = io.loadmat(filepath)
    tr_data = mat[tr_key]
    print(tr_data)
    tr_labels = mat[tr_label_key]
    print(tr_labels)
    ts_data = mat[ts_key]
    ts_labels = mat[ts_label_key]
    return tr_data, tr_labels, ts_data, ts_labels


def run_sample():
    accuracies = []
    ite = 10
    i = 0
    while i != ite:
        filepath = 'mat/sample data.mat'
        tr_data, tr_labels, ts_data, ts_labels = read_mat(filepath)
        pl_ecoc = Rand.RandPLECOC(libsvm, svm_param='-t 2 -c 1')
        pl_ecoc.fit(tr_data, tr_labels)
        pre_label_matrix, accuracy = pl_ecoc.predict(ts_data, ts_labels)
        accuracies.append(accuracy)
        i = i+1
    print(accuracies)
    print('mean = ' + str(np.mean(accuracies)))
    print('max = ' + str(max(accuracies)))
    print('min = ' + str(min(accuracies)))


def run_birdsong():
    start = time.time()
    accuracies = []
    ite = 20
    i = 0
    name = 'pl'
    # mat_list=["MSRCv2","lost","BirdSong"]
    mat_list=["MSRCv2",]
    for item in mat_list:
        accuracies=None
        for i in range(ite):
            filepath = "mat/"+item+".mat"
            # filepath = 'mat/BirdSong.mat'        
            mat = io.loadmat(filepath)
            tr_data = mat['data']
            print(tr_data.shape)
            tr_labels = mat['partial_target'].toarray()
            tr_labels = tr_labels.astype(np.int)
            # print(tr_labels)
            print(tr_labels.shape)


            # draw_hist(tr_labels.sum(axis=1).tolist(), 'class_distribution', 'class', 'number', 0, tr_labels.shape[0], 0, tr_labels.shape[1])
            true_labels = mat['target'].toarray()
            true_labels = true_labels.astype(np.int)
            tr_data=preprocessing.StandardScaler().fit_transform(tr_data)
            # tr_data=preprocessing.MinMaxScaler().fit_transform(tr_data)
            tr_idx, ts_idx,tv_idx = Tools.tr_ts_split_idx(tr_data)
            split_tr_data, split_ts_data,split_tv_data = tr_data[tr_idx], tr_data[ts_idx],tr_data[tv_idx]
            split_tr_labels, split_ts_labels,split_tv_labels = tr_labels[:,
                                                        tr_idx], true_labels[:, ts_idx],true_labels[:,tv_idx]
            
            #测试PreKNN
            pre_knn=PreKNN(split_tr_labels,split_tv_data,split_tv_labels)
            pre_knn.fit(split_tr_data,split_tr_labels)
            # pre_knn.predict(split_ts_data)
            
            # tr_data, tr_labels, ts_data, ts_labels = read_mat(filepath, tr_key='data', tr_label_key='partial_target')
            pl_ecoc = Rand.RandPLECOC(libsvm, svm_param='-t 2 -c 1')
            # pl_ecoc.fit(split_tr_data, split_tr_labels)

            # pre_label_matrix,base_accuracy,knn_accuracy,com_accuracy = pl_ecoc.predict(
            #     split_ts_data, split_ts_labels,pre_knn)

            result=pl_ecoc.fit_predict(split_tr_data, split_tr_labels,split_ts_data, split_ts_labels,split_tv_data,split_tv_labels,pre_knn)
            result=np.array(result).T
            accuracies=result if accuracies is None else np.vstack((accuracies,result))
            # for index in range(len(result)):
            #     print(str(index)+": "+str(result[index]))
            # knn_accuracy=result[1]
            # accuracies.append(result)
            # result=np.array(result)
            # result=result[:,0].T.tolist()
            # file_name=item+"_"+str(i+1)
            # draw_hist(file_name,result,"Knn_accuracy: "+str(knn_accuracy),"Features","Accuracy",0,1,0,1)
            # pl_ecoc.refit_predict(split_tr_data,split_tr_labels,split_ts_data,split_ts_labels,accuracy)
            # del pl_ecoc
            # accuracies.append(com_accuracy)
            # data_class = np.array(range(split_ts_labels.shape[0]))
            # ts_vector = np.dot(data_class, split_ts_labels)
            # pre_vector = np.dot(data_class, pre_label_matrix)
            # confusion = confusion_matrix(ts_vector.tolist(), pre_vector.tolist())
            # i = i+1
        file_name=item+"_mean"
        # accuracies=np.array(accuracies)
        for index in range(accuracies.shape[0]):
            print(str(index+1)+": "+str(accuracies[index,:]))
        for index in range(accuracies.shape[1]):
            print(str(index+1)+"列平均值: "+str(np.mean(accuracies[:,index])))
        # draw_hist(file_name,accuracies,item+"_mean:"+str(np.mean(accuracies))," ","Accuracy",0,1,0,1)
        # print(name + '_ECOC finish')
        # print('耗时: {:>10.2f} minutes'.format((time.time()-start)/60))
        # print(accuracies)
        # print('mean = ' + str(np.mean(accuracies)))
        # print('max = ' + str(max(accuracies)))
        # print('min = ' + str(min(accuracies)))


def draw_hist(file_name,myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    name_list = list(range(len(myList)))
    plt.figure()
    # name_list.reverse()
    rects = plt.bar(range(len(myList)), myList, color='rgby')
    # X轴标题
    index = list(range(len(myList)))
    # index = [float(c)+0.4 for c in range(len(myList))]
    plt.ylim(ymax=Ymax, ymin=Ymin)
    plt.xticks(index, name_list)
    plt.ylabel(Ylabel)  # X轴标签
    plt.xlabel(Xlabel)
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height,
                 str(height), ha='center', va='bottom')
    plt.title(Title)
    plt.savefig("pictures/"+file_name+".png")
    # plt.show()


if __name__ == '__main__':
    # run_sample()
    run_birdsong()
