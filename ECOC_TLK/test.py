import numpy as numpy
from CodeMatrix.CodeMatrix import ovo,ova,dense_rand,sparse_rand,decoc,agg_ecoc,cl_ecoc,ecoc_one
from FeatureSelection.FeatureSelector import BSSWSS
import DataLoader as dl
from Classifiers.ECOCClassifier import SimpleECOCClassifier
from Classifiers.BaseClassifier import get_base_clf
import sklearn.metrics as metrics
from CodeMatrix.SFFS import sffs

trainX,trainY,testX,testY,instanceNum=dl.loadDataset('data_uci/'+'abalone_train.data','data_uci/'+'abalone_test.data')

fs = BSSWSS(k=8)  # remain 2 features.o
fs.fit(trainX, trainY)
trainX, testX = fs.transform(trainX), fs.transform(testX)

codeMatrix=cl_ecoc(trainX,trainY)[0]

estimator = get_base_clf('SVM')  # get SVM classifier object.

sec = SimpleECOCClassifier(estimator, codeMatrix)
sec.fit(trainX, trainY)
pred = sec.predict(testX)

print(metrics.accuracy_score(testY,pred))
# print(numpy.unique(testY))