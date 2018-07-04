import numpy as numpy
import CodeMatrix.CodeMatrix
# from CodeMatrix.CodeMatrix import ovo,ova,dense_rand,sparse_rand,decoc,agg_ecoc,cl_ecoc,ecoc_one
from FeatureSelection.FeatureSelector import BSSWSS
import DataLoader as dl
from Classifiers.ECOCClassifier import SimpleECOCClassifier
from Classifiers.BaseClassifier import get_base_clf
import sklearn.metrics as metrics
from CodeMatrix.SFFS import sffs
from matplotlib import pyplot as plt
plt.style.use('ggplot')

trainX, trainY, testX, testY, instanceNum = dl.loadDataset(
    'data_uci/'+'abalone_train.data', 'data_uci/'+'abalone_test.data')


def getAccuracyByEncoding(encoding, feature, trainX, trainY, testX, testY):
    fs = BSSWSS(k=feature)  # remain 2 features.o
    fs.fit(trainX, trainY)
    trainX, testX = fs.transform(trainX), fs.transform(testX)
    codeMatrix = getattr(CodeMatrix.CodeMatrix, encoding)(trainX, trainY)[0]
    estimator = get_base_clf('SVM')  # get SVM classifier object.
    sec = SimpleECOCClassifier(estimator, codeMatrix)
    sec.fit(trainX, trainY)
    pred = sec.predict(testX)
    return metrics.accuracy_score(testY,pred), codeMatrix


def getAllAccuracy(dataName):
    encodingList = ("ovo", "ova", "dense_rand",
                    "sparse_rand", "decoc", "agg_ecoc")
    trainX, trainY, testX, testY, instanceNum = dl.loadDataset(
        'data_uci/'+dataName+'_train.data', 'data_uci/'+dataName+'_test.data')
    for item in encodingList:
        pred, codeMatrix = getAccuracyByEncoding(
            item, len(trainX[0]), trainX, trainY, testX, testY)
        print(pred)
        print(codeMatrix)

getAllAccuracy("abalone")
