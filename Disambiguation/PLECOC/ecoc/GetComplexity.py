import numpy as numpy
import CodeMatrix.CodeMatrix
from DataComplexity.datacomplexity import get_data_complexity
from FeatureSelection.FeatureSelector import BSSWSS
import DataLoader as dl
from Classifiers.ECOCClassifier import SimpleECOCClassifier
from Classifiers.BaseClassifier import get_base_clf
import sklearn.metrics as metrics
from CodeMatrix.SFFS import sffs
from matplotlib import pyplot as plt

def getDataComplexitybyCol(trainX,trainY):
    temp=[]
    dc = get_data_complexity('F1')
    temp.append(round(dc.score(trainX,trainY),4))
    dc = get_data_complexity('F3')
    temp.append(round(dc.score(trainX,trainY),4))
    dc = get_data_complexity('F2')
    temp.append(round(dc.score(trainX,trainY),4))
    return temp