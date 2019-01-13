# -*- coding: utf-8 -*-
from sklearn.svm import SVC
import os
import numpy as np
from consts import *
from sklearn.externals import joblib


def train_svm():
    train_data = np.load(os.path.join(Processed_Data_Dir,"train.npz"))
    train_targets=train_data["arr_0"]
    train_size=train_targets.shape[0]
    print ("training set size=%d"%train_size)
    train_set=[]
    for i in range(train_size):
        train_set.append(train_data["arr_%d"%(i+1)])
    clf=SVC(C=100,kernel='rbf',gamma=1e-3)    #C,kernel,gamma,coef0,degree
    clf.fit(train_set,train_targets)
    joblib.dump(clf,os.path.join(Model_Dir,"svm_model.m"))
    print("SVM model trained and saved successfully.")
        
    
if __name__=="__main__":
    train_svm()