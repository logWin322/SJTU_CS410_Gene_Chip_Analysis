# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from consts import *
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from scipy import stats

def train_knn():
    train_data = np.load(os.path.join(Processed_Data_Dir,"train.npz"))
    train_targets=train_data["arr_0"]
    train_size=train_targets.shape[0]
    print ("training set size=%d"%train_size)
    train_set=[]
    for i in range(train_size):
        train_set.append(train_data["arr_%d"%(i+1)])
    knn = KNeighborsClassifier(n_neighbors=3,p=1)
    knn.fit(train_set,train_targets)
    joblib.dump(knn,os.path.join(Model_Dir,"knn_model.m"))
    print("KNN model trained and saved successfully.")
        
    
if __name__=="__main__":
    train_knn()

