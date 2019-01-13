# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from consts import *
from sklearn.externals import joblib

def train_LR():
    train_data = np.load(os.path.join(Processed_Data_Dir,"train.npz"))
    train_targets=train_data["arr_0"]
    train_size=train_targets.shape[0]
    print ("training set size=%d"%train_size)
    train_set=[]
    for i in range(train_size):
        train_set.append(train_data["arr_%d"%(i+1)])
    lr = LogisticRegression(C=0.01,solver='newton-cg')
    lr.fit(train_set,train_targets)
    joblib.dump(lr,os.path.join(Model_Dir,"lr_model.m"))
    print("Logistic Regression model trained and saved successfully.")
        
    
if __name__=="__main__":
    train_LR()

