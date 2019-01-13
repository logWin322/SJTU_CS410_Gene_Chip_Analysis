# -*- coding: utf-8 -*-
from sklearn.neighbors import KNeighborsClassifier
import os
import numpy as np
from consts import *
from sklearn.externals import joblib

def test_knn():
    test_data = np.load(os.path.join(Processed_Data_Dir,"test.npz"))
    test_targets=test_data["arr_0"]
    test_size=test_targets.shape[0]
    #print (test_size)
    test_set=[]
    model=joblib.load(os.path.join(Model_Dir,"knn_model.m"))
    for i in range(test_size):
        test_set.append(test_data["arr_%d"%(i+1)])    
    result=model.predict(test_set)
    correct_count=0
    for i in range(len(result)):
        #print (test_targets[i],result[i])
        if test_targets[i]==result[i]:
            correct_count=correct_count+1
    print("The precision: {:.2f}%".format((correct_count/test_size)*100))
    return correct_count/test_size
    

if __name__=="__main__":
    test_knn()

