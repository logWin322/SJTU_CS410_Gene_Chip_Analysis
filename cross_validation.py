# -*- coding: utf-8 -*-
from test_svm import test_svm
from preprocess_data import produce_dataset
from train_svm import train_svm
from train_knn import train_knn
from test_knn import test_knn
from train_LR import train_LR
from test_LR import test_LR



if __name__=="__main__":
    accumlate=0
    times=10  
    for i in range(times):   
        produce_dataset()
        train_svm()
        print ("round %d:"%(i+1))
        precision=test_svm()
        accumlate=accumlate+precision
        print ('\n')
    print("The final precision: {:.2f}%".format((accumlate/times)*100))
