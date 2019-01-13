# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import random
from consts import *

def read_raw_data():
    with open(os.path.join(Data_Dir,"microarray.original.txt"),'r',encoding='utf-8') as f_in:
        lines=f_in.readlines()
        samples=lines[0][:-1].split('\t')[1:]  # first line sample name
        #print (samples)
        #print (len(samples))
        samples_feature=np.zeros((len(samples),Feature_Size))
        probe_index=0
        for line in lines[1:]:
            probe_data=line[:-1].split('\t')[1:]
            #print (len(probe_data))
            for i in range(len(probe_data)):
                samples_feature[i][probe_index]=probe_data[i]
            #print (samples_feature[:,0])
            print ("line %d"%probe_index)
            probe_index=probe_index+1
    #print (samples_feature[0])
    np.save(os.path.join(Sample_Feature_Dir,"samples.npy"),samples_feature)
    print ("Samples saved.")

def filter_disease(disease_times,disease_dict,thershold):
    return {disease:index_list for disease,index_list in disease_dict.items() if disease_times[disease]>=thershold},{disease:times for disease,times in disease_times.items() if times>=thershold}


def merge_disease(disease_times,disease_dict):
    mer_diseases = ["breast tumor ","B-cell lymphoma ","lung adenocarcinoma (NCI_Thesaurus C0152013)","Huntington's Disease(HD)"]
    prefix_diseases=["breast tumor","B-cell lymphoma","lung adenocarcinoma (NCI_Thesaurus C0152013)","Huntington's"]    
    del_diseases=[]
    for i in range(len(mer_diseases)):
        disease_times[mer_diseases[i]]=0
        disease_dict[mer_diseases[i]]=[]
        for  disease in disease_dict:
            if prefix_diseases[i] in disease and disease!=mer_diseases[i]:
                #print (disease)
                del_diseases.append(disease)                
                disease_dict[mer_diseases[i]]=disease_dict[mer_diseases[i]]+disease_dict[disease]
                disease_times[mer_diseases[i]]=disease_times[mer_diseases[i]]+disease_times[disease]
    mer_disease_one="acute lymphoblastic leukemia"
    mer_disease_two='"acute lymphoblastic leukemia, chemotherapy response"'
    disease_times[mer_disease_one]=disease_times[mer_disease_one]+disease_times[mer_disease_two]
    disease_dict[mer_disease_one]=disease_dict[mer_disease_one]+disease_dict[mer_disease_two]
    del_diseases.append(mer_disease_two)
    for del_disease in del_diseases:
        disease_times.pop(del_disease)
        disease_dict.pop(del_disease)
    return disease_dict,disease_times        


def read_sdrf_file():
    with open(os.path.join(Data_Dir,"E-TABM-185.sdrf.txt"),'r',encoding='utf-8') as f_in:
        lines=f_in.readlines()
        attributes=lines[0][:-1].split('\t')    # first line attribute name
        attribute_dict={}
        type_times={}
        type_indexes={}
        for i in range(len(attributes)):
            attribute_dict[attributes[i]]=i
        index=0
        for line in lines[1:]:
            line_data=line[:-1].split('\t')
            if (line_data[attribute_dict[Selected_Label[Select_Label_Index]]]!='  '):
                attribute_type=line_data[attribute_dict[Selected_Label[Select_Label_Index]]]
                #print (line_data[attribute_dict[Selected_Label[Select_Label_Index]]])
                if attribute_type not in type_times:
                    type_times[attribute_type]=1
                    type_indexes[attribute_type]=[index]        
                else:
                    type_times[attribute_type]=type_times[attribute_type]+1
                    type_indexes[attribute_type].append(index)
            index=index+1 
        if Select_Label_Index==0:                  #disease_state
            merge_type_indexes,merge_type_times=merge_disease(type_times,type_indexes)
            filter_type_indexes,filter_type_times=filter_disease(merge_type_times,merge_type_indexes,Thershold)
            return filter_type_indexes,filter_type_times
        else:                              #other labels
            return type_indexes,type_times


def features_pca():
    samples_feature=np.load(os.path.join(Sample_Feature_Dir,"samples.npy"))
    pca=PCA(n_components=Select_Percentage)
    pca.fit(samples_feature)
    samples_feat_reduced=pca.transform(samples_feature)  
    np.save(os.path.join(Sample_Feature_Dir,"samples_reduced.npy"),samples_feat_reduced)
    print ("PCA is done and reduced dimension feature data is saved.")
    
    #show it on the graph
    '''
    plt.figure(1, figsize=(8, 6))  
    plt.clf()  
    plt.plot(pca.explained_variance_[0:1000])
    plt.axis('tight') 
    plt.xlim(300,600)
    plt.ylim(3,7.5)
    plt.xlabel('n_components')  
    plt.ylabel('explained_variance_')  
    plt.show() 
    '''

def produce_label_lst():
    type_indexes,type_times=read_sdrf_file()
    type_label={}
    label=1
    for type_ in type_indexes:
        type_label[type_]=label
        label=label+1
    #print (type_times)
    print (type_label)
    label_lst=np.zeros(Sample_Size)
    for type_ in type_indexes.keys():
        #print (type_)
        #print (type_label[disease])
        for i in type_indexes[type_]:
            label_lst[i]=int(type_label[type_])
    np.save(os.path.join(Label_Dir,'labels.npy'),label_lst)
    print ('sample label lst saved.')


def produce_dataset():
    samples=np.load(os.path.join(Sample_Feature_Dir,"samples_reduced.npy"))
    labels=np.load(os.path.join(Label_Dir,"labels.npy"))
    type_indexes,type_times=read_sdrf_file()
    sum_=0
    for v in type_times:
        sum_=sum_+type_times[v]
    print ("sample size=%d"%sum_)
    data_set=np.zeros((Valid_Sample_Size[Select_Label_Index],Reduced_Feature_Size))
    targets=np.zeros(Valid_Sample_Size[Select_Label_Index])

    index=0
    for disease in type_indexes.keys():
        for i in type_indexes[disease]:
            data_set[index]=samples[i]
            targets[index]=int(labels[i])
            #print (targets[index])
            index=index+1
    test_index = random.sample(range(Valid_Sample_Size[Select_Label_Index]), int(Test_Ratio*Valid_Sample_Size[Select_Label_Index]))
    train_index=list(filter(lambda x:x not in test_index, range(Valid_Sample_Size[Select_Label_Index])))
    test_set = list(map(lambda x:data_set[x], test_index))
    test_targets=list(map(lambda x:targets[x],test_index))
    train_set=list(map(lambda x:data_set[x],train_index))
    train_targets=list(map(lambda x:targets[x],train_index))
    
    np.savez(os.path.join(Processed_Data_Dir, 'train.npz'), train_targets , *train_set)
    np.savez(os.path.join(Processed_Data_Dir, 'test.npz'), test_targets , *test_set)
    print ("Data set has been saved.")
    
    
    
if __name__=="__main__":
    '''
    read_raw_data()
    test=np.load(os.path.join(Sample_Feature_Dir,"samples.npy"))
    print (test[1])
    features_pca()
    test=np.load(os.path.join(Sample_Feature_Dir,"samples_reduced.npy"))
    '''
    produce_label_lst()
    produce_dataset()

    