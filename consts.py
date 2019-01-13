d # -*- coding: utf-8 -*-
Data_Dir="Gene_Chip_Data"
Sample_Feature_Dir="sample_feature"
Label_Dir="sample_label"
Processed_Data_Dir="processed_data"
Model_Dir="model"
Select_Label_Index=0      # 0 for Disease State ; 1 for Sex ; 2 for BioSourceType
Selected_Label=["Characteristics [DiseaseState]","Characteristics [Sex]","Characteristics [BioSourceType]"]
Feature_Size=22283           # original data dimension
Sample_Size=5896             #number of samples
Valid_Sample_Size=[3558,1675,3087]   #remove samples whose selected label value = NULL. Also, For disease_state: remove those samples with disease state occuring less than Thershold Times
Thershold=10              #occuring frequency thershold (e.g. occur less than 10 times then remove those samples)
Select_Percentage=0.9     #0.9 22283 reduced to 453
Reduced_Feature_Size=453  # dimension reduced after PCA
Test_Ratio=0.1            # training set size: test set size= 9:1


# Thershold=8,Valid_Sample_Size=3728 
# Thershold=0,Valid_Sample_Size=4028 
# Thershold=10,Valid_Sample_Size=3558
# Thershold=15,Valid_Sample_Size=3297
# Thershold=11,Valid_Sample_Size=3438
# Thershold=9,Valid_Sample_Size=3648