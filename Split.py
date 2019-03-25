from InfoGain import InfoGain as ig
import scipy.stats as ss
import pandas as pd
from collections import Counter
from math import log2
import numpy as np
import time




def attacks(InfoGain):
    totRows=InfoGain._totalRows
    dfShape=InfoGain._df.shape
    return(dfShape[0])

def labelForFeatures(feature, df, distinctLabels,cls):

    data_feature={}
    unique_values=df[feature].unique().tolist()
    df_feature=df.groupby([feature, cls]).size()
    index=df[feature].unique()
    #print("Index:", index)
    n = df_feature.count()
   # print(np.reshape(df_feature, n, len(unique_values)))
    data=[]
    for i in index:
        d = []
        df_extract = df_feature[i]
        #print(i, df_extract)
        indexV=df_extract.index.values
        for dl in distinctLabels:
            if dl in indexV:
               # print(dl)
                d.append(df_extract[dl])
            else:
                d.append(0)
       # print(d)
        data.append(d)
    #print(data)
    return data



def entropyForEachL(df, cls, listFeatures):
    data_df_attribute = {}
    # numbers of distinct classification (normal, dos, probe, U2R, R2L)
    distinctLabels = df[cls].unique().tolist()

    # count for each classfication the numbers of rows in the dataset
    countsLabel = pd.value_counts(df[cls].values, sort=False);
    for field in listFeatures:
        df_temp=df[[field, cls]]
        data=labelForFeatures(field,df_temp,distinctLabels,cls)
        infValue=ig.gain(countsLabel,data)
        data_df_attribute[field]=infValue
    return data_df_attribute

def featureSelection(df,list, cls,number, path):
    for n in number:
        listP=list[:n]
        print(cls)
        listP.append(cls)
        print(listP)
        dfPartition=df[listP]
        pathOutput=path+str(n)+"Features.csv"
        dfPartition.to_csv(pathOutput, index=0)


def main():
    pd.set_option('display.expand_frame_repr', False)
    tic=time.time()
    path='KDDTrain+aggregateOneCls';
    file2write = open("infoGainOneCls.txt", 'w');
    if(path==None):
        path=input("inserisci il path:")

    #crea oggetto della classe infoGain
    infoG=ig(path)
    infoG.extractVariables()
    cls =' classification.'
    print(cls)


    # numbers of distinct classification (normal, dos, probe, U2R, R2L)
    distinctLabels=infoG._df[cls].unique().tolist()
    print(distinctLabels)
    # count for each classfication the numbers of rows in the dataset
    countsLabel = pd.value_counts(infoG._df[cls].values, sort=False);


    print("Entropy of dataset is %f"  %ig.entropy(countsLabel))
    listFeatures=infoG._categories
    if cls in listFeatures:
        listFeatures.remove(cls)
    data_df_attribute = {}
    for field in listFeatures:
        print(field)
        df_temp=infoG._df[[field, cls]]
        data=labelForFeatures(field,df_temp,distinctLabels,cls)
       # print("finish", field)
        infValue=ig.gain(countsLabel,data)
        data_df_attribute[field]=infValue
    print("finish", field)
    file2write.write("Total information gain \n");
    #sort for infoGain
    data_df_attributeList = sorted(data_df_attribute, key=data_df_attribute.get, reverse=True)

    for element in data_df_attributeList:
        desc = "information gain for %s is %f" % (element, data_df_attribute[element]);
        print(desc)
        file2write.write(desc);
        file2write.write('\n');

    file2write.close()
    print("FInish")
    # Feature selection
    number = [10]
    df=pd.read_csv(path+".csv")
    featureSelection(df, data_df_attributeList,  cls, number, path)

    toc=time.time()
    print('TIme infoGain'+ str(toc-tic))

    #######same features on test set KDDTest+###############
    # per mostrare tutte le colonne pandas
    for n in number:
        pathTrain = pathOutput=path+str(n)+"Features"

        pathtest = 'KDDTest+aggregateOneCls'
        train_df = pd.read_csv(pathTrain + '.csv')
        test_df = pd.read_csv(pathtest + '.csv')
        columns = train_df.columns
        test_dfReduction = test_df[columns]
        print(columns)
        print(test_dfReduction.head(10))
        pathOutput = pathtest + str(n) + "Features.csv"
        test_dfReduction.to_csv(pathOutput, index=0)

        #######same features on test set KDDTest-21###############
        # per mostrare tutte le colonne pandas
    for n in number:
        pathTrain = pathOutput = path + str(n) + "Features"

        pathtest = 'KDDTest-21aggregateOneCls'
        train_df = pd.read_csv(pathTrain + '.csv')
        test_df = pd.read_csv(pathtest + '.csv')
        columns = train_df.columns
        test_dfReduction = test_df[columns]
        print(columns)
        print(test_dfReduction.head(10))
        pathOutput = pathtest + str(n) + "Features.csv"
        test_dfReduction.to_csv(pathOutput, index=0)




if __name__ == "__main__":
    main()