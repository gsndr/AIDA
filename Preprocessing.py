import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import scale



class Preprocessing():
    def __init__(self, train, test):
        #classification column
        self._clsTrain = train.columns[-1]

        #only categorical features
        obj_dfTrain = train.select_dtypes(include=['object']).copy()
        self._objectListTrain = obj_dfTrain.columns.values

        #remove classification column
        self._objectListTrain = np.delete(self._objectListTrain, -1)
        #classification test column
        self._clsTest = test.columns[-1]

        # ronly categorical features test set
        obj_dfTest = test.select_dtypes(include=['object']).copy()
        self._objectListTest= obj_dfTest.columns.values

       #remove classification column from test set
        self._objectListTest = np.delete(self._objectListTest, -1)


    '''   def __init__(self):
        print("Preprocessing void")
        '''


    def getCls(self):
        return self._clsTrain, self._clsTest

    #one-hot encoding function
    def preprocessingOneHot(self,train,test):
        #print(self._objectListTrain)
        train = pd.get_dummies(train, columns=self._objectListTrain)
        #print(self._objectListTest)
        test = pd.get_dummies(test, columns=self._objectListTest)
        return  train, test



    def mapLabel(self, df):
        # creating labelEncoder
        le = preprocessing.LabelEncoder()
        # Converting string labels into numbers.
        cls_encoded = le.fit_transform(df[self._clsTrain])
       # print(list(le.classes_))
       # print("Econde", cls_encoded)
        df[self._clsTrain] = le.transform(df[self._clsTrain])
        return cls_encoded

    #map label classification to number
    def preprocessinLabel(self,train, test):
        distinctLabels1 = train[self._clsTrain].unique().tolist()
        distinctLabels1=sorted(distinctLabels1)
        #print(distinctLabels1)
        cls_encoded = self.mapLabel(train)
        cls_encoded2 = self.mapLabel(test)
        return train, test





    def minMaxScale(self, Y_train, Y_test):
        scaler = preprocessing.MinMaxScaler()
        Y_train=scale(Y_train)
        Y_test=scale(Y_test)
        return Y_train, Y_test


    def getXY(self, train, test):
        clssList = train.columns.values
        target = [i for i in clssList if i.startswith(' classification')]

        # remove label from dataset to create Y ds
        train_Y=train[target]
        test_Y=test[target]
        print("test y s", test_Y.shape)
        # remove label from dataset
        train_X = train.drop(target, axis=1)
        train_X=train_X.values
        test_X = test.drop(target, axis=1)
        test_X=test_X.values

        return train_X, train_Y, test_X, test_Y
