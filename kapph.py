#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split


class preProcessing(object):
       
    featureScaling = True
    dataSplit = True
    splitRatio = 0.2
    
    def __init__(self):
        """ Constructor Class for preprocessing """
        

    def dummyPrint(self):
        print("hello world karthick");
    
    def dataArray(self,data, yName = False):
        if yName == False:
            X = data.iloc[:, :-1].values
            y = data.iloc[:, -1].values
        else:
            y = data.iloc[:, data.columns == yName]
            X = data.iloc[:, data.columns != yName]
            
        return (X, y)

    def splitter(self,X,y,SR= splitRatio):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = SR, random_state = 0)
        return X_train, X_test, y_train, y_test
    
    def printCols(self,data):
        print("The number of null values in the column")
        print(data.isnull().sum())
        colTypes = set(data.dtypes.tolist())
        objCol = list(data.select_dtypes(include = ['object']).columns)
        numCol = list(data.select_dtypes(include = ['float64','int64']).columns)
        print(colTypes,"\n")
        print(objCol,"\n")
        print(numCol,"\n")
        
        
        
    def convertToObj(self,data,colToCon="all"):
        if colToCon == "all":
            for col in data.columns:
                data['col']  = data['col'].astype('object') 
        else:
            for col in colToCon:
                data[data.columns[col-1]] = data[data.columns[col-1]].astype('object')

    def convertToNum(self,data,colToCon="all"):
        if colToCon == "all":
            for col in data.columns:
                data['col']  = data['col'].astype('float64') 
        else:
            for col in colToCon:
                data[data.columns[col-1]] = data[data.columns[col-1]].astype('float64')
                
    def bucket(self,data,col,valueList,labelNames):
        data[col] = pd.cut(data[col],valueList,labels = labelNames)
        data[col] = data[col].astype('object')
        
        
        
    def removeNull(self,data):
        nullCount = data.isnull().sum()
        data = data.dropna()
        return nullCount
    
    def oneHotEncoding(self,data):
        data = pd.get_dummies(data,drop_first = True)
        return data
    
    
        
########################################################

        