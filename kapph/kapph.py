#import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split

import time

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r runtime %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed

class preProcessing(object):
       
    featureScaling = True
    dataSplit = True
    splitRatio = 0.2
    
    def __init__(self):
        """ Constructor Class for preprocessing """
        
    
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
    
    @timeit
    def printCols(self,data):
        #print("The number of NA values in the column")
        #print(data.isna().sum())
        objCol = list(data.select_dtypes(include = ['object']).columns)
        numCol = list(data.select_dtypes(include = ['float64','int64']).columns)
        columndetails = []
        for i in objCol:
            columndetails.append({'Column Name':i,'Type' : 'Object' ,'Number of NULL values': float(data[i].isnull().sum())})
        for i in numCol:
            columndetails.append({'Column Name':i,'Type' : 'Numeric' ,'Number of NULL values': float(data[i].isnull().sum())})
        return(pd.DataFrame(columndetails))
        
        
        
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
                
    def binning(self,data,col,valueList,labelNames):
        data[col] = pd.cut(data[col],valueList,labels = labelNames)
        data[col] = data[col].astype('object')
        return data
        
        
        
    def removeNull(self,data):
        nullCount = data.isnull().sum()
        data = data.dropna()
        return nullCount
    
    def oneHotEncoding(self,data):
        data = pd.get_dummies(data,drop_first = True)
        return data
    
        
        
########################################################
