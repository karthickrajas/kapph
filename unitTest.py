# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:04:27 2018

@author: Lenovo
"""
'''
import os
os.chdir('C:\\Users\\Lenovo\\Desktop\\ML\\preProcessingPackage')
'''


from kapph import preProcessing
pp = preProcessing()

import pandas as pd
import numpy as np

data = pd.read_csv("Data.csv")

pp = preProcessing()

print(pp.dummyPrint())

pp.printCols(data)

pp.convertToObj(data,[1,2])

pp.convertToNum(data,[2])

pp.bucket(data,'Age',[0,18,35,50],["young","medium","old"])

pd.cut(np.array([.2, 1.4, 2.5, 6.2, 9.7, 2.1]), 3)

print(pp.removeNull(data))

X,y = pp.dataArray(data = data,yName = "Purchased")

Xt,Xte,yt,yte = pp.splitter(X,y,0.2)