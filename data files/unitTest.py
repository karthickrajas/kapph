# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 16:04:27 2018

@author: Lenovo
"""
'''
import os
os.chdir('C:\\Users\\Lenovo\\Desktop\\ML\\preProcessingPackage\\kapph\\data files')
'''


from kapph.kapph import preProcessing

import pandas as pd

data = pd.read_csv("Data.csv")

pp = preProcessing()

pp.printCols(data)

pp.convertToObj(data,[1,2])

pp.convertToNum(data,[2])

pp.binning(data,'Age',[0,18,35,50],["young","medium","old"])

print(pp.removeNull(data))

X,y = pp.dataArray(data = data,yName = "Purchased")

Xt,Xte,yt,yte = pp.splitter(X,y,0.2)


from kapph.kapph import res