# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

@author: MinThantHein
"""
import pandas as pd 

"""
Series

"""

Age= pd.Series([10,20,30,40], index=['age1','age2','age3','age4'])

Age.age3

Age.age1

#Filtering the values of the Series

Filtered_Age= Age[Age>10]

Filtered_Age2= Age[(Age>10)& (Age<40)]

Filteded_Age3= Age[Age.between(20,30)]

#Calling values of the Series
Age.values

#Calling indexes of the Series
Age.index

"""
Dataframe

"""
import numpy as np

##Creating DataFrame

DF= np.array([20, 10, 8], [25,8,10], [27,5,3], [30,9,7])

Data_set =pd.DataFrame(DF)

Data_set =pd.DataFrame(DF, index= ['S1', 'S2', 'S3', 'S4'])

Data_set =pd.DataFrame(DF, index= ['S1', 'S2', 'S3', 'S4'], columns=['Age', 'Grade1', 'Grade2'])

Data_set['Grade3'] = [9,6,7,10]

#Indexing DataFrame
#.loc and .iloc

#.iloc is label-based indexing and include the end
#.iloc is integer-based indexing and exclude at the end 

Data_set.loc['S1']

Data_set.loc['S1': 'S2']

Data_set.iloc[1, 2]

Data_set.iloc[:, 0]

Data_set.iloc[:, 3]

Data_set.iloc[:, 1]

Data_set.iloc[:, 1:3]

Data_set.iloc[:, :3]

Data_set.iloc[1:2,:]

Data_set.iloc[:3, :]

Data_set.iloc[3, :]

Data_set.iloc[:, :-1]

Filtered_Data_set = Data_set.iloc[:, 1:3]

#Drop and replace

Data_set.drop()

Data_set.change1 = 

Data_set.change2 = 

Data_set.head()

Data_set.head()

Data_set.tail()

#Sorting values of the DataFrame

Data_set.sort_values()

Data_set.sort_values()

Data_set.sort_index()

Data_set.sort_index()

#rearange data based on its index labels rather than values

#import dataset 

DF = pd.read_csv("household_electricity_consumption.csv")

DF.head() 

DF.tail()

DF.to_csv('output1.csv')

DF1 = pd.read_csv('output1.csv')

DF.to_csv('output2.csv', index=False)

DF2 = pd.read_csv('output2.csv')

two_columns = DF.iloc[:, 1:3]

all_columns_but_exclude_last_column = DF.iloc[:, :5]

last_column = DF.iloc[:, -1]

first_six_hours = DF.iloc[:7, :]







