# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:37:00 2026

@author: ASUS
"""

import pandas as pd
import matplotlib.pyplot as plt

Data = pd.read_csv('global_temp_index.csv')

year = Data.iloc[:, 0]
temp_index = Data.iloc[:, 1] 

#X axis has year, y axis has temperature


#For running need to select all the lines! and run! 
plt.plot(year, temp_index)
plt.xlabel('Year')
plt.ylabel('Temp_Index')
plt.xticks(year, rotation=45)
plt.title('Global Warming', family='Times New Roman')
plt.tight_layout()
#To save before the show! after the save it not save! 

plt.savefig('Global_Warming.png', dpi=300)
plt.show()


#########

