# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 14:01:51 2026

@author: ASUS
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

Data = pd.read_csv('montly_electricity_consumption.csv')

month = Data.iloc[:,0]
customerA = Data.iloc[:,1]
customerB = Data.iloc[:,2]

plt.plot(month, customerA)
plt.plot(month, customerB)
plt.xticks(month, rotation = 45)
plt.tight_layout()
plt.show()


plt.plot(month, customerA, color='red', label='Customer A', marker= 'o')
plt.plot(month, customerB, color='blue', label='Customer B', marker='*')
plt.xlabel()
plt.ylabel()
plt.title('Electiricty Consumption Chart')
plt.xticks(month, rotation =45) 
plt.legend()
plt.tight_layout()
plt.show()


###########


plt.subplot(2, 1, 1) #2 roles and 1 column
plt.plot (month, cusomterA, color='red', label='Customer A', marker='o')
plt.xlabel('Month')
plt.ylabel('kWh')
plt.legend()


plt.subplot(2,1,2)
plt.plot(month, customerB, color='blue', label='CustomerB', marker='*')
plt.xlabel('Month')
plt.ylabel('kWh')
plt.suptitle('Electricity Consumption Chart')
plt.tight_layout()
plt.legend()
plt.show()

#######


plt.scatter(month, customerA, color='red', label='customer A')
plt.scatter(month, customerB, color='blue', label='customer B')
plt.xlabel('Month')
plt.ylabel('kWh')
plt.title('Scatterplot of Electricity Consumption')
#plt.legend(loc=(0.1, 0.1))
plt.legend(loc='best')
plt.grid()
plt.show()

#######

#histogram plot

plt.hist(customerA, bins=20, color='blue', rwidth=0.8)
plt.xlabel('Month')
plt.ylabel('kWh')
plt.title('Histogram')
plt.show()


##############
#bargraph

plt.bar(month, customerA, color='blue')
plt.xlabel('Month')
plt.ylabel('kWh')
plt.title('Electricity Consumption Bar Graph')
plt.show()

########

bar_width= 0.4
Month_b = np.arange(1, 13)

plt.bar(Month_b, customerA, bar_width, color='blue', label='Customer A')
plt.bar(Month_b+bar_width, customerB, bar_width, color='red', label='Customer B')
plt.xlabel('Month')
plt.ylabel('kWh')
plt.title('bar Chart')
plt.legend(loc ='best')
plt.xticks(Month_b + (bar_width)/2, (month))
plt.show()

###########






