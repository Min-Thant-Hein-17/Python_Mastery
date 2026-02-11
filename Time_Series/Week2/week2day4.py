# -*- coding: utf-8 -*-
"""
Created on Wed Feb 11 13:15:01 2026

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,5,11)
y = x**2

plt.plot(x,y)
plt.Xlabel('X label')
plt.ylabel('Y label')
plt.title('Title')
plt.show()


plt.scatter(x,y)
plt.xlabel('X label')
plt.ylabel('Y label')
plt.title('Title')
plt.show()

###################


plt.subplot(1,2,1)  #row, column and position
plt.plot(x,y, 'r')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title 1')

plt.subplot(1,2,2) #2nd figure in second positon!
plt.plot(y, x, 'b')
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Title 2')

plt.suptitle('Title')

plt.savefig('my_figure.png', dpi=300)


#########
plt.figure()
plt.plot(x, x**2, label='x squared')
plt.plot(x, x**3, label='x cubed')
plt.title('Title')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

#######


