import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x is the sorting of the values of list one

data_set = pd.read_csv('Data_Set.csv')

'''
THIS WORKS
for i in data_set['X1']:
    print(i)
'''

x = np.sort(data_set['X1'])
y = np.arange(1, len(x)+1) / len(x)

_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.title("CUMULATIVE PLOT")
plt.margins(0.02)
plt.show()