import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# x is the sorting of the values of list one

data_set = pd.read_csv('Data_Set.csv')

x1 = np.sort(data_set['X1'])
y1 = np.arange(1, len(x1)+1) / len(x1)

values, base = np.histogram(data_set['X1'], bins=40)
#evaluate the cumulative
cumulative = np.cumsum(values)
# plot the cumulative function
plt.plot(base[:-1], cumulative, c='blue')
#plot the survival function
plt.plot(base[:-1], len(data_set)-cumulative, c='green')
plt.margins(0.02)
plt.show()