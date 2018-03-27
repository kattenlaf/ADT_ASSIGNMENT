import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Data_Set.csv')
list_X1 = []
list_X2 = []
list_X3 = []

list_1 = data_set['X1']
list_2 = data_set['X2']
list_3 = data_set['X3']

for x in range(len(list_1)):
    list_X1.append(x)

for x in range(len(list_2)):
    list_X2.append(x)

for x in range(len(list_3)):
    list_X3.append(x)
#SCATTER PLOT

plt.scatter(list_X1, list_1, label='X1 PLOT', color='k')
plt.scatter(list_X2, list_2, label='X2 PLOT', color='r')
plt.scatter(list_X3, list_3, label='X3 PLOT', color='b')

plt.xlabel('X VALUES')
plt.ylabel('Y VALUES')
plt.title('SCATTER PLOT')
plt.legend()
plt.show()