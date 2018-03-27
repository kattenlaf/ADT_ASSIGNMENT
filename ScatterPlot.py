import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Data_Set.csv')
list_1 = data_set['X1']
list_0 = []
for x in range(len(list_1)):
    list_0.append(x)
#SCATTER PLOT

plt.scatter(list_0, list_1, label='MyPlot', color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SCATTER PLOT')
plt.legend()
plt.show()