import matplotlib.pyplot as plt
import pandas as pd

data_set = pd.read_csv('Data_Set.csv')

list1 = data_set['X1']

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

plt.hist(list1, bins, histtype='bar', rwidth=0.8)
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.title("HISTOGRAM")
plt.show()