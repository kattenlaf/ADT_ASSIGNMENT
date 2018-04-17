import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns; sns.set()

from sklearn.linear_model import LinearRegression

# I added a new panda feature 'x'

data_set = pd.read_csv('Data_Set.csv')

x = data_set['x']

y1 = np.sort(data_set['X1'])
y2 = np.sort(data_set['X2'])
y3 = np.sort(data_set['X3'])

#x = data_set['x']
#x = np.array(x)

model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y1)

xfit = np.linspace(0, 10, 1000)
yfit = model.predict(xfit[:, np.newaxis])


print("Model slope:    ", model.coef_[0])
print("Model intercept:", model.intercept_)

plt.scatter(x, y1)
plt.plot(xfit, yfit)
plt.show()

