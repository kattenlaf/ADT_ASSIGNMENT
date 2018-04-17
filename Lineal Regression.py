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

model1 = LinearRegression(fit_intercept=True)
model2 = LinearRegression(fit_intercept=True)
model3 = LinearRegression(fit_intercept=True)


model1.fit(x[:, np.newaxis], y1)
model2.fit(x[:, np.newaxis], y2)
model3.fit(x[:, np.newaxis], y3)

x1fit = np.linspace(0, 10, 1000)

y1fit = model1.predict(x1fit[:, np.newaxis])
y2fit = model2.predict(x1fit[:, np.newaxis])
y3fit = model3.predict(x1fit[:, np.newaxis])


print("Model One slope:    ", model1.coef_[0])
print("Model One intercept:", model1.intercept_)
print("Model Two slope:    ", model2.coef_[0])
print("Model Two intercept:", model2.intercept_)
print("Model Three slope:    ", model3.coef_[0])
print("Model Three intercept:", model3.intercept_)

plt.scatter(x, y1, label='Systolic Blood Pressure', color='k')
plt.scatter(x, y2, label='Age in Years', color='r')
plt.scatter(x, y3, label='Weight in Pounds', color='b')

plt.plot(x1fit, y1fit)
plt.plot(x1fit, y2fit)
plt.plot(x1fit, y3fit)
plt.legend()
plt.show()

