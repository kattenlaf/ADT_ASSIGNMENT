from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew
import seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv('Data_Set.csv')

def scatterPlot(data_set):
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

    # SCATTER PLOT
    correlation1 = 0
    correlation2 = 0
    correlation3 = 0

    plt.scatter(list_X1, list_1, label='Systolic Blood Pressure', color='k')
    correlation1 = np.corrcoef(list_X1, list_1)[0, 1]
    print("The correlation to set one", correlation1)
    print("The correlation squared is", correlation1*correlation1)

    plt.scatter(list_X2, list_2, label='Age in Years', color='r')
    correlation2 = np.corrcoef(list_X2, list_2)[0, 1]
    print("The correlation to set two", correlation2)
    print("The correlation squared is", correlation2*correlation2)

    plt.scatter(list_X3, list_3, label='Weight in Pounds', color='b')
    correlation3 = np.corrcoef(list_X3, list_3)[0, 1]
    print("The correlation to set three", correlation3)
    print("The correlation squared is", correlation3*correlation3)

    plt.xlabel('X VALUES')
    plt.ylabel('Y VALUES')
    plt.title('SCATTER PLOT')
    plt.legend()
    plt.show()

def findCumulativeValues(myarr):
    newArr = []
    sum = 0;
    for _ in myarr:
        sum += _
        newArr.append(sum)
    return newArr

def cumulativePlot(data_set):

    test = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    x1 = np.sort(data_set['X1'])
    newx1 = findCumulativeValues(x1)

    y1 = np.arange(1, len(x1) + 1) / len(x1)

    x2 = np.sort(data_set['X2'])
    newx2 = findCumulativeValues(x2)

    y2 = np.arange(1, len(x2) + 1) / len(x2)

    x3 = np.sort(data_set['X3'])
    newx3 = findCumulativeValues(x3)

    y3 = np.arange(1, len(x3) + 1) / len(x3)

    _ = plt.plot(test, newx1, marker='.', linestyle='none', label='Systolic Blood Pressure')
    _ = plt.plot(test, newx2, marker='.', linestyle='none', label='Age in Years')
    _ = plt.plot(test, newx3, marker='.', linestyle='none', label='Weight in Pounds')
    _ = plt.xlabel('X VALUES')
    _ = plt.ylabel('Y VALUES')
    _ = plt.title("CUMULATIVE PLOT")
    plt.margins(0.02)
    plt.legend()
    plt.show()

def histogram(data_set):

    list1 = data_set['X1']
    list2 = data_set['X2']
    list3 = data_set['X3']
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230]

    plt.hist(list1, bins, histtype='bar', label='Systolic Blood Pressure', rwidth=0.8)
    plt.hist(list2, bins, histtype='bar', label='Age in Years', rwidth=0.8)
    plt.hist(list3, bins, histtype='bar', label = 'Weight in Pounds', rwidth=0.8)

    _ = plt.xlabel('X')
    _ = plt.ylabel('Y')
    _ = plt.title("HISTOGRAM")
    _ = plt.legend()
    plt.show()

def linearRegressionX1(data_set):
    
    x = data_set['x']

    y1 = np.sort(data_set['X1'])

    model = LinearRegression(fit_intercept=True)

    model.fit(x[:, np.newaxis], y1)

    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    print("Model One regression coefficient:", model.coef_[0])
    print("Model One intercept:", model.intercept_)

    plt.scatter(x, y1, label='Systotic Blood Pressure', color='b')
    plt.plot(xfit, yfit, color='k')
    plt.legend()
    plt.show()

def linearRegressionX2(data_set):

    x = data_set['x']

    y2 = np.sort(data_set['X2'])

    model = LinearRegression(fit_intercept=True)

    model.fit(x[:, np.newaxis], y2)

    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    print("Model Two regression coefficient:", model.coef_[0])
    print("Model Two intercept:", model.intercept_)

    plt.scatter(x, y2, label='Age in Years', color='r')
    plt.plot(xfit, yfit, color='k')
    plt.legend()
    plt.show()

def linearRegressionX3(data_set):

    x = data_set['x']

    y3 = np.sort(data_set['X3'])

    model = LinearRegression(fit_intercept=True)

    model.fit(x[:, np.newaxis], y3)

    xfit = np.linspace(0, 10, 1000)
    yfit = model.predict(xfit[:, np.newaxis])

    print("Model Three regression coefficient:", model.coef_[0])
    print("Model Three intercept:", model.intercept_)

    plt.scatter(x, y3, label='Weight in Pounds', color='brown')
    plt.plot(xfit, yfit, color='k')
    plt.legend()
    plt.show()

# MULTIPLE REGRESSION ANALYSIS y = a + b1X1 + b2X2 + b3X3
'''
def printMultipleRegression():
    yset = data_set['x']
    yavg = 0
    for _ in yset:
        yavg += _
    yavg = yavg/len(yset)
'''

def kMeans_Clusters():
    
def doCalculations():

    Data_set = []

    Data_set.append(data_set['X1'])
    Data_set.append(data_set['X2'])
    Data_set.append(data_set['X3'])

    a = []
    sum = 0

    for _ in Data_set:
        for u in _:
            a.append(u)
            sum += u

    print("Count:", len(a))
    print("Minimum:", min(a))
    print("Maximum:", max(a))

    print("Mean:", round(np.mean(a), 2))
    m = stats.mode(a)
    print("Mode:", m[0])

    print("Range:", max(a) - min(a))
    print("Variance", round(np.var(a), 2))
    print("Standard Deviation", round(np.std(a), 2))

    print("Coefficient of Variation", round(scipy.stats.variation(a), 2))
    print("Skewness", round(skew(a), 2))
    print("Kurtosis", round(kurtosis(a), 2))

    def quartiles(data):
        sorted_data = sorted(data)
        mid = len(data) / 2
        if (len(sorted_data)) % 2 == 0:
            # even
            Q_1 = np.median(sorted_data[:int(mid)])
            Q_3 = np.median(sorted_data[int(mid):])
        else:
            # odd
            Q_1 = np.median(sorted_data[:int(mid)])  # same as even
            Q_3 = np.median(sorted_data[int(mid):])

        print("Quartile 1:", Q_1)
        print("Median:", np.median(a))  # Quartile 2 is the median
        print("Quartile 3:", Q_3)

    quartiles(a)
################ GUI STUFF ###################

root = Tk()

ourLabel = Label(root, text="OUR DATA SCIENCE GUI")
ourLabel.pack(side=TOP)

topFrame = Frame(root)
topFrame.pack()
bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

button1 = Button(topFrame, text="Print Calculations", command=doCalculations, fg="red")
button2 = Button(topFrame, text="Show Scatter Plot", command=lambda: scatterPlot(data_set), fg="blue")
button3 = Button(topFrame, text="Show Cumulative Plot", command=lambda: cumulativePlot(data_set), fg="green")
button4 = Button(topFrame, text="Show Histogram Plot", command=lambda: histogram(data_set), fg="black")
button5 = Button(bottomFrame, text="Show X1 Linear Regression", command=lambda: linearRegressionX1(data_set), fg="black")
button6 = Button(bottomFrame, text="Show X2 Linear Regression", command=lambda: linearRegressionX2(data_set), fg="black")
button7 = Button(bottomFrame, text="Show X3 Linear Regression", command=lambda: linearRegressionX3(data_set), fg="black")

button1.pack(side=LEFT)
button2.pack(side=LEFT)
button3.pack(side=LEFT)
button4.pack(side=LEFT)
button5.pack(side=LEFT)
button6.pack(side=LEFT)
button7.pack(side=LEFT)

root.mainloop()

#############################################################################