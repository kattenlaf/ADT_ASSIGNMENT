import numpy as np
#import statsmodels.api as sm

#What is this data supposed to represent?
y = [1,2,3,4,3,4,5,4,5,5,4,5,4,5,4,5,6,5,4,5,4,3,4]

x = [
     [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5],
     [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,7,7,7,7,7,6,5],
     [4,1,2,5,6,7,8,9,7,8,7,8,7,7,7,7,7,7,6,6,4,4,4]
     ]

def reg_m(y, x):
    ones = np.ones(len(x[0]))
    X = sm.add_constant(np.column_stack((x[0], ones)))
    for ele in x[1:]:
        X = sm.add_constant(np.column_stack((ele, X)))
    results = sm.OLS(y, X).fit()
    return results

print (reg_m(y, x).summary())



# I added a new panda feature

#data_set = pd.read_csv('Data_Set.csv')


#x = data_set['x']
#y1 = np.sort(data_set['X1'])
#y2 = np.sort(data_set['X2'])
#y3 = np.sort(data_set['X3'])

#x = data_set['x']
#x = np.array(x)

#def lin_reg(x,y):
#    fig, ax = plt.subplots()
#    fit = np.polyfit(x, y, deg=1)
#    ax.plot(x, fit[0] * x + fit[1], color='red')
#    ax.scatter(x, y)
#    plt.show()

#lin_reg(x,y1)
#lin_reg(x,y2)
#lin_reg(x,y3)
#
#import numpy as np