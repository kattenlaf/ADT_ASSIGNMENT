import matplotlib.pyplot as plt 

list_1 = [132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166]
list_2 = [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72]
list_3 = [173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217]

#SCATTER PLOT

plt.scatter(list_1, list_2, label='MyPlot', color='k')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SCATTER PLOT')
plt.legend()
plt.show()