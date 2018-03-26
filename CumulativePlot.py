import matplotlib.pyplot as plt
import numpy as np

# x is the sorting of the values of list one

x = np.sort([132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166])
y = np.arange(1, len(x)+1) / len(x)

_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.title("CUMULATIVE PLOT")
plt.margins(0.02)
plt.show()
