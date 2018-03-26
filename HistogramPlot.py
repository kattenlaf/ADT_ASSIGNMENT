import matplotlib.pyplot as plt

list1 = [132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166]

bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]

plt.hist(list1, bins, histtype='bar', rwidth=0.8)
_ = plt.xlabel('x')
_ = plt.ylabel('y')
_ = plt.title("HISTOGRAM")
plt.show()