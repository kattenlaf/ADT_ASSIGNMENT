import numpy as np
import scipy
from scipy import stats
from scipy.stats import kurtosis, skew

Data_set = [[132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166], [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72], [173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217]]

a = [ ]
sum = 0


for _ in Data_set:
    for u in _:
        a.append(u)
        sum += u

print ( "Count:", len(a) )
print ( "Minimum:", min(a) )
print ( "Maximum:", max(a) )


print ( "Mean:", round(np.mean(a),2) )
m = stats.mode(a)
print ( "Mode:", m[0] )

print ( "Range:", max(a)- min(a) )
print ( "Variance", round(np.var(a), 2) )
print ( "Standard Deviation", round(np.std(a),2) )

print ( "Coefficient of Variation", round(scipy.stats.variation(a), 2) )
print ( "Skewness", round(skew(a), 2) )
print ( "Kurtosis", round(kurtosis(a), 2) )

def quartiles(data):
    sorted_data = sorted(data)
    mid =len(data)/2
    if (len(sorted_data)) % 2 == 0:
        #even
        Q_1 = np.median(sorted_data[:int(mid)])
        Q_3 = np.median(sorted_data[int(mid):])
    else:
    #odd
        Q_1 = np.median(sorted_data[:int(mid)])  #same as even
        Q_3 = np.median(sorted_data[int(mid):])

    print ( "Quartile 1:", Q_1 )
    print ( "Median:", np.median(a) )  #Quartile 2 is the median
    print ( "Quartile 3:", Q_3 )

quartiles(a)