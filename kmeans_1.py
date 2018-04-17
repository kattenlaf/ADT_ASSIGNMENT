import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


df = pd.DataFrame({
    'x': [132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166],
    'y': [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72],
    #'x': [7, 3, 4, 3],
    #'y': [9, 3, 1, 8],
    #'y2':[173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217],
})

kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=3, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)

#learning labels

labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(5, 5))

colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'gold',  6: 'c', 7: 'lightcoral'}

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])

#take care here, change xlim and y lim if new data is used
plt.xlim(125, 170)
plt.ylim(40, 80)
plt.show()

