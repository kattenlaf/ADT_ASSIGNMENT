import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy


df = pd.DataFrame({
    'x': [132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166],
    'y': [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72],
    'z':[173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217],
})

#x = (df['x'],df['x'])
#y = (df['y1'], df['y2'])

#Data_set = [[132, 143, 153, 162, 154, 168, 137, 149, 159, 128, 166], [52, 59, 67, 73, 64, 74, 54, 61, 65, 46, 72], [173, 184, 194, 211, 196, 220, 188, 188, 207, 167, 217]]

np.random.seed(200)
k = 3
#centroids[i] = [x, y]
centroids = {
    i + 1: [np.random.randint(125, 170), np.random.randint(40, 80)]
    for i in range(k)
}


#fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], df['z'], color='k')
colmap = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'gold',  6: 'c', 7: 'lightcoral'}
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(125, 170)
plt.ylim(40, 80)
plt.show()


## Assignment Stage

def assignment(df, centroids):
    for i in centroids.keys():
        # sqrt((x1 - x2)^2 - (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
                #+ (df['y2'] - centroids[i][1]) ** 2   #todoooo000000000000000000000000000000000000000
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])   ##############checkear la x que no conflija con mi dataframe
    return df

df = assignment(df, centroids)
print(df.head())

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(125, 170)
plt.ylim(40, 80)
plt.show()


old_centroids = copy.deepcopy(centroids)


def update(k):
    for i in centroids.keys():
        centroids[i][0] = np.mean(df[df['closest'] == i]['x'])
        centroids[i][1] = np.mean(df[df['closest'] == i]['y'])
    return k


centroids = update(centroids)
fig = plt.figure(figsize=(5, 5))
ax = plt.axes()
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(125, 170)
plt.ylim(40, 80)

for i in old_centroids.keys():
    old_x = old_centroids[i][0]
    old_y = old_centroids[i][1]
    dx = (centroids[i][0] - old_centroids[i][0]) * 0.75
    dy = (centroids[i][1] - old_centroids[i][1]) * 0.75
    ax.arrow(old_x, old_y, dx, dy, head_width=2, head_length=3, fc=colmap[i], ec=colmap[i])
plt.show()

## Repeat Assigment Stage

df = assignment(df, centroids)

# Plot results
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(125, 170)
plt.ylim(40, 80)
plt.show()


# Continue until all assigned categories don't change any more
while True:
    closest_centroids = df['closest'].copy(deep=True)
    centroids = update(centroids)
    df = assignment(df, centroids)
    if closest_centroids.equals(df['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i])
plt.xlim(125, 170)
plt.ylim(40, 80)
plt.show()