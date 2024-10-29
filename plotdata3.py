from safe_limitv import compute_vx

radius=0.1
error=0.01
sense_range=4.3
safe_bound=0.37
amax=20
latency_depth = 0.05
latency_vins = 0
latency=latency_depth + latency_vins
jerk = 120
R=3

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')

xx = np.linspace(0.002,0.1002,100)
yy = np.linspace(1,11,100)
X, Y = np.meshgrid(xx, yy)

# print(X)
# print(Y)

Y2 = Y - 4
Y2[np.where(Y2 < 0)] = 0
# Z = X * 0.2 + Y * 0.01
Z = np.reciprocal(X) * 0.001   + np.square(Y2) * 0.03
# Z = np.square(Y2) * 0.03
# Z = np.reciprocal(X) * 0.001  
# print(Z.shape)
v = np.zeros((100,100))
# print(xx[20], yy[50], Z.T[20][50], )
for i in range(100):
    for j in range(100):
        error = xx[i]
        sense_range = yy[j]
        latency = Z[j][i]
        v[i][j] = compute_vx(R,radius,error,sense_range,safe_bound,amax,latency,jerk)

print(xx[20], yy[50], Z[20][50], v[20][50])
min_v = v.min()
max_v = v.max()

color = []
for j in range(100):
    line = []
    for i in range(100):
        line.append(plt.get_cmap("seismic", 10000)(int(float(v[i][j]-min_v)/(max_v-min_v)*10000)))
    color.append(line)

plt.set_cmap(plt.get_cmap("seismic", 10000))
im = ax.plot_surface(X,Y,Z,facecolors=color) 
# im = ax.scatter(X, Y, Z, s=10,c=color,marker='.')

plt.savefig('graph_new.svg')
plt.show()