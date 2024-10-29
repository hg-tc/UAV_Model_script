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

import json
import re
import numpy as np

with open('/home/zsq/proj/DepthNNModel_accuracy_200_only08.json') as f:
    parsed_json1 = json.load(f)

with open('/home/zsq/proj/VINS_accuracy_0226_1833_MH04.json') as f:
    parsed_json3 = json.load(f)
with open('/home/zsq/proj/scheduled_latency.json') as f:
    parsed_json2 = json.load(f)

data = []
datax = []
datax_p = []
datay = []
datay_p = []
dataz = []
#######################################################
dict1 = {
    "[8, 3]": 3,
    "[8, 6]": 1,
    "[16, 3]": 4,
    "[16, 6]": 2,
    "[32, 3]": 5,
    "[32, 6]": 8,
    "[48, 3]": 6,
    "[48, 6]": 9,
    "[64, 3]": 10,
    "[64, 6]": 7
}

sen = []
sx = [8,8,16,16,32,32,48,48,64,64]
sy = [3,6,3,6,3,6,3,6,3,6]
param_sen = []
for parameter,sensing_range in zip(parsed_json1,parsed_json1.values()):
    sen.append(sensing_range)
    param_sen.append(parameter)

indice = np.argsort(sen)[::-1]
print(indice)



########################################################
dict2 = {}
e = []
ex = []
ey = [3,3,3,3,3,6,6,6,6,6,10,10,10,10,10,15,15,15,15,15,20,20,20,20,20]
for parameter,error in zip(parsed_json3,parsed_json3.values()):
    e.append(error)
for i in range(25):
    ex.append(i%5+1)

indice = np.argsort(e)[::-1]
print(indice)

for i,parameter in enumerate(parsed_json3):
    dict2[parameter] = np.where(indice == i)[0][0] + 1

########################################################
    
for parameter,latency in zip(parsed_json2,parsed_json2.values()):
    res1 = re.findall(r'\[\d+, \d+',parameter)
    srange = parsed_json1[res1[0]+']']
    sparam = dict1[res1[0]+']']

    res2 = re.findall(r'\'\w+\', \d+, \d+, 0.015, \d+, \d+, \d+\]',parameter)
    err = parsed_json3['['+res2[0]]
    eparam = dict2['['+res2[0]]
    
    data.append([float(err),float(srange),float(latency)])
    datax.append(float(err))
    datax_p.append(eparam)
    datay.append(float(srange))
    datay_p.append(sparam)
    dataz.append(float(latency))

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

########################################################

pdata = []
pdatax = []
pdatay = []
pdataz = []
for i in range(250):
    flag = 0
    for j in range(250):
        if(datax[i] > datax[j] and datay[i] < datay[j] and dataz[i] > dataz[j]):
            flag = 1
    if(flag):
        pass
    else:
        pdata.append(data[i])
        pdatax.append(datax[i])
        pdatay.append(datay[i])
        pdataz.append(dataz[i])

########################################################

from scipy.io import savemat
import numpy as np

file_name = 'data.mat'
data = np.array(data)
x = np.array(datax)
y = np.array(datay)
z = np.array(dataz)


pdata = np.array(pdata)
px = np.array(pdatax)
py = np.array(pdatay)
pz = np.array(pdataz)
savemat(file_name, {'x':x,'y':y,'z':z,'px':px,'py':py,'pz':pz,'data':data,'pdata':pdata})

########################################################

# 
fig = plt.figure(0)
ax = fig.add_subplot(111,projection='3d')
im = ax.scatter(datax, datay, dataz, s=100,marker='.')
im2 = ax.scatter(pdatax, pdatay, pdataz, c='r', s=100,marker='.')
ax.set_xlabel('localization_error')
ax.set_ylabel('sensoing_range / m')
ax.set_zlabel('latency / s')



fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
im = ax.scatter(sx, sy, sen, s=100,marker='.')
ax.set_xlabel('param1')
ax.set_ylabel('param2')
ax.set_zlabel('sensoing_range / m')


fig = plt.figure(2)
ax = fig.add_subplot(111,projection='3d')
im = ax.scatter(ex, ey, e, s=100,marker='.')
ax.set_xlabel('param1')
ax.set_ylabel('param2')
ax.set_zlabel('localization_error ')


fig = plt.figure(3)
ax = fig.add_subplot(111,projection='3d')
im = ax.scatter(datax_p, datay_p, dataz, s=100,marker='.')
ax.set_xlabel('localizatin param')
ax.set_ylabel('sensoing_range param')
ax.set_zlabel('latency / s')
plt.show()