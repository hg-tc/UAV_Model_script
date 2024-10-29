# from safe_limitv import compute_vx

# radius=0.15
# error=0.01
# sense_range=6
# safe_bound=0.35
# amax=20
# latency_depth = 0.05
# latency_vins = 0
# latency=latency_depth + latency_vins
# jerk = 120
# R=2.5

import json
import re
import numpy as np

with open('DepthNNModel_accuracy_200_only08.json') as f:
    parsed_json1 = json.load(f)

with open('VINS_accuracy_0226_1833_MH04.json') as f:
    parsed_json3 = json.load(f)
with open('scheduled_latency.json') as f:
    parsed_json2 = json.load(f)
with open('velocity.json') as f:
    parsed_json4 = json.load(f)
data = []
datax = []
datax_p = []
datay = []
datay_p = []
dataz = []
dataz_p = []
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
    e.append(error*100)
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
    datax.append(float(err)*100)
    datax_p.append(eparam)
    datay.append(float(srange))
    datay_p.append(sparam)
    dataz.append(float(latency))

v_max = 0
index_best = 0
index_best_param = 0
min_latency = 100
index_low_latency = 0

index_best_s = 0
index_best_param_s = 0
index_low_latency_s = 0

index_best_e = 0
index_best_param_e = 0
index_low_latency_e = 0
for i,vx in enumerate(parsed_json4.values()):
    # vx = compute_vx(R,radius,datax[i],datay[i],safe_bound,amax,dataz[i],jerk)
    dataz_p.append(vx)
    if(vx > v_max):
        index_best = i
        v_max = vx
        index_best_s = datay_p[i]
        index_best_e = datax_p[i]
    if(datax_p[i]==25 and datay_p[i] ==10):
        index_best_param = i
        index_best_param_s = 10
        index_best_param_e = 25
        
    if(dataz[i]<min_latency):
        index_low_latency = i
        min_latency = dataz[i]
        index_low_latency_s = datay_p[i]
        index_low_latency_e = datax_p[i]

print(index_best)
print(index_low_latency)

#######################################################

for i,index in enumerate(dict1.values()):
    if(index == index_best_s):
        index_best_s2=i
    if(index == index_best_param_s):
        index_best_param_s2=i
    if(index == index_low_latency_s):
        index_low_latency_s2=i


for i,index in enumerate(dict2.values()):
    if(index == index_best_e):
        index_best_e2=i
    if(index == index_best_param_e):
        index_best_param_e2=i
    if(index == index_low_latency_e):
        index_low_latency_e2=i
    
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

########################################################

pdata = []
pdatax = []
pdatay = []
pdataz = []

pdatax_p = []
pdatay_p = []
pdataz_p = []
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

        pdatax_p.append(datax_p[i])
        pdatay_p.append(datay_p[i])
        pdataz_p.append(dataz_p[i])

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
im = ax.scatter(x, y, z, s=30,marker='.')
im2 = ax.scatter(px, py, pz, c='k', s=30,marker='*')


im3 = ax.scatter(datax[index_best], datay[index_best], dataz[index_best],c='g', s=200,marker='.')
# ax.text(datax[index_best], datay[index_best], dataz[index_best],s='max_velocity',color='g')
im4 = ax.scatter(datax[index_best_param], datay[index_best_param], dataz[index_best_param],c='b', s=200,marker='.')
# ax.text(datax[index_best_param], datay[index_best_param], dataz[index_best_param],s='best_param',color='b')
im5 = ax.scatter(datax[index_low_latency], datay[index_low_latency], dataz[index_low_latency],c='r', s=200,marker='.')
# ax.text(datax[index_low_latency], datay[index_low_latency], dataz[index_low_latency],s='min_latency',color='r')

# ax.set_xlabel('localization_error')
# ax.set_ylabel('sensoing_range / m')
# ax.set_zlabel('latency / s')

a = [8,16,32,48,64]
b = [3,6]
SX,SY = np.meshgrid(a, b)

SZ = [[3.8565655552434737,4.346217218606056,4.489530831646479,4.763257859470633,5.202098681683131],[3.1897993592171834,3.713684120776331,5.0736919062570776,5.193639149736084,4.98119264538968]]
SZ = np.array(SZ)
# SZ = 


fig = plt.figure(1)
ax = fig.add_subplot(111,projection='3d')
im = ax.plot_surface(SX,SY,SZ,color='aliceblue') 
im = ax.scatter(sx, sy, sen, s=200,marker='.')
# ax2=ax.twinx()

im2 = ax.scatter(sx[index_best_s2], sy[index_best_s2], sen[index_best_s2],c='g', s=200,marker='.')
# ax.text(sx[index_best_s2], sy[index_best_s2], sen[index_best_s2],s='max_velocity',color='g')
im3 = ax.scatter(sx[index_best_param_s2], sy[index_best_param_s2], sen[index_best_param_s2],c='b', s=200,marker='.')
# ax.text(sx[index_best_param_s2], sy[index_best_param_s2], sen[index_best_param_s2],s='best_param',color='b')
im4 = ax.scatter(sx[index_low_latency_s2], sy[index_low_latency_s2], sen[index_low_latency_s2],c='r', s=200,marker='.')
# ax.text(sx[index_low_latency_s2], sy[index_low_latency_s2], sen[index_low_latency_s2],s='min_latency',color='r')

# ax.set_xlabel('param1')
# ax.set_ylabel('param2')
# ax.set_zlabel('sensoing_range / m')

a = [1,2,3,4,5]
b = [3,6,10,15,20]
EX,EY = np.meshgrid(a, b)
print(EX)
print(ey)
EZ= np.zeros((5,5))
for i in range(5):
    for j in range(5):
        EZ[i][j] = e[j+i*5]

my_x_ticks = [3,6,10,15,20]


fig = plt.figure(2)
ax = fig.add_subplot(111,projection='3d')
im = ax.scatter(ex, ey, e, s=200,marker='.')
im = ax.plot_surface(EX,EY,EZ,color='aliceblue') 
plt.yticks(my_x_ticks)

ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
im2 = ax.scatter(ex[index_best_e2], ey[index_best_e2], e[index_best_e2],c='g', s=200,marker='o')
# ax.text(ex[index_best_e2], ey[index_best_e2], e[index_best_e2],s='max_velocity',color='g')
im3 = ax.scatter(ex[index_best_param_e2], ey[index_best_param_e2], e[index_best_param_e2],c='b', s=200,marker='.')
# ax.text(ex[index_best_param_e2], ey[index_best_param_e2], e[index_best_param_e2],s='best_param',color='b')
im4 = ax.scatter(ex[index_low_latency_e2], ey[index_low_latency_e2], e[index_low_latency_e2],c='r', s=200,marker='.')
# ax.text(ex[index_low_latency_e2], ey[index_low_latency_e2], e[index_low_latency_e2],s='min_latency',color='r')
# ax.set_xlabel('param1')
# ax.set_ylabel('param2')
# ax.set_zlabel('localization_error ')


fig = plt.figure(3)
ax = fig.add_subplot(111,projection='3d')
im = ax.scatter(datax_p, datay_p, dataz_p, s=30,marker='.')
im5 = ax.scatter(pdatax_p, pdatay_p, pdataz_p,c='k', s=30,marker='*')
im2 = ax.scatter(datax_p[index_best], datay_p[index_best], dataz_p[index_best],c='g', s=200,marker='.')
# ax.text(datax_p[index_best], datay_p[index_best], dataz_p[index_best],s='max_velocity',color='g')
im3 = ax.scatter(datax_p[index_best_param], datay_p[index_best_param], dataz_p[index_best_param],c='b', s=200,marker='.')
# ax.text(datax_p[index_best_param], datay_p[index_best_param], dataz_p[index_best_param],s='best_param',color='b')
im4 = ax.scatter(datax_p[index_low_latency], datay_p[index_low_latency], dataz_p[index_low_latency],c='r', s=200,marker='.')
# ax.text(datax_p[index_low_latency], datay_p[index_low_latency], dataz_p[index_low_latency],s='min_latency',color='r')
# ax.set_xlabel('localizatin_error param')
# ax.set_ylabel('sensoing_range param')
# ax.set_zlabel('velocity m/s')
plt.show()