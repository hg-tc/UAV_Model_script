import math

import matplotlib.pyplot as plt
import numpy as np

def compute_vy(vx, radius=1,error=0.0,sense_range=4,safe_bound=0.36,amax=0.5,latency=0):

    r,e,S,s,am = radius,error,sense_range,safe_bound,amax

    A = r+s
    A2 = A + e*vx*latency
    B = e*vx
    T = (S - s) / vx - latency

    if(T<=0):
        return 0

    if((am + 2*B/T - 2*A2/(T**2))>0):
        t_left = 2*B / ( am + 2*B/T - 2*A2/(T**2))
        t_left = min(T, t_left)
        t_left = max(0.0000001, t_left)
    else:
        t_left = T
    t = T - t_left

    vy = 2*B*math.log(T/t_left,math.e) - 2*B*t/T + 2*A2*t/(T**2) + am * t_left

    vy = min(vy, 40)

    return vy, t_left, T
    
def compute_y(vx, radius=1,error=0.0,sense_range=4,safe_bound=0.36,amax=0.5,latency=0):

    r,e,S,s,am = radius,error,sense_range,safe_bound,amax

    A = r+s
    A2 = A + e*vx*latency
    B = e*vx
    T = (S - s) / vx - latency

    if(T<=0):
        return 0

    if((am + 2*B/T - 2*A2/(T**2))>0):
        t_left = 2*B / ( am + 2*B/T - 2*A2/(T**2))
        t_left = min(T, t_left)
        t_left = max(0.0000001, t_left)
    else:
        t_left = T
    t = T - t_left

    
    vy_t = 2*B*math.log(T/t_left,math.e) - 2*B*t/T + 2*A2*t/(T**2) 
    y_t = A2+B*T - 2*B*t_left*math.log(T/t_left,math.e) - B*t_left**2/T - A2*(T**2-t**2)/T**2
    y = y_t + vy_t*t_left + 0.5 * am *(t_left**2)

    return y

def compute_y_stop(vx, radius=1,error=0.0,sense_range=4,safe_bound=0.36,amax=0.5,latency=0,jerk=120):

    vy, t_left, T = compute_vy(vx,radius,error,sense_range,safe_bound,amax,latency)

    # print("vy",vy)
    vy2 = vy + 0.5 * amax**2 / jerk
    print("vy2",vy2)
    y = compute_y(vx,radius,error,sense_range,safe_bound,amax,latency)
    # print("y", y)
    # print("y2",y+vy**2 / (2 * amax))
    y = y + 2 * amax * vy / jerk + 2 * amax**3 / (3 * jerk **2) + vy**2 / (2 * amax)
    
    return y


def compute_vx(R, radius=1,error=0.0,sense_range=4,safe_bound=0.36,amax=0.5,latency=0,jerk=120):

    if(sense_range == 0):
        print("error: senser_range = 0")    
    result = 0
    i = 0

    vx_list = []
    vy_list = []
    y_list = []
    lim_vy_list = []


    while(1):
        vx = (i+1) * 0.01
        vy, t_left, T = compute_vy(vx,radius,error,sense_range,safe_bound,amax,latency)
        y = compute_y(vx,radius,error,sense_range,safe_bound,amax,latency)
        
        left_distance = R - radius - safe_bound - y
        # print("y1", left_distance)
        if(1):
            left_distance = left_distance - 2 * amax * vy / jerk - 2 * amax**3 / (3 * jerk **2)
        # print("y2", left_distance)
        if(left_distance > 0):
            lim_vy = math.sqrt(2*amax*left_distance)
        else:
            lim_vy = 0
        
        # print("y3", lim_vy)
        vx_list.append(vx)
        vy_list.append(vy)
        y_list.append(y)
        lim_vy_list.append(lim_vy)

        if(vy >= lim_vy or y <= (radius+safe_bound)):
            # print("info", vx, vy, y ,lim_vy)
            result = vx
            break
        i +=1


    # plt.figure(2)
    # plt.plot(vx_list,vy_list,'b')
    # plt.plot(vx_list,lim_vy_list,'r')
    # plt.draw()
    # plt.pause(0.00001)

    # plt.figure(3)
    # plt.plot(vx_list,y_list,'g')
    # plt.axhline(radius+safe_bound,color='r')
    # plt.draw()
    # plt.pause(0.00001)
    
    return result

if __name__ == '__main__':

    '''
    change parameter here !!!!

    radius: Obstacle radius
    error: Positioning accuracy percentage
    sense_range: The sensor's sensing range
    safe_bound: Safe distance, the amount of obstacle expansion
    amax: Maximum vertical acceleration
    R: Average spacing between obstacles
    '''
    # ===============================compare two==================================
    # radius=0.1
    # error=0.015
    # sense_range=5.1
    # safe_bound=0.37
    # amax=5
    # latency_depth = 0.1
    # latency_vins = 0.13
    # latency=latency_depth + latency_vins
    # R=1.2
    radius=0.1
    error=0.01
    sense_range=6
    safe_bound=0.37
    amax=20
    latency_depth = 0.1
    latency_vins = 0
    latency=latency_depth + latency_vins
    jerk = 120
    R=3

    v_max = compute_vx(R,radius,error,sense_range,safe_bound,amax,latency,jerk)
    ystop = compute_y_stop(10,radius,error,sense_range,safe_bound,amax,latency,jerk)

    print(ystop)

    # for i in range(11):
    #     sense_range = 0.5*i + 2
    #     v_max = compute_vx(R,radius,error,sense_range,safe_bound,amax,latency,jerk)

    # # ystop = compute_y_stop(8,radius,error,sense_range,safe_bound,amax,latency,jerk)
    # # vel_x = compute_y(1,radius,error,sense_range,safe_bound,amax,latency)
    #     print(sense_range,' ',v_max)



    # radius=0.1
    # error=0.018
    # sense_range=4.3
    # safe_bound=0.37
    # amax=5
    # latency_depth = 0.0
    # latency_vins = 0.11
    # latency=latency_depth + latency_vins
    # R=1.2

    # v_max = compute_vx(R,radius,error,sense_range,safe_bound,amax,latency)
    # print(v_max)
    # --------------------------------------------------------------------------------

    e_list = []
    v_list = []
    v2_list = []
    v3_list = []
    v4_list = []
    v5_list = []
    v6_list = []
    y_list = []
    y2_list = []
    lm_list = []
    lef_list = []
    s_list = []




    for i in range(2000):
        # error = (i+1) * 0.0001
        v_max = (i+1) * 0.01
        error2 = (i+1) * 0.0001
        sense_range2 = (i) * 0.005+1
        v_max_comp = compute_vx(R,radius,error2,sense_range,safe_bound,amax,latency,jerk)
        v_max_comp2 = compute_vx(3.5,radius,error2,sense_range,safe_bound,amax,latency,jerk)
        v_max_comp3 = compute_vx(R,radius,error,sense_range2,safe_bound,amax,latency,jerk)
        v_max_comp4 = compute_vx(R,radius,error+0.02,sense_range2,safe_bound,amax,latency,jerk)

        vy,t_left,T = compute_vy(v_max,radius,error,sense_range,safe_bound,amax,latency)
        y = compute_y(v_max,radius,error,sense_range,safe_bound,amax,latency)
        
        left_distance = R - radius - safe_bound - y
        if(1):
            left_distance = left_distance - 2 * amax * vy / jerk - 2 * amax**3 / (3 * jerk **2)
        if(left_distance>0):
            lim_vy = math.sqrt(2*amax*left_distance)
        else: lim_vy = 0
        # v_max2 = compute_vx(30,radius,error,sense_range,safe_bound,amax,latency)
        # v_max3 = compute_vx(2,radius,error,sense_range,safe_bound,amax,latency)
        e_list.append(error2)
        v_list.append(v_max)
        v2_list.append(vy)
        v3_list.append(v_max_comp)
        v4_list.append(v_max_comp2)
        v5_list.append(v_max_comp3)
        v6_list.append(v_max_comp4)
        s_list.append(sense_range2)


        y_list.append(y)
        y2 = radius+safe_bound+error*(sense_range-safe_bound)
        y2_list.append(y2)
        lef_list.append(left_distance)
        lm_list.append(lim_vy)

    vx_maxline = (sense_range - safe_bound) / (latency + math.sqrt(2 * (radius+safe_bound) / amax))
    print(vx_maxline)

    plt.figure(0)
    # plt.plot(v_list, v_list, label='vx')
    plt.plot(v_list, v2_list, label='vy')
    plt.plot(v_list, lm_list, label='v_yT,max')
    plt.axvline( vx_maxline,color='red', label='v_x,max')
    plt.axvline( 11.57,color='black', linestyle='--')
    plt.axvline( 15.63,color='black', linestyle='--')
    # plt.title("vx-S")
    plt.xlabel("vx m/s")
    plt.ylabel("vy m/s")
    plt.legend()

    plt.figure(1)
    # plt.plot(v_list, y_list, label='y')
    # plt.plot(v_list, lef_list ,label='left')
    plt.plot(e_list, v3_list, label='R = 3m')
    plt.plot(e_list, v4_list, label='R = 3.5m')
    plt.xlabel("position_error m/s")
    plt.ylabel("max_vx m/s")
    plt.legend()

    t_T_list = []

    vx_list = []
    for i in range(100,20000):
        v_max = (i+1) * 0.001 

        vy,t_left,T = compute_vy(v_max,radius,error,sense_range,safe_bound,amax,latency)
        
        t_T_list.append(t_left/T)


        vx_list.append(v_max)

    plt.figure(2)
    # plt.plot(vx_list, t_left_list, label='t_left')
    # plt.plot(vx_list, T_list, label='T')
    plt.plot(vx_list, t_T_list, label='t\'/T')
    # plt.title("Percentage of the maximum time for acceleration")
    plt.xlabel("vx m/s")
    plt.ylabel("t'/T ")
    plt.axvline( 11.57,color='black', linestyle='--')
    plt.axvline( 15.63,color='black', linestyle='--')
    plt.legend()

    plt.figure(3)
    plt.plot(v_list, y_list, label='Actual y')
    plt.plot(v_list, y2_list, label='Ideal y')
    plt.axvline( 11.57,color='black', linestyle='--')
    plt.axvline( 15.63,color='black', linestyle='--')
    plt.xlabel("vx m/s")
    plt.ylabel("y m")
    plt.legend()

    plt.figure(4)
    plt.plot(s_list, v5_list, label='error = 0.01')
    plt.plot(s_list, v6_list, label='error = 0.03')
    plt.xlabel("sensor_range m")
    plt.ylabel("max_vx m/s")
    plt.legend()

    plt.show()
    #
    # -----------------------------------------------------plot2D---------------------------------------------
    # # pass
    # from matplotlib import cm
    # from mpl_toolkits.mplot3d import Axes3D

    # e,t = np.mgrid[0:5:100j,0:0.1:100j]
    # z = np.zeros((100,100))
    # for i in range(100):
    #     for j in range(100):
    #         sense_range = (i+1) * 0.05
    #         error = (j+1) * 0.001

    #         v = compute_vx(R,radius,error,sense_range,safe_bound,amax,latency)
    #         z[i][j] = v
    # fig = plt.figure(figsize=(8,6))
    # ax = fig.add_axes(Axes3D(fig))
    # ax.plot_surface(e,t,z,cmap=cm.ocean)
    # ax.set_xlabel('sensor range')
    # ax.set_ylabel('localization error')
    # ax.set_title('vmax_range_latency')
    # plt.show()
