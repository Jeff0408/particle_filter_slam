# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 00:31:39 2020

@author: CMH
"""
import os
import math
import numpy as np
import matplotlib.pyplot as plt

from load_data import get_joint, get_lidar, get_depth, get_rgb, getExtrinsics_IR_RGB, getIRCalib, getRGBCalib
from p2_utils import mapCorrelation, bresenham2D
from function import MapInitialize,softmax,T_matrix, mapping



lidar_file = "lidar/train_lidar2"
lidar_data = get_lidar(lidar_file)

joint_file = "joint/train_joint2"
joint_data = get_joint(joint_file)
joint_time = joint_data['ts']


# time matching
for i in range(joint_time.shape[1]):
    joint_time[0,i] = round(joint_time[0,i],2)
    
joint_time = joint_time[0,:]


for i in range(len(lidar_data)):
    lidar_time = round(lidar_data[i]['t'][0,0], 2)
    
    while 1:
        index = np.where(np.array(joint_time)==lidar_time)
        
        if index[0].size == 0:
            lidar_time = round((lidar_time - 0.01), 2)
        else:
            break

    lidar_data[i]['joint'] = joint_data['head_angles'][:,index[0][0]]

#Initialize MAP
MAP = MapInitialize()
lamda = MAP['map'] 
resolution = MAP['res'] #0.05
l = 5
N = 1
particle_state = np.zeros((N,3))
weight = np.ones(N)/N

x_im = np.arange(lamda.shape[0]) #x-positions of each pixel of the map
y_im = np.arange(lamda.shape[1]) #y-positions of each pixel of the map
xdif, ydif = np.arange(-resolution*l, resolution*l + resolution, resolution),  np.arange(-resolution*l, resolution*l + resolution, resolution)


noise_motion = np.array([0.001, 0.001, 0.3*np.pi/180])
lidar_angles = np.arange(-135,135.25,0.25)*np.pi/180.0    


#robot position from lidar data
x_t = np.zeros((len(lidar_data),1))
y_t = np.zeros((len(lidar_data),1))
theta_t = np.zeros((len(lidar_data),1))

xt_curr = 0
yt_curr = 0
thetat_curr = 0

for i, data in enumerate(lidar_data):
    xt_curr += data['delta_pose'][0][0]
    yt_curr += data['delta_pose'][0][1]
    thetat_curr += data['delta_pose'][0][2] 
    x_t[i] = xt_curr
    y_t[i] = yt_curr
    theta_t[i] = thetat_curr

#trajectory    



for i in range(0, len(lidar_data), 1):
    
    print(i)
    data = lidar_data[i]
    
    particle_state[:,2] %= (2*np.pi)
    particle_state = particle_state + data['delta_pose'] 
    scan = T_matrix(data['joint'], particle_state[0],data['scan'], lidar_angles)
    
    xi, yi   = (scan[:,0]/resolution).astype(int), (scan[:,1]/resolution).astype(int)
    lamda = mapping(lamda, xi, yi,  particle_state[0], resolution)
    
    plt.plot(y_t / resolution + lamda.shape[0]//2, x_t / resolution + lamda.shape[1]//2, c='g')
    plt.imshow(lamda, cmap='gray', vmin=-10, vmax=10, origin = 'lower')
#plt.scatter(x_t, y_t)    
#plt.scatter(particle_state[0][0], particle_state[0][1])       
    #plt.imshow(lamda, cmap='gray', vmin=-10, vmax=10)
#plt.show()
