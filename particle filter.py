# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 08:57:37 2020

@author: CMH
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt

from load_data import get_joint, get_lidar, get_depth, get_rgb, getExtrinsics_IR_RGB, getIRCalib, getRGBCalib
from p2_utils import mapCorrelation, bresenham2D
from function import MapInitialize, softmax,T_matrix, mapping, lamba2Binary



lidar_file = "lidar/train_lidar0"
lidar_data = get_lidar(lidar_file)

joint_file = "joint/train_joint0"
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
    
#%%
#Initialize MAP
MAP = MapInitialize()
lamda = MAP['map'] 
resolution = MAP['res'] #0.05
l = 5
N = 25
particle_state = np.zeros((N,3))
weight = np.ones(N)/N

x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
xdif, ydif = np.arange(-resolution*l, resolution*l + resolution, resolution),  np.arange(-resolution*l, resolution*l + resolution, resolution)

#Initialize Particles

noise = np.array([0.001, 0.001, 0.1*np.pi/180])
lidar_angles = np.arange(-135,135.25,0.25)*np.pi/180.0
trajectory = np.empty(shape=(1,2))
best_trajectory=[]
best_particle_state = np.zeros((3,))


xt_curr = 0
yt_curr = 0
thetat_curr = 0
x_t = np.zeros((len(lidar_data),1))
y_t = np.zeros((len(lidar_data),1))
theta_t = np.zeros((len(lidar_data),1))

    

for i, data in enumerate(lidar_data):
    xt_curr += data['delta_pose'][0][0]
    yt_curr += data['delta_pose'][0][1]
    thetat_curr += data['delta_pose'][0][2] 
    x_t[i] = xt_curr
    y_t[i] = yt_curr
    theta_t[i] = thetat_curr



#updating loop
for i in range(0, len(lidar_data), 1):
    print(i)
    data = lidar_data[i]

    noise_motion = np.random.randn(N,3) * noise
    particle_state[:,2] %= (2*np.pi)
    particle_state = particle_state + data['delta_pose'] + noise_motion

 
    bi_lamda = np.zeros_like(lamda)
    bi_lamda = lamba2Binary(lamda, bi_lamda)
    correlation = []
    
    #correlation 
    for j in range(N):
        scan = T_matrix(data['joint'], particle_state[j,:],data['scan'], lidar_angles)
        #print(scan)
        x, y = scan[:,0]/resolution + lamda.shape[0]//2, scan[:,1]/resolution + lamda.shape[1]//2
        
        # offset to center
        vp = np.vstack((x, y))
        particle_cor_x = xdif + particle_state[j,0] 
        particle_cor_y = ydif + particle_state[j,1] 
        #offset
        #print(particle_state[j,0])
        map_correlation = mapCorrelation(bi_lamda, x_im, y_im, vp, (particle_cor_x / resolution) , (particle_cor_y / resolution))
        #index = np.argmax(map_correlation)
        #print(np.max(map_correlation))
        correlation.append(0.02*np.max(map_correlation))
        #print(correlation)
        
    #get weight  
    correlation = weight*np.array(correlation)
    weight = softmax(correlation)
    
    #build map
    
    best_particle_index = np.argmax(weight)
    best_particle_state = particle_state[best_particle_index]
    best_trajectory.append(best_particle_state)
    best_scan = T_matrix(data['joint'], best_particle_state, data['scan'], lidar_angles)
    xi, yi   = (best_scan[:,0]/resolution).astype(int), (best_scan[:,1]/resolution).astype(int)
    lamda = mapping(lamda, xi, yi be,st_particle_state, resolution)
    
    #plot
    plt.plot(y_t / resolution + lamda.shape[0]//2, x_t / resolution + lamda.shape[1]//2, c='g')
    plt.imshow(lamda, cmap='gray', vmin=-10, vmax=10)
    plt.show()
    

    


