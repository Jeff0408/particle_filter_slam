# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 08:52:45 2020

@author: CMH
"""

import numpy as np
from numpy.random import randn, rand
import matplotlib.pyplot as plt; plt.ion()
import cv2
from p2_utils import bresenham2D
from numpy.linalg import inv
from scipy.special import logsumexp
from scipy.special import expit


       
def MapInitialize():
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -20  #meters
    MAP['ymin']  = -20
    MAP['xmax']  =  20
    MAP['ymax']  =  20 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']  = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.int8) 
    
    return MAP

def w2b(x,y,dphi,phi,theta):
    
    w2b  = np.array([[np.cos(dphi), -np.sin(dphi), 0, x], [np.sin(dphi), np.cos(dphi), 0, y], [0, 0, 1, 0.93], [0, 0, 0, 1]])
     
    return w2b
    
def b2h(x,y,dphi,phi,theta):
    
    b2h = np.array([[np.cos(phi)*np.cos(theta), -np.sin(phi), np.cos(phi)*np.sin(theta), 0], [np.sin(phi)*np.cos(theta), np.cos(phi), np.sin(phi)*np.sin(theta), 0], [-np.sin(theta), 0, np.cos(theta), 0.33], [0, 0, 0, 1]])
    
    return b2h
       
    
def h2l():
    
    h2l = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.15], [0, 0, 0, 1]])
    
    return h2l


def T_matrix(data, pose, points, angles):
    
    phi      = data[0]
    theta    = data[1]
    x    = pose[0]
    y    = pose[1]
    dphi = pose[2]
   
    matrix = w2b(x,y,dphi,phi,theta)(*(b2h(x,y,dphi,phi,theta)*h2l()))
    sc = np.empty([1,2])
    
    a = points * np.cos(angles) 
    b = points * np.sin(angles)
    
    for (a_i,b_i) in zip(a.flatten().tolist(), b.flatten().tolist()):
        
        sc = np.vstack((sc, np.array([[a_i, b_i]])))
        
    sc = sc[1:].T
    sc = np.dot(matrix, sc)
    sc = sc.T
        
    return sc
    

def lamba2Binary(lamda, bi_lamda): 
    bi_lamda[lamda < 0] = 0
    bi_lamda[lamda == 0] = 0
    bi_lamda[lamda > 0] = 1
    return bi_lamda

def mapping(lamda, xi, yi, pos_now, res):

    empty_set = {}
    wall_set = {}
    log_odd  = np.log(9)
    
    for i , (a,b) in enumerate(zip(xi,yi)):
        #print(i)
        #get the grids where beam can go through
        bresenham = np.array(bresenham2D(int(pos_now[0]/res),int(pos_now[1]/res),a,b)).astype(int)
        #xx, yy is the position of the wall with respect to robot
        xx = a + lamda.shape[0]//2 
        yy = b + lamda.shape[1]//2
        wall_set[xx,yy]= True
        # "-1" because the last element is the wall
        for j in range(len(bresenham[0])-1):
            empty_set[(bresenham[0][j] +  lamda.shape[0]//2 ,bresenham[1][j] + lamda.shape[1]//2)] = True
            
    for j, _ in wall_set.items():
        xx,yy = j[0], j[1]
        if 0<=xx<=lamda.shape[0]-1 and 0<=yy<=lamda.shape[1]-1:
            if(lamda[xx,yy] <= 10):
                lamda[xx,yy] += log_odd
            #print('wall :' , grid[xx, yy])
            #grid[xx,yy]+= occup_odds
            
    for k, _ in empty_set.items():
        xx,yy = k[0], k[1]
        #xx = xx + int(pos_now[0])+ grid.shape[0]//2
        #yy = yy + int(pos_now[1])+ grid.shape[0]//2
        if 0<=xx<=lamda.shape[0]-1 and 0<=yy<=lamda.shape[1]-1:
            if(lamda[xx,yy] >= -10):
                lamda[xx,yy] -= log_odd
            #print('empty :' , grid[xx, yy])
            
    #print ('grid', grid)
        
    return lamda


def xy2map(x):
    MAP = MapInitialize()
    xis = np.ceil((x - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    return xis

def softmax(x):
    e_x = np.exp(x-np.max(x))
    return e_x / e_x.sum()


