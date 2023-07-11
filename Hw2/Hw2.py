#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:52:07 2022

@author: juanmeriles
"""
import numpy as np
import matplotlib.pyplot as plt
x = [1,1,0]
F = np.array([[1+x[0],(x[0]**2)*x[1],0],
             [-2*x[1],1+2*x[0]*x[1],0],
             [0,0,1]])
FT = F.transpose()
J = np.linalg.det(F)
DJDF1 = np.zeros((6,3,3))
DJDF2 = np.zeros((6,3,3))
eps = np.finfo(float).eps
w = np.array([10**0,10**1,10**2,10**3,10**4,10**5])*eps


dirs = np.zeros((3,3))

for j in range(len(w)):
    count1 = 0
    count2 = 0
    for i in range(len(dirs)):
        for n in range(len(dirs[0])):
            dirs = np.zeros((3,3))
            dirs[i,n] = 1
            Fad = F+w[j]*dirs
            Jad1 = np.linalg.det(Fad)
            DJDF1[j,i,n] = (1/w[j])*(Jad1-J)
            DJDF2[j,i,n] = (1/w[j])*(Jad1**2-J**2)/(2*J)


    #DJDF1[j,:,:] = DJDF1[j,:,:] @ np.linalg.inv(FT)
    #DJDF2[j,:,:] = DJDF2[j,:,:] @ np.linalg.inv(FT)

DJDF = np.linalg.inv(FT)*J
error1 = np.zeros(len(w))
error2 = np.zeros(len(w))

for i in range(len(w)):
    error1[i] = np.linalg.norm(DJDF-DJDF1[i,:,:],2)
    error2[i] = np.linalg.norm(DJDF-DJDF1[i,:,:],2)

plt.figure(2)
plt.plot(error1,label = "Traditional Estimate")
plt.plot(error2,label = "Squared Estimate")
plt.xlabel("Order of perterbation")
plt.ylabel("2 norm of error")
plt.legend()
           
        
    

             