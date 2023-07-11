#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 13:37:20 2022

@author: juanmeriles
"""

import sympy as sym
import numpy as np
import matplotlib.pyplot as plt

lam = 4 * 10**10
miu = 2.7 * 10**10
rho = 2.7 * 10**3

Xref = np.array([[0,0,0,1,0,0,1,1,0,0,1,0]])

X1 = sym.Symbol('X1')
X2 = sym.Symbol('X2')
X3 = sym.Symbol('X3')
u11 = sym.Symbol('u11')
u21 = sym.Symbol('u21')
u31 = sym.Symbol('u31')
u41 = sym.Symbol('u41')
u12 = sym.Symbol('u12')
u22 = sym.Symbol('u22')
u32 = sym.Symbol('u32')
u42 = sym.Symbol('u42')
u13 = sym.Symbol('u13')
u23 = sym.Symbol('u23')
u33 = sym.Symbol('u33')
u43 = sym.Symbol('u43')
t = sym.Symbol('t')

N1 = (1-X1)*(1-X2)
N2 = X1*(1-X2)
N3 = X1*X2
N4 = (1-X1)*X2
N = [N1,N2,N3,N4]

Ne = np.array([[N1,0,0,N2,0,0,N3,0,0,N4,0,0],
               [0,N1,0,0,N2,0,0,N3,0,0,N4,0],
               [0,0,N1,0,0,N2,0,0,N3,0,0,N4]])

Be = []
for i in range(len(N)):
    Be.append([[sym.diff(N[i],X1),0,0],
              [0,sym.diff(N[i],X2),0],
              [0,0,sym.diff(N[i],X3)],
              [sym.diff(N[i],X2),0,0],
              [0,sym.diff(N[i],X3),0],
              [0,0,sym.diff(N[i],X1)],
              [sym.diff(N[i],X3),0,0],
              [0,sym.diff(N[i],X1),0],
              [0,0,sym.diff(N[i],X2)]])

Be= np.array(Be)
Be = np.hstack(Be)

ue = np.array([[0,0,0,u21,u22,0,u31,u32,0,0,0,0]])
U = Ne @ ue.T

F = np.array([[sym.diff(U[0],X1)[0],sym.diff(U[0],X2)[0],0],
              [sym.diff(U[1],X1)[0],sym.diff(U[1],X2)[0],0],
              [0,0,0]])
F = F + np.eye(3)
C = F.T @ F
E = 1/2*(C-np.eye(3))
S = lam*(np.trace(E))*np.eye(3)+2*miu*E
P = F@S
P = np.array([[P[0][0],P[1][1],P[2][2],P[0][1],P[1][2],P[2][0],P[0][2],P[1][0],P[2][1]]])
P = P.T
M = rho * Ne.T @ Ne


R = Be.T @ P

#2nd order Gauss quadrature
e = [-.57735,.57735]
n = [-.57735,.57735]

R_int = []
M_int = np.zeros((np.shape(M)[0],np.shape(M)[1]))

F_t = np.array([[0,0,0,(10**8)*t/6,0,0,(10**8)*t/2,0,0,0,0,0]])
F_b = rho*Ne.T@np.array([[0],[(-10**7)*X1*X2*t],[0]])
F = F_t.T+F_b
F_int = []

for i in range(len(R)):
    temp1 = 0
    for j in range(len(n)):
        for k in range(len(e)):
                if R[i]!=0:
                    temp1 = temp1+R[i][0].subs(X1,(e[j]/2+1/2)).subs(X2,(n[k]/2+1/2))
    R_int.append(1/4*temp1)

for l in range(np.shape(M)[0]):
    for p in range(np.shape(M)[1]):
        temp2 = 0
        for j in range(len(n)):
            for k in range(len(e)):
                if M[l][p]!=0:
                    temp2 = temp2+M[l][p].subs(X1,(e[j]/2+1/2)).subs(X2,(n[k]/2+1/2))

        M_int[l][p] = 1/4*temp2

for i in range(len(F)):
    temp3 = 0
    for j in range(len(n)):
        for k in range(len(e)):
                if F[i]!=0:
                    temp3 = temp3+F[i][0].subs(X1,(e[j]/2+1/2)).subs(X2,(n[k]/2+1/2))

    F_int.append(1/4*temp3)

#To approximate derivative

def approxD(current_u,func):
    eps = np.finfo(float).eps
    w = (10**3)*eps
    
    DA_o = func(current_u[0],current_u[1],\
                current_u[2],current_u[3])
    DA = []
    for i in range(len(ue[0])):
        pert = np.zeros(len(ue[0]))
        pert[i] = w
        u_cur_tot = np.array([0,0,0,current_u[0],current_u[1],0,current_u[2],current_u[3],0,0,0,0])
        u_pert = u_cur_tot+pert
        
        DA_col = func(u_pert[3],u_pert[4],\
                      u_pert[6],u_pert[7])
        DA_col = (np.array([DA_col]) - np.array([DA_o]))/w
        DA.append(DA_col[0])
        
    DA = np.hstack(DA)
    freeDofs = [0,0,0,1,1,0,1,1,0,0,0,0]     
    for i in range(len(freeDofs)-1,-1,-1):
        if freeDofs[i] == 0:
            DA = np.delete(DA,i,0)
            DA = np.delete(DA,i,1)
    return DA

#Do newmark method
beta = .25
gamma = .5

u_hist = []
v_hist = []
a_hist = []
F_hist = []
R_hist = []

time = 0
dt = 1
tf = 1
steps = np.arange(dt,tf+.000000001,dt)
u_hist.append(np.zeros(12))
v_hist.append(np.zeros(12))
a_hist.append(np.zeros(12))
F_hist.append(np.zeros(12))
R_hist.append(np.zeros(12))

F_temp = []
k=0
func2 = sym.lambdify([t],F_int)
func3 = sym.lambdify([t,u11,u12,u13,u21,u22,u23,u31,u32,u33,u41,u42,u43],R_int)
for n in steps:
    print(k)
    F_temp = func2(n)
    # for i in range(len(F_int)):
    #     if F_int[i]!=0:
    #         F_temp.append(F_int[i].subs(t,n))
    #     else:
    #         F_temp.append(0)
    F_hist.append(F_temp)
    
    F_eff = F_hist[k+1]+M_int @ ((np.array(u_hist[k])+np.array(v_hist[k])*dt)*1/(beta*dt**2)+((1-2*beta)/(2*beta))*np.array(a_hist[k]))
    
    
    def Substitute1D(vect,time,u_hist):
        vect_eval = []
        for i in range(len(vect)):        
            if vect[i] !=0:
                temp = vect[i].subs(u11,u_hist[time][0])
                temp = temp.subs(u12,u_hist[time][1])
                temp = temp.subs(u13,u_hist[time][2])
                temp = temp.subs(u21,u_hist[time][3])
                temp = temp.subs(u22,u_hist[time][4])
                temp = temp.subs(u23,u_hist[time][5])
                temp = temp.subs(u31,u_hist[time][6])
                temp = temp.subs(u32,u_hist[time][7])
                temp = temp.subs(u33,u_hist[time][8])
                temp = temp.subs(u41,u_hist[time][9])
                temp = temp.subs(u42,u_hist[time][10])
                temp = temp.subs(u43,u_hist[time][11])
                temp = temp.subs(t,time)
                vect_eval.append(temp)
            else:
                vect_eval.append(0)
        return vect_eval
            
        
    R_hist.append(Substitute1D(R_int,0,u_hist))
    
    
    A = (1/(beta*dt**2))*M_int @ ue.T + np.array([R_int]).T-np.array([F_eff]).T
    func = sym.lambdify([u21,u22,u31,u32],A)
    
    def func1(u,func):
        vec = func(u[0],u[1],u[2],u[3])
        return np.array([vec[4],vec[5],vec[7],vec[8]]).T[0]
        
      
    #Implement Newton Raphson formulation
    DA_o = func(u_hist[0][3],u_hist[0][4],\
                u_hist[0][6],u_hist[0][7])
    #u_it_n = np.array([u_hist[0][3],u_hist[0][4],u_hist[0][6],u_hist[0][7]])
    u_it_n = np.array([0,0,0,0])
    f_it_n = np.array([DA_o[3][0],DA_o[4][0],DA_o[6][0],DA_o[7][0]])
    
    error = 10
    i = 0
    while error>10**-5:
        #print(i)
        current_u = u_it_n
        DA = approxD(current_u,func)
        u_it_n1 = u_it_n - np.linalg.solve(DA,f_it_n)
        #print(u_it_n1)
        new_f = func(u_it_n1[0],u_it_n1[1],\
                u_it_n1[2],u_it_n1[3])
        f_it_n = np.array([new_f[3][0],new_f[4][0],new_f[6][0],new_f[7][0]])
        error= np.linalg.norm(f_it_n,2)
        u_it_n = u_it_n1
        i = i+1
        
    u_hist.append(np.array([u_hist[0][0],u_hist[0][0],u_hist[0][0],u_it_n1[0],u_it_n1[1],u_hist[0][0],\
                            u_it_n1[2],u_it_n1[3],u_hist[0][0],u_hist[0][0],u_hist[0][0],u_hist[0][0]]))
    anew = (1/(beta*(dt**2)))*(u_hist[k+1]-u_hist[k]-v_hist[k]*dt)-((1-2*beta)/(2*beta))*a_hist[k]
    a_hist.append(anew)
    vnew = v_hist[k]+((1-gamma)*a_hist[k]+gamma*a_hist[k+1])*dt
    v_hist.append(vnew)
    k = k+1
    
        
pos_x_ref = [Xref[0][0],Xref[0][3],Xref[0][6],Xref[0][9],Xref[0][0]]
pos_y_ref = [Xref[0][1],Xref[0][4],Xref[0][7],Xref[0][10],Xref[0][1]]

pos_x_def = [Xref[0][0],Xref[0][3]+u_hist[-1][3],Xref[0][6]+u_hist[-1][6],Xref[0][9],Xref[0][0]]
pos_y_def = [Xref[0][1],Xref[0][4]+u_hist[-1][4],Xref[0][7]+u_hist[-1][7],Xref[0][10],Xref[0][1]]

plt.plot(pos_x_ref,pos_y_ref)
plt.plot(pos_x_def,pos_y_def)


#Explicit Newmark
M_int_lump = rho*1/4*np.eye(12)
beta = 0
gamma = .5
dt = 20*10**-6
tf = 1

u_hist_exp = []
v_hist_exp = []
a_hist_exp = []
F_hist_exp = []
R_hist_exp = []

steps = np.arange(dt,tf+.000000001,dt)
u_hist_exp.append(np.zeros(12))
v_hist_exp.append(np.zeros(12))
a_hist_exp.append(np.zeros(12))
F_hist_exp.append(np.zeros(12))
R_hist_exp.append(np.zeros(12))

u_hist_exp.append(np.zeros(12))
v_hist_exp.append(np.zeros(12))
a_hist_exp.append(np.zeros(12))
F_hist_exp.append(np.zeros(12))
R_hist_exp.append(np.zeros(12))

k = 0
func2 = sym.lambdify([t],F_int)

count =0
for n in steps:
    #print(k)
    F_temp = func2(n)
    
    # for i in range(len(F_int)):
    #     if F_int[i]!=0:
    #         F_temp.append(func2(n))
    #     else:
    #         F_temp.append(0)
    F_hist_exp[k+1]=(np.array(F_temp))
    u_hist_exp[k+1]=(u_hist_exp[k]+v_hist_exp[k]*dt+(1/2)*a_hist_exp[k]*dt**2)
    #print(F_hist_exp)
    Feff = (np.array(F_hist_exp[k+1])-np.array(func3(n,0,0,0,u_hist_exp[k+1][3],u_hist_exp[k+1][4],0,u_hist_exp[k+1][6],u_hist_exp[k+1][7],0,0,0,0)))
    anew = np.linalg.inv(M_int_lump)@Feff
    anew = np.array([0,0,0,anew[3],anew[4],0,anew[6],anew[7],0,0,0,0])
    a_hist_exp[k+1]=(anew)
    vnew = v_hist_exp[k]+((1-gamma)*a_hist_exp[k]+gamma*a_hist_exp[k+1])*dt
    v_hist_exp[k+1]=(vnew)
    u_hist_exp[k] = u_hist_exp[k+1]
    v_hist_exp[k] = v_hist_exp[k+1]
    a_hist_exp[k] = a_hist_exp[k+1]
    F_hist_exp[k] = F_hist_exp[k+1]
    print(count)
    count = count+1
    
plt.figure(3)
pos_x_def = [Xref[0][0],Xref[0][3]+u_hist_exp[-1][3],Xref[0][6]+u_hist_exp[-1][6],Xref[0][9],Xref[0][0]]
pos_y_def = [Xref[0][1],Xref[0][4]+u_hist_exp[-1][4],Xref[0][7]+u_hist_exp[-1][7],Xref[0][10],Xref[0][1]]

plt.plot(pos_x_ref,pos_y_ref)
plt.plot(pos_x_def,pos_y_def)






