#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  9 15:47:40 2022

@author: juanmeriles
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 22:22:35 2022

@author: juanmeriles
"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sympy as sym
from matplotlib.animation import FuncAnimation
import matplotlib
    
class element:
    NODE = []
    CON = []
    BOUN = []
    id_v = []
    
    def __init__(self):
        pass 
    
def createBlock(origin,lenx,leny,numx,numy):
    xlocs = np.arange(origin[0]-lenx/2,origin[0]+lenx/2+.00000000000001,lenx/(numx-1))
    ylocs = np.arange(origin[1]-leny/2,origin[1]+leny/2+.00000000000001,leny/(numy-1))
    #print(len(xlocs))
    #print(len(ylocs))
    nodes = np.zeros((2,numx*numy))
    count = 0
    DOFcount = 0
    globalDOF = []
    for i in range(len(ylocs)):
        for j in range(len(xlocs)):
            nodes[0][count] = xlocs[j]
            nodes[1][count] = ylocs[i]
            globalDOF.append([DOFcount,DOFcount+1])
            count = count+1
            DOFcount = DOFcount+2
    con = []
    count = 0
    elid = []
    for i in range(len(ylocs)-1):
        for j in range(len(xlocs)-1):
            con.append([count,count+1,count+numx+1,count+numx])
            elid.append(np.hstack([globalDOF[count],globalDOF[count+1],globalDOF[count+numx+1],globalDOF[count+numx]]))
            if (j == len(xlocs)-2):
                count = count+2
            else:
                count = count+1
            #print(count)

    
    return nodes,con,elid,xlocs,ylocs

def ShapeFcn(e,n):
    N1 = 1/4*(1-e)*(1-n)
    N2 = 1/4*(1+e)*(1-n)
    N3 = 1/4*(1+e)*(1+n)
    N4 = 1/4*(1-e)*(1+n)
    N = np.array([[N1,0,N2,0,N3,0,N4,0],
                  [0,N1,0,N2,0,N3,0,N4]])
    Nsmall = np.array([[N1,N2,N3,N4]])
    return N, Nsmall

def MakeB(e,n,el):
    dN1de = 1/4*(-1)*(1-n)
    dN2de = 1/4*(1)*(1-n)
    dN3de = 1/4*(1)*(1+n)
    dN4de = 1/4*(-1)*(1+n)
    dN1dn = 1/4*(1-e)*(-1)
    dN2dn = 1/4*(1+e)*(-1)
    dN3dn = 1/4*(1+e)*(1)
    dN4dn = 1/4*(1-e)*(1)
    J = Jacobian(e,n,el)
    temp = np.linalg.inv(J) @ np.array([[dN1de],[dN1dn]])
    dN1dx = temp[0][0]
    dN1dy = temp[1][0]
    temp = np.linalg.inv(J) @ np.array([[dN2de],[dN2dn]])
    dN2dx = temp[0][0]
    dN2dy = temp[1][0]
    temp = np.linalg.inv(J) @ np.array([[dN3de],[dN3dn]])
    dN3dx = temp[0][0]
    dN3dy = temp[1][0]
    temp = np.linalg.inv(J) @ np.array([[dN4de],[dN4dn]])
    dN4dx = temp[0][0]
    dN4dy = temp[1][0]
    
    
    B = np.array([[dN1dx,0,dN2dx,0,dN3dx,0,dN4dx,0],
                  [0,dN1dy,0,dN2dy,0,dN3dy,0,dN4dy],
                  [dN1dy,0,dN2dy,0,dN3dy,0,dN4dy,0],
                  [0,dN1dx,0,dN2dx,0,dN3dx,0,dN4dx]])
    return B

def Jacobian(e,n,el):
    dxde = 1/4*(-(1-n)*el.NODE[0][0]+(1-n)*el.NODE[1][0]+(1+n)*el.NODE[2][0]-(1+n)*el.NODE[3][0])
    dyde = 1/4*(-(1-n)*el.NODE[0][1]+(1-n)*el.NODE[1][1]+(1+n)*el.NODE[2][1]-(1+n)*el.NODE[3][1])
    dxdn = 1/4*(-(1-e)*el.NODE[0][0]-(1+e)*el.NODE[1][0]+(1+e)*el.NODE[2][0]+(1-e)*el.NODE[3][0])
    dydn = 1/4*(-(1-e)*el.NODE[0][1]-(1+e)*el.NODE[1][1]+(1+e)*el.NODE[2][1]+(1-e)*el.NODE[3][1])
    
    J = np.array([[dxde,dyde],
                  [dxdn,dydn]])
    return J

def createKe(el,Bgauss,p0,Beta,dt):
    gp = [-0.57735,0.57735]
    Ke = np.zeros((8,8))
    count = 0
    for i in gp:
        for j in gp:
            B = Bgauss[count]
            F = Fmat(i,j,el)
            DPDF = DPDFmat(F)
            J = Jacobian(i,j,el)
            J = np.linalg.det(J)
            Ne, Ne_small = ShapeFcn(i,j)
            Me = createMe(p0,el)
            Ke = Ke+(B.T@DPDF@B)*J
            count = count+1
    Ke = Ke + (1/(Beta*dt**2))*Me
    return Ke

def Fmat(e,n,el):
    U = el.nodeloc
    N,Nsmall = ShapeFcn(e,n)
    B = MakeB(e,n,el)
    dN1dx = B[0,0]
    dN2dx = B[0,2]
    dN3dx = B[0,4]
    dN4dx = B[0,6]
    dN1dy = B[1,1]
    dN2dy = B[1,3]
    dN3dy = B[1,5]
    dN4dy = B[1,7]
    
    dNdx = np.array([[dN1dx,0,dN2dx,0,dN3dx,0,dN4dx,0],
                     [0,dN1dx,0,dN2dx,0,dN3dx,0,dN4dx]])
    dNdy = np.array([[dN1dy,0,dN2dy,0,dN3dy,0,dN4dy,0],
                     [0,dN1dy,0,dN2dy,0,dN3dy,0,dN4dy]])
    
    F = np.array([dNdx@U.T,dNdy@U.T])
    F = np.hstack(F)
    return F

def createMe(rho,el):
    # gp = [-0.57735,0.57735]
    # Me = np.zeros((8,8))
    # count = 0
    # for i in gp:
    #     for j in gp:
    #         N,Nsmall = ShapeFcn(i, j)
    #         J = Jacobian(i,j,el)
    #         J = np.linalg.det(J)
    #         Me += rho*N.T@N*J
    # return Me
    Me = np.zeros((8,8))
    for i in range(len(Me)):
        J = Jacobian(0,0,el)
        J = np.linalg.det(J)
        Me[i,i] = 4*1/4*rho*J
    return Me

def createRe(el):
    gp = [-0.57735,0.57735]
    Re = np.zeros((8,1))
    count = 0
    for i in gp:
        for j in gp:
            B = Bgauss[count]
            F = Fmat(i,j,el)
            #print(F)
            P = Pmat(miu,lam,F)
            J = Jacobian(i,j,el)
            J = np.linalg.det(J)
            Re = Re + B.T@P.T*J
            count = count+1
    return Re



###Creating dPdF matrix
def Pmat(miu,lam,F):
    C = F.T@F
    E = (1/2)*(C-np.eye(2))
    S = 2*miu*E+lam*np.trace(E)*np.eye(2)
    P = F@S
    P = np.array([[P[0][0],P[1][1],P[0][1],P[1][0]]])
    return P



###Creating dPdF matrix
lam = 100
miu = 40
Youngs = miu*(3*lam+2*miu)/(lam+miu)
rho = .01
F11 = sym.Symbol('F11')
F12 = sym.Symbol('F12')
F21 = sym.Symbol('F21')
F22 = sym.Symbol('F22')

Fsym = np.array([[F11,F12],
              [F21,F22]])
C = Fsym.T@Fsym
E = (1/2)*(C-np.eye(2))
S = 2*miu*E+lam*np.trace(E)*np.eye(2)
P = Fsym@S

#DPDF
DP11F11 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][0],F11))
DP11F22 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][0],F22))
DP11F12 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][0],F12))
DP11F21 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][0],F21))
DP22F11 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][1],F11))
DP22F22 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][1],F22))
DP22F12 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][1],F12))
DP22F21 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][1],F21))
DP12F11 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][1],F11))
DP12F22 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][1],F22))
DP12F12 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][1],F12))
DP12F21 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[0][1],F21))
DP21F11 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][0],F11))
DP21F22 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][0],F22))
DP21F12 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][0],F12))
DP21F21 = sym.lambdify([F11,F12,F21,F22],sym.diff(P[1][0],F21))

def DPDFmat(F):
    temp11 = DP11F11(F[0][0],F[0][1],F[1][0],F[1][1])
    temp12 = DP11F22(F[0][0],F[0][1],F[1][0],F[1][1])
    temp13 = DP11F12(F[0][0],F[0][1],F[1][0],F[1][1])
    temp14 = DP11F21(F[0][0],F[0][1],F[1][0],F[1][1])
    temp21 = DP22F11(F[0][0],F[0][1],F[1][0],F[1][1])
    temp22 = DP22F22(F[0][0],F[0][1],F[1][0],F[1][1])
    temp23 = DP22F12(F[0][0],F[0][1],F[1][0],F[1][1])
    temp24 = DP22F21(F[0][0],F[0][1],F[1][0],F[1][1])
    temp31 = DP12F11(F[0][0],F[0][1],F[1][0],F[1][1])
    temp32 = DP12F22(F[0][0],F[0][1],F[1][0],F[1][1])
    temp33 = DP12F12(F[0][0],F[0][1],F[1][0],F[1][1])
    temp34 = DP12F21(F[0][0],F[0][1],F[1][0],F[1][1])
    temp41 = DP21F11(F[0][0],F[0][1],F[1][0],F[1][1])
    temp42 = DP21F22(F[0][0],F[0][1],F[1][0],F[1][1])
    temp43 = DP21F12(F[0][0],F[0][1],F[1][0],F[1][1])
    temp44 = DP21F21(F[0][0],F[0][1],F[1][0],F[1][1])
    DPDF = np.array([[temp11,temp12,temp13,temp14],
                     [temp21,temp22,temp23,temp24],
                     [temp31,temp32,temp33,temp34],
                     [temp41,temp42,temp43,temp44]])
    return DPDF

def Assemble(el):
    #assemble global matrices
    K = np.zeros((2*len(nodes),2*len(nodes)))
    R = np.zeros((2*len(nodes)))
    for i in range(len(el)):
        Re = createRe(el[i])
        Ke = createKe(el[i],Bgauss,rho,Beta,dt)
        for j in range(8):
            R[ReDOFS[elements[i].id_v[j]]] += Re[j][0]
            for k in range(8):
                ind1 = ReDOFS[el[i].id_v[j]]
                ind2 = ReDOFS[el[i].id_v[k]]
                K[ind1,ind2] += Ke[j,k]
    return K,R

def UpdateLoc(elvec,ut): 
    for i in range(len(elvec)):
        for j in range(len(elvec[0].nodeloc[0])):
            elvec[i].nodeloc[0][j] = elvec[i].nodeVec[0][j]+ut[ReDOFS[elvec[i].id_v[j]]]
    return elvec

def gapfunc(u):
    xw = 1+wall
    newpos = ogpos[BoundaryDOFS].T+u[BoundaryDOFS]
    
    return (xw-newpos)[0]

def stressCalc(el):
    #calculate cauchy stress for each element
    F = Fmat(0,0,el)
    B = F@F.T
    J = np.linalg.det(F)
    T = (((lam/2)*(np.trace(B)-2)-miu)*B+miu*B@B.T)/J
    return T

def energyCalc(el):
    #gp = [-0.57735,0.57735]
    gp = [-np.sqrt(3/5),0,np.sqrt(3/5)]
    w = [5/9,8/9,5/9]
    SE = 0
    KE = 0
    ind1 = 0
    for i in gp:
        ind2 = 0
        for j in gp:
            N, Nsmall = ShapeFcn(i,j)
            v_el = np.zeros(len(el.id_v))
            for k in range(len(v_el)):
                v_el[k] = v[ReDOFS[el.id_v[k]]]
            
            J = Jacobian(i, i, el)
            J = np.linalg.det(J)
            v_el = N@v_el
            KE += w[ind1]*w[ind2]*(1/2*rho*v_el.T@v_el)*J
            F = Fmat(i,j,el)
            C = F.T@F
            IC = np.trace(C)
            IIC = 1/2*((IC)**2-np.trace(C@C.T))
            SE += w[ind1]*w[ind2]*((1/8)*lam*(IC-2)**2+(1/4)*miu*(IC**2-2*IC-2*IIC+2))*J
            
            ind2 += 1
        ind1 += 1
            
    return SE+KE,KE,SE
    

    

#set up Nodes
numx = 21
numy = 5
nodes,con,elid,xlocs,ylocs = createBlock([.5,.1],1,.2,numx,numy)
nodes = nodes.T
ogpos = np.reshape(nodes,(2*len(nodes),1))
boun = []
cboun = []
vi = 10
wall = .1 #1

#Newmark params
Beta = .25
gamma = .5
#CFL Condition for dt
dtcfl = .05/np.sqrt(Youngs/rho)
dt = .0002
T = .05 #.25

for i in range(len(nodes)):
    if (nodes[i][0] == xlocs[-1]):
        boun.append([0,0])
        cboun.append([1,0])
    else:
        boun.append([0,0])
        cboun.append([0,0])
        
        
# for i in range(len(nodes)):
#         boun.append([0,0])
        
elements = []

for i in range(len(con)):
    elements.insert(i,element())
    elements[i].NODE = np.array([nodes[con[i][0]],nodes[con[i][1]],nodes[con[i][2]],nodes[con[i][3]]])
    elements[i].CON = con[i]
    elements[i].BOUN = np.array([boun[con[i][0]],boun[con[i][1]],boun[con[i][2]],boun[con[i][3]]])
    elements[i].nodeVec = np.array([[elements[i].NODE[0][0],elements[i].NODE[0][1],elements[i].NODE[1][0],elements[i].NODE[1][1],\
                                  elements[i].NODE[2][0],elements[i].NODE[2][1],elements[i].NODE[3][0],elements[i].NODE[3][1]]])
    elements[i].id_v = elid[i]
    elements[i].nodeloc = np.copy(elements[i].nodeVec)
    elements[i].nodelochist = []
    elements[i].nodelochist.append(np.copy(elements[i].nodeloc))
    
    
phist = []
phist.append(np.zeros((20*4)))
vhist = []
Energy = []
KEhist = []
SEhist = []
Lambdahist = []
ghist = []
BDOFS = []
BoundaryDOFS = []
FDOFS = []
Bgauss = []
count = 0


for i in range(len(nodes)):
    if (boun[i][0] == 1):
        BDOFS.append(int(count))
        count = count+1
    else:
        FDOFS.append(int(count))
        count = count+1
    if (boun[i][1] == 1):
        BDOFS.append(int(count))
        count = count+1
    else:
        FDOFS.append(int(count))
        count = count+1
        
count = 0
for i in range(len(nodes)):
    if (cboun[i][0] == 1):
        BoundaryDOFS.append(int(count))
        count = count+1
    else:
        count = count+1
    if (cboun[i][1] == 1):
        BoundaryDOFS.append(int(count))
        count = count+1
    else:
        count = count+1

#Creates the ReDOF and RepDOF vectors which map the old global dofs to new positions
if BDOFS != []:
    DOForder = np.hstack([FDOFS,BDOFS])
else:
    DOForder = np.array(FDOFS)
ReDOFS = [0]*(2*len(nodes))
for i in range(len(DOForder)):
    ReDOFS[int(DOForder[i])] = i 

#Create Be for gauss points
gp = [-0.57735,0.57735]
for i in range(len(gp)):
    for j in range(len(gp)):
        Bgauss.append(MakeB(gp[i],gp[j],elements[0]))


#Assemble mass matrix and constraint vector
Me = createMe(rho,elements[0])
M = np.zeros((2*len(nodes),2*len(nodes)))
for i in range(len(elements)):
        for j in range(8):
            for k in range(8):
                ind1 = ReDOFS[elements[i].id_v[j]]
                ind2 = ReDOFS[elements[i].id_v[k]]
                if ind1 == 0 and ind2 == 0:
                    print(i,j,k)
                M[ind1,ind2] += Me[j,k]


#assemble constraint vector
constind = 0
d = []
for i in range(len(DOForder)):
    d.append([0,0,0,0,0]) 
    if (i in BoundaryDOFS):
       d[i][constind] = 1
       constind += 1
d = np.vstack(d)
dc = d
   
Mff = M[0:len(FDOFS),0:len(FDOFS)]

#Create initial stiffness matrix and stress div vector 
K,R = Assemble(elements)
Kff = K[0:len(FDOFS),0:len(FDOFS)]
Rf = R[0:len(FDOFS)]
#Set initial vectors
u = np.zeros(len(FDOFS))
du = np.zeros(len(FDOFS))
v = np.zeros(len(DOForder))
ut = np.zeros(len(DOForder))
lamda = np.zeros(len(BoundaryDOFS))
dlam = np.zeros(len(BoundaryDOFS))
contact = np.zeros(5)
fullcontact = 0
for i in range(len(v)):
    if np.mod(i,2) == 0:
        v[i] = vi
v = v[DOForder]
v = v[0:len(FDOFS)]
vold = v
a = np.zeros(len(FDOFS))

T = np.arange(0,T,dt)
maxit = 10
vold = v
zero = np.zeros((5,5))
for i in tqdm(range(len(T))):
    err = 100
    tol = 10**-5
    unew = u
    aold = a
    it = 0
    g = gapfunc(u)
    while err>tol and it<maxit:
       
        anew = (1/(Beta*dt**2))*(unew-u-dt*v)-((1-2*Beta)/(2*Beta))*a
        #Splitting up matrices into known and unknown parts
        Kff = K[0:len(FDOFS),0:len(FDOFS)]
        Rf = R[0:len(FDOFS)]
        
        #Check Contact   
        lamlam = np.zeros((5,5))
        dT = np.copy(d.T)
        
        if fullcontact == 1:
            v[BoundaryDOFS] = 0
            a[BoundaryDOFS] = 0
            anew[BoundaryDOFS] = 0
            A = np.block([[Kff,d], [dT, lamlam]])
            b = -np.hstack(((M@anew+R)+d@lamda,g))
            du = np.linalg.solve(A,b)
            dlam = du[len(FDOFS):]
            du = du[0:len(FDOFS)]
            du[BoundaryDOFS] = 0
            lamda = lamda+dlam
            
            
        if fullcontact == 0:
            lamda = np.zeros(len(BoundaryDOFS))
            A = Kff
            b = -(M@anew+R)
            du = np.linalg.solve(A,b)
            du = du[0:len(FDOFS)]
    
        unew = unew+du
        g = gapfunc(unew)
        
        
        #check if contact should be initiated or ended
        for j in range(len(BoundaryDOFS)):
            if contact[j] == 0 and (g[j]<=-10**-10):
                print(g[j])
                contact[j] = 1
                unew[BoundaryDOFS[j]] -= du[BoundaryDOFS[j]]
                g = gapfunc(unew)
                
                #set the system to touch the wall
                unew[BoundaryDOFS[j]] += g[j]
                v[BoundaryDOFS[j]] = 0
                a[BoundaryDOFS[j]] = 0
                anew[BoundaryDOFS[j]] = 0
                
            elif contact[j] == 1 and (lamda[j]<-10**-3):
                contact[j] = 0
                #print('neg') 
                #print(lamda)
                unew[BoundaryDOFS[j]] = unew[BoundaryDOFS[j]] - du[BoundaryDOFS[j]]
                lamda[j] = lamda[j]-dlam[j]
          
        if np.all(contact == 1):
            fullcontact = 1
        else:
            fullcontact = 0
            
        g = gapfunc(unew)
        err = np.linalg.norm(b,2)
        ut = np.hstack([unew,np.zeros(len(BDOFS))])
        elements = UpdateLoc(elements, ut)
        K,R = Assemble(elements)
        it = it+1
        if it == maxit:
            print("iteration limit reached")
        
    a = (1/(Beta*dt**2))*(unew-u-dt*v)-((1-2*Beta)/(2*Beta))*a
    v = v + dt*((1-gamma)*aold+gamma*a)
    u = unew
    p = np.zeros((len(elements)))
    Energy.append(0)
    KEhist.append(0)
    SEhist.append(0)
    for l in range(len(elements)):
        elements[l].nodelochist.append(np.copy(elements[l].nodeloc))
        p[l] = stressCalc(elements[l])[0][0]
        
        #calculate energy
        temp1,temp2,temp3 = energyCalc(elements[l])
        Energy[i] += temp1
        KEhist[i] += temp2
        SEhist[i] += temp3
        
    #save gap and multipliers
    ghist.append(g)
    Lambdahist.append(lamda)
    phist.append(np.copy(p))
    

    
ghist = np.vstack(ghist)
Lambdahist = np.vstack(Lambdahist)
undef = []

#Animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig=plt.figure(2)
ax = fig.add_subplot(111)
#div = make_axes_locatable(ax)
#cax = div.append_axes('right', '5%', '5%')

                
ax.set_xlim(-.5,2.5)
ax.set_ylim(-.2,.4)

def DefShapeFrame(k,el):
#Plot deformed shape for a specific time
#Extrapolate and plot Deformed Shape

    defshape=[]
    p = np.zeros((len(elements)))
    pxlocs = np.zeros((len(elements)))
    pylocs = np.zeros((len(elements)))
    for i in range(len(el)):
        elemdefvec = np.reshape(elements[i].nodelochist[k],(4,2))
        elemdefvec = elemdefvec.T
        elemdefvec = np.hstack([elemdefvec,np.array([elemdefvec[:,0]]).T])
        defshape.extend(ax.plot(elemdefvec[0],elemdefvec[1],'r',linewidth=.75))
        
        N, Nsmall = ShapeFcn(0,0)
        centroidpos = N@elements[i].nodelochist[k].T
        pxlocs[i] = centroidpos[0][0]
        pylocs[i] = centroidpos[1][0]
        
        
    pxlocs = pxlocs.reshape(numy-1,numx-1)
    pylocs = pylocs.reshape(numy-1,numx-1)
    p = phist[k].reshape(numy-1,numx-1)
    
    #w = ax.axvline(1+wall)
    defshape.extend(ax.plot([1+wall,1+wall],[-.2,.4],'g'))
    cs = ax.contourf(pxlocs,pylocs,p,vmin = -20, vmax = 10)
    #plt.colorbar()
    #fig.colorbar(cs)
    defshape.extend(cs.collections)
    #defshape.extend(w)
    #cax.cla()
    #fig.colorbar(cs, cax=cax)
    
    return defshape

#animation = FuncAnimation(fig, DefShapeFrame, frames=10, fargs=(elements,),blit=True)
    
animation = FuncAnimation(fig,func=DefShapeFrame,frames=range(len(elements[0].nodelochist)),fargs=(elements,),interval=.001,blit=True)
#plt.colorbar(cs)
norm = matplotlib.colors.Normalize(vmin=-20, vmax=10, clip=False)
cblims = matplotlib.cm.ScalarMappable(norm)
plt.colorbar(cblims)
#plt.clim(-20,10)
#plt.show()


time = np.arange(0,dt*len(Energy),dt)
plt.figure(6)
plt.plot(time,Energy, label = 'Total Energy')
plt.plot(time,KEhist, label = 'Kinetic Energy')
plt.plot(time,SEhist, label = 'Strain Energy')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("Energy (J)")
plt.title("Total Energy as function of Time")

plt.figure(8)
plt.plot(time,Lambdahist[:,0],label = 'Node 1')
plt.plot(time,Lambdahist[:,1],label = 'Node 2')
plt.plot(time,Lambdahist[:,2],label = 'Node 3')
plt.plot(time,Lambdahist[:,3],label = 'Node 4')
plt.plot(time,Lambdahist[:,4],label = 'Node 5')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("Multiplier")
plt.title("Multipliers as function of Time")

plt.figure(9)
plt.plot(time,ghist[:,0],label = 'Node 1')
plt.plot(time,ghist[:,1],label = 'Node 2')
plt.plot(time,ghist[:,2],label = 'Node 3')
plt.plot(time,ghist[:,3],label = 'Node 4')
plt.plot(time,ghist[:,4],label = 'Node 5')
plt.legend()
plt.xlabel("time (s)")
plt.ylabel("gap (m)")
plt.title("Gap as function of Time")
