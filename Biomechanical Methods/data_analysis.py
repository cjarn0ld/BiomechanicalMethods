#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import math
from scipy.signal import find_peaks
from scipy import signal
from PIL import Image



# In[ ]:





# In[9]:


L_Anklex, L_Ankley, L_Anklez, L_Ankle_Medx, L_Ankle_Medy, L_Ankle_Medz, L_ASISx, L_ASISy, L_ASISz, L_Heelx, L_Heely, L_Heelz, L_Kneex, L_Kneey, L_Kneez, L_Knee_Medx, L_Knee_Medy, L_Knee_Medz, L_Shankx, L_Shanky, L_Shankz, L_Thighx, L_Thighy, L_Thighz, L_Toex, L_Toey, L_Toez, R_Anklex, R_Ankley, R_Anklez, R_Ankle_Medx, R_Ankle_Medy, R_Ankle_Medz, R_ASISx, R_ASISy, R_ASISz, R_Heelx, R_Heely, R_Heelz, R_Kneex, R_Kneey, R_Kneez, R_Knee_Medx, R_Knee_Medy, R_Knee_Medz, R_Shankx, R_Shanky, R_Shankz, R_Thighx, R_Thighy, R_Thighz, R_Toex, R_Toey, R_Toez, V_Sacralx, V_Sacraly, V_Sacralz, a, b = np.loadtxt("static.txt",skiprows = 2, unpack = True)
tempnames, coord= np.genfromtxt("static.txt",max_rows = 2,dtype = str)
tempnames = np.char.array([tempnames])
coord = np.char.array([coord])
names = tempnames + coord

names = np.array(names[0])


# In[10]:


L_Anklex1, L_Ankley1, L_Anklez1, L_Ankle_Medx1, L_Ankle_Medy1, L_Ankle_Medz1, L_ASISx1, L_ASISy1, L_ASISz1, L_Heelx1, L_Heely1, L_Heelz1, L_Kneex1, L_Kneey1, L_Kneez1, L_Knee_Medx1, L_Knee_Medy1, L_Knee_Medz1, L_Shankx1, L_Shanky1, L_Shankz1, L_Thighx1, L_Thighy1, L_Thighz1, L_Toex1, L_Toey1, L_Toez1, R_Anklex1, R_Ankley1, R_Anklez1, R_Ankle_Medx1, R_Ankle_Medy1, R_Ankle_Medz1, R_ASISx1, R_ASISy1, R_ASISz1, R_Heelx1, R_Heely1, R_Heelz1, R_Kneex1, R_Kneey1, R_Kneez1, R_Knee_Medx1, R_Knee_Medy1, R_Knee_Medz1, R_Shankx1, R_Shanky1, R_Shankz1, R_Thighx1, R_Thighy1, R_Thighz1, R_Toex1, R_Toey1, R_Toez1, V_Sacralx1, V_Sacraly1, V_Sacralz1, a1, b1 = np.loadtxt("Walk.txt",skiprows = 2, unpack = True)
tempnames1, coord1= np.genfromtxt("Walk.txt",max_rows = 2,dtype = str)
tempnames1 = np.char.array([tempnames1])
coord1 = np.char.array([coord1])
names1 = tempnames1 + coord1

names1 = np.array(names1[0])


# In[11]:


#af,F2x, F2y, F2z, F3x, F3y, F3z, COFP2x, COFP2y, COFP2z, COFP3x, COFP3y, COFP3z, FREEMOMENT2x, FREEMOMENT2y, FREEMOMENT2z, FREEMOMENT3x, FREEMOMENT3y, FREEMOMENT3z
data = np.loadtxt("force.txt", skiprows = 4, unpack = True)
y = signal.decimate(data, 4)
af,F2x, F2y, F2z, F3x, F3y, F3z, COFP2x, COFP2y, COFP2z, COFP3x, COFP3y, COFP3z, FREEMOMENT2x, FREEMOMENT2y, FREEMOMENT2z, FREEMOMENT3x, FREEMOMENT3y, FREEMOMENT3z = y


# In[12]:


print(af)


# In[13]:


RF, LF, RF...IMPORT HELP PLEASE 
y=improt
downsample
a,b,c = downsample


# In[14]:


R_ASIS_N = np.array([R_ASISx, R_ASISy, R_ASISz])
L_ASIS_N = np.array([L_ASISx, L_ASISy, L_ASISz])
V_SACRUM_N = np.array([V_Sacralx, V_Sacraly, V_Sacralz])

l_thigh = np.array([L_Thighx, L_Thighy, L_Thighz]) 
l_thigh1 = np.array([L_Thighx1, L_Thighy1, L_Thighz1]) 
r_thigh = np.array([R_Thighx, R_Thighy, R_Thighz]) 
r_thigh1 = np.array([R_Thighx1, R_Thighy1, R_Thighz1]) 

r_knee = np.array([R_Kneex, R_Kneey, R_Kneez])
r_knee1 = np.array([R_Kneex1, R_Kneey1, R_Kneez1])
l_knee = np.array([L_Kneex, L_Kneey, L_Kneez])
l_knee1 = np.array([L_Kneex1, L_Kneey1, L_Kneez1])

l_shank = np.array([L_Shankx, L_Shanky, L_Shankz])
l_shank1 = np.array([L_Shankx1, L_Shanky1, L_Shankz1])
r_shank = np.array([R_Shankx, R_Shanky, R_Shankz])
r_shank1 = np.array([R_Shankx1, R_Shanky1, R_Shankz1])


r_ank = np.array([R_Anklex, R_Ankley, R_Anklez]) 
r_ank1 = np.array([R_Anklex1, R_Ankley1, R_Anklez1]) 
l_ank = np.array([L_Anklex, L_Ankley, L_Anklez]) 
l_ank1 = np.array([L_Anklex1, L_Ankley1, L_Anklez1]) 

r_med_ank = np.array([R_Ankle_Medx, R_Ankle_Medy, R_Ankle_Medz])
r_med_ank1 = np.array([R_Ankle_Medx1, R_Ankle_Medy1, R_Ankle_Medz1])
l_med_ank = np.array([L_Ankle_Medx, L_Ankle_Medy, L_Ankle_Medz])
l_med_ank1 = np.array([L_Ankle_Medx1, L_Ankle_Medy1, L_Ankle_Medz1])

r_heel = np.array([R_Heelx, R_Heely, R_Heelz])
r_heel1 = np.array([R_Heelx1, R_Heely1, R_Heelz1])
l_heel = np.array([L_Heelx, L_Heely, L_Heelz])
l_heel1 = np.array([L_Heelx1, L_Heely1, L_Heelz1])

r_toe = np.array([R_Toex, R_Toey, R_Toez])
r_toe1 = np.array([R_Toex1, R_Toey1, R_Toez1])
l_toe = np.array([L_Toex, L_Toey, L_Toez])
l_toe1 = np.array([L_Toex1, L_Toey1, L_Toez1])




# In[15]:


def rot_pelvis(rx, ry, rz, lx, ly, lz, sx, sy, sz):
    r_ASIS = np.array([rx, ry, rz])
    l_ASIS = np.array([lx, ly, lz])

    
    v_3 = (r_ASIS - l_ASIS).T #(3 Vector or v3) 
   
    v3=[]
    v2=[]
    v1=[]
    hip_origin=[]
    rot_matrix1 = [[],[],[]]
    a = len(v_3)
    i=0
    print(a)
    
    while i in range(a):
        R_ASIS = np.array([rx[i], ry[i], rz[i]]).T
        L_ASIS = np.array([lx[i], ly[i], lz[i]]).T
        V_SACRUM = np.array([sx[i], sy[i], sz[i]]).T

        HIP_O = ((R_ASIS + L_ASIS)/2)

        temp1 = (R_ASIS - V_SACRUM)
        temp2 = (L_ASIS - V_SACRUM)

        V3 = (R_ASIS - L_ASIS) #(3 Vector or v3) 
        V1 = np.cross(temp2, temp1)
        V2 = np.cross(V3, V1)
        
        hip_o = np.array(HIP_O)
        hipOmag = np.linalg.norm(hip_o)
        v3mag = np.linalg.norm(V3)
        v1mag = np.linalg.norm(V1)
        v2mag = np.linalg.norm(V2)
        v3 = (1/v3mag*V3)
        v1 = (1/v1mag*V1)
        v2 = (1/v2mag*V2)
        #print(np.shape(v3))
        hip_origin = (1/hipOmag*hip_o)
        R = np.array([v1,v2,v3]).T
        rot_matrix1 = np.append(rot_matrix1,R)
        #rot_matrix = np.append(rot_matrix,R)
        i+=1
    rot_matrix1 = np.reshape(rot_matrix1,(a,3,3))
    
    return rot_matrix1

#loop through normaization 
#y=downsample(x,n) force,4


# In[16]:


pelvis_D = rot_pelvis(R_ASISx1, R_ASISy1, R_ASISz1, L_ASISx1, L_ASISy1, L_ASISz1, V_Sacralx1, V_Sacraly1, V_Sacralz1)
print(pelvis_D)


# In[17]:


#assignment 4


# In[18]:


def hip_centers(rx, ry, rz, lx, ly, lz, sx, sy, sz):
    r_ASIS = np.array([rx, ry, rz])
    l_ASIS = np.array([lx, ly, lz])

    
    v_3 = (r_ASIS - l_ASIS).T #(3 Vector or v3) 
   
    v3=[]
    v2=[]
    v1=[]
    hip_origin=[]
    rot_matrix1 = [[],[],[]]
    a = len(v_3)
    i=0
    print(a)
    P=[]
    
    while i in range(a):
        R_ASIS = np.array([rx[i], ry[i], rz[i]]).T
        L_ASIS = np.array([lx[i], ly[i], lz[i]]).T
        V_SACRUM = np.array([sx[i], sy[i], sz[i]]).T

        HIP_O = ((R_ASIS + L_ASIS)/2)

        temp1 = (R_ASIS - V_SACRUM)
        temp2 = (L_ASIS - V_SACRUM)

        V3 = (R_ASIS - L_ASIS) #(3 Vector or v3) 
        V1 = np.cross(temp2, temp1)
        V2 = np.cross(V3, V1)
        
        hip_o = np.array(HIP_O)
        hipOmag = np.linalg.norm(hip_o)
        v3mag = np.linalg.norm(V3)
        v1mag = np.linalg.norm(V1)
        v2mag = np.linalg.norm(V2)
        v3 = (1/v3mag*V3)
        v1 = (1/v1mag*V1)
        v2 = (1/v2mag*V2)
        #print(np.shape(v3))
        hip_origin = (1/hipOmag*hip_o)
        R = np.array([v1,v2,v3]).T
        v3x, v3y, v3z = v3
        ASISB = np.array([0.24*v3x, -0.21*v3y, 0.32*v3z]).T
        p = hip_o+(R@ASISB)
        P.append(p)
        i+=1
        
    P = np.array(P)
    P = np.reshape(P,(a,3))
    print(np.shape(P))
    #print(rot_pelvis)
    #print(np.shape(v3))
    print(P)
    return P

        


# In[19]:


static = hip_centers(R_ASISx, R_ASISy, R_ASISz, L_ASISx, L_ASISy, L_ASISz, V_Sacralx, V_Sacraly, V_Sacralz)


# In[20]:


dynamic = hip_centers(R_ASISx1, R_ASISy1, R_ASISz1, L_ASISx1, L_ASISy1, L_ASISz1, V_Sacralx1, V_Sacraly1, V_Sacralz1)


# In[21]:


#assignment 5


# In[22]:


def R_rot_matrix(rx, ry, rz, lx, ly, lz, Z):
    lat = np.array([rx, ry, rz])
    med = np.array([lx, ly, lz])
    Z = np.array(Z)
    zx, zy, zz = Z.T  
    v_3 = (lat - med).T #(3 Vector or v3) 
   
    v3=[]
    v2=[]
    v1=[]
    hip_origin=[]
    rot_matrix1 = [[],[],[]]
    a = len(v_3)
    i=0
    print(a)
    
    while i in range(a):
        lat = np.array([rx[i], ry[i], rz[i]]).T
        med = np.array([lx[i], ly[i], lz[i]]).T
        jc = np.array([zx[i], zy[i], zz[i]]).T

        jc1 = ((lat + med)/2)

        temp1 = (lat - jc1).T

        V1 = (jc1 - jc) #(1 Vector or v1) 
        V2 = np.cross(temp1, V1)
        V3 = np.cross(V1, V2)
        
        v3mag = np.linalg.norm(V3)
        v1mag = np.linalg.norm(V1)
        v2mag = np.linalg.norm(V2)
        v3 = (1/v3mag*V3)
        v1 = (1/v1mag*V1)
        v2 = (1/v2mag*V2)

        R = np.array([v1,v2,v3]).T
        rot_matrix1 = np.append(rot_matrix1,R)
        #rot_matrix = np.append(rot_matrix,R)
        i+=1
    rot_matrix1=np.reshape(rot_matrix1,(a,3,3))
    
    return rot_matrix1
def jc(rx, ry, rz, lx, ly, lz):
    jc = []
    for i in range(len(rx)):
        lat = np.array([rx[i], ry[i], rz[i]]).T
        med = np.array([lx[i], ly[i], lz[i]]).T
        jc.append((lat-med)/2)
    return jc


# In[23]:


Rknee_jc = jc(R_Kneex, R_Kneey, R_Kneez, R_Knee_Medx, R_Knee_Medy, R_Knee_Medz)
Rknee_jc1 = jc(R_Kneex1, R_Kneey1, R_Kneez1, R_Knee_Medx1, R_Knee_Medy1, R_Knee_Medz1)

Rthigh_rotS = R_rot_matrix(R_Kneex, R_Kneey, R_Kneez, R_Knee_Medx, R_Knee_Medy, R_Knee_Medz, static)
Rthigh_rotD = R_rot_matrix(R_Kneex1, R_Kneey1, R_Kneez1, R_Knee_Medx1, R_Knee_Medy1, R_Knee_Medz1, dynamic)
print(Rthigh_rotS)
print(" ")
print(Rthigh_rotD)

Rshank_rotS = R_rot_matrix(R_Anklex, R_Ankley, R_Anklez, R_Shankx, R_Shanky, R_Shankz, Rknee_jc)
print(Rshank_rotS)

print("")

Rshank_rotD = R_rot_matrix(R_Anklex1, R_Ankley1, R_Anklez1, R_Shankx1, R_Shanky1, R_Shankz1, Rknee_jc1)
print(Rshank_rotD)


# In[24]:


def L_rot_matrix(rx, ry, rz, lx, ly, lz, Z):
    lat = np.array([rx, ry, rz])
    med = np.array([lx, ly, lz])
    Z = np.array(Z)
    zx, zy, zz = Z.T 
    v_3 = (lat - med).T #(3 Vector or v3) 
   
    v3=[]
    v2=[]
    v1=[]
    hip_origin=[]
    rot_matrix1 = [[],[],[]]
    a = len(v_3)
    i=0
    print(a)
    
    while i in range(a):
        lat = np.array([rx[i], ry[i], rz[i]]).T
        med = np.array([lx[i], ly[i], lz[i]]).T
        jc = np.array([zx[i], zy[i], zz[i]]).T

        jc1 = ((lat + med)/2)

        temp1 = (jc1-lat).T

        V1 = (jc1 - jc) #(1 Vector or v1) 
        V2 = np.cross(temp1, V1)
        V3 = np.cross(V1, V2)
        
        v3mag = np.linalg.norm(V3)
        v1mag = np.linalg.norm(V1)
        v2mag = np.linalg.norm(V2)
        v3 = (1/v3mag*V3)
        v1 = (1/v1mag*V1)
        v2 = (1/v2mag*V2)

        R = np.array([v1,v2,v3]).T
        rot_matrix1 = np.append(rot_matrix1,R)
        #rot_matrix = np.append(rot_matrix,R)
        i+=1
    rot_matrix1=np.reshape(rot_matrix1,(a,3,3))
    
    return rot_matrix1


# In[25]:


Lthigh_rotS = L_rot_matrix(L_Kneex, L_Kneey, L_Kneez, L_Knee_Medx, L_Knee_Medy, L_Knee_Medz, static)
Lthigh_rotD = L_rot_matrix(L_Kneex1, L_Kneey1, L_Kneez1, L_Knee_Medx1, L_Knee_Medy1, L_Knee_Medz1, dynamic)


# In[26]:


Lknee_jc = jc(L_Kneex, L_Kneey, L_Kneez, L_Knee_Medx, L_Knee_Medy, L_Knee_Medz)
Lknee_jc1 = jc(L_Kneex1, L_Kneey1, L_Kneez1, L_Knee_Medx1, L_Knee_Medy1, L_Knee_Medz1)

Lshank_rotS = L_rot_matrix(L_Anklex, L_Ankley, L_Anklez, L_Shankx, L_Shanky, L_Shankz, Lknee_jc)
print(Lshank_rotS)

print("")

Lshank_rotD = L_rot_matrix(L_Anklex1, L_Ankley1, L_Anklez1, L_Shankx1, L_Shanky1, L_Shankz1, Lknee_jc1)
print(Lshank_rotD)


# In[27]:


#assignment 6


# In[28]:


def P_prime(rot, rx, ry, rz, lx, ly, lz):
    p = np.array([rx, ry, rz])
    o = np.array([lx, ly, lz])

    

    P_prime = []

    R = np.array([[rot[0][0][0], rot[0][0][1], rot[0][0][2]], [rot[0][1][0],rot[0][1][1],rot[0][1][2]], [rot[0][2][0],rot[0][2][1],rot[0][2][2]]]).T
    P = np.array([[p[0][0]], [p[1][0]], [p[2][0]]])
    O = np.array([[o[0][0]], [o[1][0]], [o[2][0]]])
    P_prime = R@(P-O)
        
    P_prime = np.array(P_prime)
    
   # P_prime = np.reshape(P_prime, (a,3))

    print(np.shape(P_prime))
    print(P_prime)
    
    return P_prime


# In[29]:


RK_MED_P_PRIME = P_prime(Rthigh_rotS, R_Knee_Medx, R_Knee_Medy, R_Knee_Medz, R_Kneex, R_Kneey, R_Kneez) 


# In[30]:


LKnee_MED_P_PRIME = P_prime(Lthigh_rotS, L_Knee_Medx, L_Knee_Medy, L_Knee_Medz, L_Kneex, L_Kneey, L_Kneez) 


# In[31]:


RAnkle_MED_P_prime = P_prime(Rshank_rotS, R_Ankle_Medx, R_Ankle_Medy, R_Ankle_Medz, R_Anklex, R_Ankley, R_Anklez)


# In[32]:


LAnkle_MED_P_prime = P_prime(Lshank_rotS, L_Ankle_Medx, L_Ankle_Medy, L_Ankle_Medz, L_Anklex, L_Ankley, L_Anklez)


# In[33]:


def P(rot, P_prime, ox, oy, oz):

    o = np.array([ox, oy, oz])    
    PP = P_prime
    print(np.shape(PP))
    
    
    P = []
    a = len(rot)
    print(a)
    
    i=0
    while i in range(a):
        R = np.array([[rot[i][0][0], rot[i][0][1], rot[i][0][2]], [rot[i][1][0],rot[i][1][1],rot[i][1][2]], [rot[i][2][0],rot[i][2][1],rot[i][2][2]]])
        O = np.array([[o[0][i]], [o[1][i]], [o[2][i]]])
        #print(M[0][0][0])
        P.append((R@PP)+O)
        #print(np.shape(P))
        i+=1
       
    P = np.array(P)
    
    P = np.reshape(P, (a,3))

    print(np.shape(P))

    print(P)


# In[34]:


P_RAnkle_Med = P(Rthigh_rotD, RAnkle_MED_P_prime, R_Kneex1, R_Kneey1, R_Kneez1)


# In[35]:


def RFoot(toex, toey, toez, ankx, anky, ankz, ankMx, ankMy, ankMz, heelx, heely, heelz):
    toe = np.array([toex, toey, toez])
    heel = np.array([heelx, heely, heelz])
    
    v_3 = (toe - heel).T #(3 Vector or v3) 
   
    v3=[]
    v2=[]
    v1=[]
    hip_origin=[]
    RFoot = [[],[],[]]
    a = len(v_3)
    i=0
    print(a)
    
    for i in range(a):
        toe = np.array([toex[i], toey[i], toez[i]]).T
        ankL = np.array([ankx[i], anky[i], ankz[i]]).T
        ankM = np.array([ankMx[i], ankMy[i], ankMz[i]]).T
        heel = np.array([heelx[i], heely[i], heelz[i]]).T

        jc = ((ankL + ankM)/2)

        temp1 = (ankL - jc).T

        V1 = (toe - heel) #(1 Vector or v1) 
        V2 = np.cross(temp1, V1)
        V3 = np.cross(V2, V1)
        
        v3mag = np.linalg.norm(V3)
        v1mag = np.linalg.norm(V1)
        v2mag = np.linalg.norm(V2)
        v3 = (1/v3mag*V3)
        v1 = (1/v1mag*V1)
        v2 = (1/v2mag*V2)

        R = np.array([v1,v2,v3]).T
        RFoot = np.append(RFoot,R)
        #rot_matrix = np.append(rot_matrix,R)
        i+=1
    RFoot = np.reshape(RFoot,(a,3,3))
    
    return RFoot


def LFoot(toex, toey, toez, ankx, anky, ankz, ankMx, ankMy, ankMz, heelx, heely, heelz):
    toe = np.array([toex, toey, toez])
    heel = np.array([heelx, heely, heelz])
    
    v_3 = (toe - heel).T #(3 Vector or v3) 
   
    v3=[]
    v2=[]
    v1=[]
    hip_origin=[]
    LFoot = [[],[],[]]
    a = len(v_3)
    i=0
    print(a)
    
    for i in range(a):
        toe = np.array([toex[i], toey[i], toez[i]]).T
        ankL = np.array([ankx[i], anky[i], ankz[i]]).T
        ankM = np.array([ankMx[i], ankMy[i], ankMz[i]]).T
        heel = np.array([heelx[i], heely[i], heelz[i]]).T

        jc = ((ankL + ankM)/2)

        temp1 = (jc - ankL).T

        V1 = (toe - heel) #(1 Vector or v1) 
        V2 = np.cross(temp1, V1)
        V3 = np.cross(V2, V1)
        
        v3mag = np.linalg.norm(V3)
        v1mag = np.linalg.norm(V1)
        v2mag = np.linalg.norm(V2)
        v3 = (1/v3mag*V3)
        v1 = (1/v1mag*V1)
        v2 = (1/v2mag*V2)

        R = np.array([v1,v2,v3]).T
        LFoot = np.append(LFoot,R)
        #rot_matrix = np.append(rot_matrix,R)
        i+=1
    LFoot = np.reshape(LFoot,(a,3,3))
    
    return LFoot


# In[159]:


Rfoot_D = RFoot(R_Toex1, R_Toey1, R_Toez1, R_Anklex1, R_Ankley1, R_Anklez1, R_Ankle_Medx1, R_Ankle_Medy1, R_Ankle_Medz1, R_Heelx1, R_Heely1, R_Heelz1)
print(Rfoot_D)


# In[161]:


Lfoot_D = LFoot(L_Toex1, L_Toey1, L_Toez1, L_Anklex1, L_Ankley1, L_Anklez1, L_Ankle_Medx1, L_Ankle_Medy1, L_Ankle_Medz1, L_Heelx1, L_Heely1, L_Heelz1)
print(Lfoot_D)


# In[162]:


#assignment 7


# In[163]:


pelvis_dynamic
Rthigh_rotD 
Lthigh_rotD
Rshank_rotD
Lshank_rotD
Rfoot_D
Lfoot_D


# In[164]:


pelvis_D = rot_pelvis(R_ASISx1, R_ASISy1, R_ASISz1, L_ASISx1, L_ASISy1, L_ASISz1, V_Sacralx1, V_Sacraly1, V_Sacralz1)


# In[165]:


def joint_angles(a,b):
    print(np.shape(a))
    print(np.shape(b))
    c = len(a)
    C = []
    alpha = []
    beta = []
    gamma = []
    
    i=0
    for i in range(c):
        A = np.array([[a[i][0][0], a[i][0][1], a[i][0][2]], [a[i][1][0],a[i][1][1],a[i][1][2]], [a[i][2][0],a[i][2][1],a[i][2][2]]])
        B = np.array([[b[i][0][0], b[i][0][1], b[i][0][2]], [b[i][1][0],b[i][1][1],b[i][1][2]], [b[i][2][0],b[i][2][1],b[i][2][2]]])
        #print(M[0][0][0])
        C1 = A.T@B
        C1 = np.array(C1)
        #print(A)
        #print(np.shape(C1))
        C.append(C1)
        alpha.append(np.arctan2(C1[1,0],C1[0,0]))
        beta.append(np.arctan2(-C1[2,0], np.sqrt(C1[0,0]**2+C1[1,0]**2)))
        gamma.append(np.arctan2(C1[2,1],C1[2,2]))
        #print(np.shape(d1))
        #print(r)
        #print(" ")
       # i+=1
       
    C = np.array(C)
    
    alpha = np.array(alpha)
    beta = np.array(beta)
    gamma = np.array(gamma)
    
    #C = np.reshape(C, (3,3,c))

    print(np.shape(alpha))

    #print(C)
    return alpha,beta,gamma


# In[166]:


hip_angles = joint_angles(pelvis_D, Rthigh_rotD) 

hipz, hipy, hipx = hip_angles
#print([np.shape(hipx),np.shape(hipy),np.shape(hipz)])
#plt.plot(a1, hip_angles)
plt.figure()
plt.plot(a1, hipy)
plt.title("Right Hip Y angle")
plt.figure()
plt.plot(a1, hipx)
plt.title("Right Hip X angle")
plt.figure()
plt.plot(a1, hipz)
plt.title("Right Hip Z angle")

# In[167]:


hip_angles = joint_angles(pelvis_D, Lthigh_rotD) 

hipz, hipy, hipx = hip_angles
#plt.plot(a1, hip_angles)
plt.figure()
plt.plot(a1, hipy)
plt.title("left Hip Y angle")
plt.figure()
plt.plot(a1, hipx)
plt.title("left Hip X angle")
plt.figure()
plt.plot(a1, hipz)
plt.title("left Hip z angle")

# In[168]:


#assignment 8


# In[169]:


def helical_angles(a,b):
    print(np.shape(a))
    print(np.shape(b))
    c = len(a)
    C = []
    theta = []
    k = []
    
    
    i=0
    while i in range(c):
        A = np.array([[a[i][0][0], a[i][0][1], a[i][0][2]], [a[i][1][0],a[i][1][1],a[i][1][2]], [a[i][2][0],a[i][2][1],a[i][2][2]]])
        B = np.array([[b[i][0][0], b[i][0][1], b[i][0][2]], [b[i][1][0],b[i][1][1],b[i][1][2]], [b[i][2][0],b[i][2][1],b[i][2][2]]])
        #print(M[0][0][0])
        C1 = A.T@B
        C1 = np.array(C1)
        #print(A)
        #print(np.shape(C1))
        C.append(C1)
        theta1 = np.arccos((C1[0,0]+C1[1,1]+C1[2,2]-1)/2)
        theta.append(theta1)
        k.append(1/(2*np.sin(theta1))*np.array([C1[2,1]-C1[1,2], C1[0,2]-C1[2,0], C1[1,0]-C1[0,1]]))   
        #print(np.shape(d1))
        #print(r)
        #print(" ")
        i+=1
       
    C = np.array(C)
    
    theta = np.array(theta)
    K = np.array(k)
    
    #C = np.reshape(C, (3,3,c))


    #print(C)
    return theta, K


# use sanme prox.t and distal rot matrixes and 

# just plot angle a costheta and find K

# In[244]:

plt.figure()
Lthigh_theta, Lthigh_K = helical_angles(pelvis_D, Lthigh_rotD)
print(Lthigh_theta)
print(Lthigh_K)
plt.plot(a1, Lthigh_theta)
plt.title("Left Thigh Helical Angle")

# In[243]:
    

plt.figure()
Rthigh_theta, Rthigh_K = helical_angles(pelvis_D, Rthigh_rotD)
print(Rthigh_theta)
print(Rthigh_K)
plt.plot(a1, Rthigh_theta)
plt.title("Right Thigh Helical Angle")


# In[267]:

plt.figure()
Rshank_theta, Rshank_K = helical_angles(Rthigh_rotD, Rshank_rotD)
print(Rshank_theta)
print(Rshank_K)
plt.plot(a1, Rshank_theta)
plt.title("Right Shank Helical Angle")

plt.figure()
Lshank_theta, Lshank_K = helical_angles(Lthigh_rotD, Lshank_rotD)
print(Lshank_theta)
print(Lshank_K)
plt.plot(a1, Lshank_theta)
plt.title("Left Shank Helical Angle")

# In[269]:
plt.figure()

Rfoot_theta, Rfoot_K = helical_angles(Rshank_rotD, Rfoot_D)
print(Rfoot_theta)
print(Rfoot_K)
plt.plot(a1, Rfoot_theta)
plt.title("Right Foot Helical Angle")

plt.figure()
Lfoot_theta, Lfoot_K = helical_angles(Lshank_rotD, Lfoot_D)
print(Lfoot_theta)
print(Lfoot_K)
plt.plot(a1, Lfoot_theta)
plt.title("Left Foot Helical Angle")
# In[172]:


# assignment 9


# In[173]:


HJC = np.array(dynamic) 
RKJC = np.array(jc(R_Kneex1, R_Kneey1, R_Kneez1, R_Knee_Medx1, R_Knee_Medy1, R_Knee_Medz1))
RAJC = np.array(jc(R_Anklex1, R_Ankley1, R_Anklez1, R_Ankle_Medx1, R_Ankle_Medy1, R_Ankle_Medz1))
Rtoe = np.array([R_Toex1, R_Toey1, R_Toez1]).T
print(np.shape(Rtoe))


# In[174]:


HJC = np.array(dynamic) 
LKJC = np.array(jc(L_Kneex1, L_Kneey1, L_Kneez1, L_Knee_Medx1, L_Knee_Medy1, L_Knee_Medz1))
LAJC = np.array(jc(L_Anklex1, L_Ankley1, L_Anklez1, L_Ankle_Medx1, L_Ankle_Medy1, L_Ankle_Medz1))
Ltoe = np.array([L_Toex1, L_Toey1, L_Toez1]).T
print(np.shape(Ltoe))


# In[175]:


def COM_x(HJC, KJC, AJC, TOE):
    COM_T = np.array((KJC-HJC)*(0.433)+HJC)
    COM_S = np.array((AJC-KJC)*(0.433)+KJC)
    COM_F = np.array((TOE-AJC)*(0.5)+AJC)
    return COM_T, COM_S, COM_F


# In[176]:


i = 0

ank_jcR = (R_lat_ank1 + R_med_ank1)/2

ank_jcL = (L_lat_ank1 + L_med_ank1) 

R_Knee1 
R_Knee_Med1 = np.array([R_Knee_Medx1, R_Knee_Medy1, R_Knee_Medz1]).T
knee_jcR = (R_knee1+R_Knee_Med1)/2

L_Knee1 
L_Knee_Med1 = np.array([L_Knee_Medx1, L_Knee_Medy1, L_Knee_Medz1]).T
knee_jcL = (L_knee1+L_Knee_Med1)/2

R_toe1 = R_toe1.T
L_toe1 = L_toe1.T

print(np.shape(R_ank_jc))


# In[177]:


print(np.shape(HJC))


# In[223]:


R_COM_T, R_COM_S, R_COM_F = COM_x(HJC, RAJC , RKJC, Rtoe) 
print(R_COM_T)


# In[224]:


L_COM_T, L_COM_S, L_COM_F = COM_x(HJC, LAJC , LKJC, Ltoe)


# In[225]:


RCOP = np.array([COFP2x, COFP2y, COFP2z]).T 
LCOP = np.array([COFP3x, COFP3y, COFP3z]).T

def end_vectors(HJC, KJC, AJC, TOE, COM_T, COM_S, COM_F, PLATE_COP):
    V_FD = np.array(PLATE_COP-COM_F)#foot distal
    V_FP = np.array(AJC-COM_F) #foot proximal
    V_SD = np.array(AJC-COM_S) #shank distal
    V_SP = np.array(KJC-COM_S) #shank proximal
    V_TD = np.array(KJC-COM_T) #thigh distal
    V_TP = np.array(HJC-COM_T) #thigh proximal
    
    return V_FD, V_FP, V_SD, V_SP, V_TD, V_TP





# In[ ]:


# In[ ]:




# In[226]:


#F_FD, F_FP, F_SD, F_SP, F_TD, F_TP = end_vectors(HJC, KJC, AJC, TOEJC, COM_T, COM_S, COM_F, PLATE_COP)


# In[227]:


RV_FD, RV_FP, RV_SD, RV_SP, RV_TD, RV_TP = end_vectors(HJC, RAJC , RKJC, Rtoe, R_COM_T, R_COM_S, R_COM_F ,RCOP)
print(RV_FD)


# In[228]:


LV_FD, LV_FP, LV_SD, LV_SP, LV_TD, LV_TP = end_vectors(HJC, RAJC , RKJC, Rtoe, L_COM_T, L_COM_S, L_COM_F, LCOP)
print(LV_FD)


# In[229]:


#assignment 10

def derivative(X, dt):
    X_prime = []
    for i in range(1,len(X)):
        X_prime.append((X[i]-X[i-1])/(2*dt))
    X_prine = np.array(X_prime)
    return X_prime

def force(m,a):
    F = []
    for i in range(a):
        a = a[i]
        F.append(m*a)
    return F


#Right Side Segments
Rthigh_v = derivative(R_COM_T, 300)
Rshank_v = derivative(R_COM_S, 300)
Rfoot_v = derivative(R_COM_F, 300)

Rthigh_a = derivative(Rthigh_v, 300)
Rshank_a = derivative(Rshank_v, 300)
Rfoot_a = derivative(Rfoot_v, 300)
print(Rfoot_v)

#Rthigh_F = force(50,Rthigh_a)
#Left Side Segments
Lthigh_v = derivative(R_COM_T, 300)
Lshank_v = derivative(R_COM_S, 300)
Lfoot_v = derivative(R_COM_F, 300)

Lthigh_a = derivative(Lthigh_v, 300)
Lshank_a = derivative(Lshank_v, 300)
Lfoot_a = derivative(Lfoot_v, 300)
print(Lfoot_v)


# In[230]:


#assignment 11


# In[231]:


def Global_M(P,D,FP,FD): #global moment 
    GM_COM = (P*FP)+(D*FD)#p-proximal vector d-distal vector 
    return GM_COM


# In[232]:


R_F_GM = Global_M(RV_FD, RV_FP, FFP, FFD) #forces are invisible as shown in assignment... 10 :) 
R_S_GM = Global_M(RV_SD, RV_SP, SFP, SFD)
R_T_GM = Global_M(RV_TD, RV_TP, TFP, TFD)

L_F_GM = Global_M(LV_FD, LV_FP, FFP, FFD)
L_S_GM = Global_M(LV_SD, LV_SP, SFP, SFD)
L_T_GM = Global_M(LV_TD, LV_TP, TFP, TFD)


# In[ ]:





# In[233]:


def Ic(Itable):
    Ic = Itable*((87.5*180.3**2)/(73.59*175.5**2))
    return Ic


# In[234]:


I_LTx =Ic(0.0005)
I_LTy =Ic(0.0878)
I_LTz =Ic(0.0878)

I_LTx = Ic(0.0001)
I_LTy = Ic(0.0414)
I_LTz = Ic(0.0414)

I_LFx = Ic(0.0000)
I_LFy = Ic(0.0061)
I_LFz = Ic(0.0061)


# In[235]:


I_RTx =Ic(0.0005)
I_RTy =Ic(0.0878)
I_RTz =Ic(0.0878)

I_RTx = Ic(0.0001)
I_RTy = Ic(0.0414)
I_RTz = Ic(0.0414)

I_RFx = Ic(0.0000)
I_RFy = Ic(0.0061)
I_RFz = Ic(0.0061)


# In[236]:


#assignment 12


# In[239]:


def g2l(r, com): #global to local COM
    local = []
    
    for i in range(len(com)):
        R = r[i]
        COM = com[i]
        local.append(list(R@COM.T))
     local = np.array(local)   
    return local

 



# In[ ]:


R_COM_Thigh = g2l(Rthigh_rotD, R_COM_T)
R_COM_Shank = g2l(Rshank_rotD, R_COM_S)
R_COM_Foot = g2l(Rfoot_D, R_COM_S)

#print(R_COM_Thigh) #print with caution :'(  
print(np.shape(R_COM_Thigh))

# In[240]:


R_T_GM = g2l(Rthigh_rotD, R_T_GM) #momment refrenced in assignment 11 :) 

R_S_GM = g2l(Rshank_rotD, R_S_GM)

R_F_GM = g2l(Rfoot_D, R_F_GM)

#same can be done with left side 


# In[241]:


F_plate_local = g2l(Rfoot_D, F_plate)#assignment 10 :) 


# In[261]:


def derivative(X, dt):
    X_prime = []
    for i in range(1,len(X)):
        X_prime.append((X[i]-X[i-1])/(2*dt))
    return X_prime


# In[266]:


omega_Rthigh = derivative(Rthigh_theta, 300)#idk or remember what dt was or how it changes when doing ang acceleration...
alpha_Rthigh = derivative(omega_Rthigh, 300)
print(np.shape(alpha_Rthigh))

omega_Rshank = derivative(Rshank_theta, 300)#idk or remember what dt was or how it changes when doing ang acceleration...
alpha_Rshank = derivative(omega_Rshank, 300)
print(np.shape(alpha_Rshank))

omega_Rfoot = derivative(Rfoot_theta, 300)#idk or remember what dt was or how it changes when doing ang acceleration...
alpha_Rfoot = derivative(omega_Rfoot, 300)
print(np.shape(alpha_Rfoot))


# In[275]:#assignment 13 :) 
def Mp_1d(Mdx, MCGx, Ix, Iy, Iz, ax, wy, wz):
    Mpx = []
    for i in range(len(MCGx)):
        Mdx = Mdx[i]
        MCGx = MCGx[i]
        ax = ax[i]
        wy = wy[i]
        wz = wz[i]
        Mpx.append(-Mdx-MCGx+Ix*ax-(wy*wz*(Iy-Iz)))
    return Mpx

RThigh_Mpx = Mp_1d(Mdx, MCGx, Ix, Iy, Iz, ax, wy, wz)
RThigh_Mpy = Mp_1d(Mdy, MCGy, Ix, Iy, Iz, ay, wz, wx)
RThigh_Mpz = Mp_1d(Mdz, MCGz, Ix, Iy, Iz, az, wy, wx)    

RShank_Mpx = Mp_1d(Mdx, MCGx, Ix, Iy, Iz, ax, wy, wz)
RShank_Mpy = Mp_1d(Mdy, MCGy, Ix, Iy, Iz, ay, wz, wx)
RShank_Mpz = Mp_1d(Mdz, MCGz, Ix, Iy, Iz, az, wy, wx)

RFoot_Mpx = Mp_1d(Mdx, MCGx, Ix, Iy, Iz, ax, wy, wz)
RFoot_Mpy = Mp_1d(Mdy, MCGy, Ix, Iy, Iz, ay, wz, wx)
RFoot_Mpz = Mp_1d(Mdz, MCGz, Ix, Iy, Iz, az, wy, wx)


# In[310]:
    def power(Mpx,Mpy,Mpz,omega):
        power = []
        for i in range(omega):
            Mp = [Mpx[i], Mpy[i], Mpz[i]]
            w = omega[i]
            power.append(Mp@w)
        power = np.array(power)
        return power
Rthigh_p = power(RThigh_Mpx,RThigh_Mpy,RThigh_Mpz, omega_Rthigh)