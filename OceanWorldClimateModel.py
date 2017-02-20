# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:05:47 2016

@author: markprosser
"""

from IPython import get_ipython
get_ipython().magic('reset -f')
import sys
#sys.path.append('PythFunctions')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import time
import datetime
import math
from datetime import timedelta
#from Function import show_plot
plt.close("all")

def show_plot(figure_id=None):
    import matplotlib.pyplot as plt
    if figure_id is None:
        fig = plt.gcf()
    else:
        # do this even if figure_id == 0
        fig = plt.figure(num=figure_id)

    plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()



opt1 = 3
# 2=2D
# 3=3D

opt2 = 3
# 1=Oceandiffusion only
# 2=Atmsopherediffusion only
# 3=both

#PoleFrac=0.99
SC = 1361
ALB = 0.3
LATRES = 30
LONGRES = 60
OIT = -273.15 #initial temp of water degC
AIT = -273.15
R = 6371000 #m
U = 90/LATRES #number of cells vertically for just 1/4
SB = 5.67E-8
#AA = 0.7814
AA = 0.7814
poleTi = 0
Ts = 3600
NTs = 365*24*1.5
KX = 800000 #diffusion constant in X #90000
KY = 800000#2500000 #diffusion constant in Y
kick=0#9999#99999
OD=1


xticks= np.arange(-180,181,LONGRES)
yticks= np.arange(-90,91,LATRES)
LATMAT = np.empty((int(180/LATRES),int(360/LONGRES)))
LONGMAT = np.empty((int(180/LATRES),int(360/LONGRES)))
LATDX = np.empty((int(180/LATRES),int(360/LONGRES)))
LONGDX = np.empty((int(180/LATRES),int(360/LONGRES)))
ALBMAT = np.empty((int(180/LATRES),int(360/LONGRES)))
STABILITY = np.empty((int(180/LATRES),int(360/LONGRES))) #for stability analysis
STABILITY[:]=np.nan
SR = np.empty((int(180/LATRES),int(360/LONGRES)))
S2E = np.empty((int(180/LATRES),int(360/LONGRES)))
A2E = np.empty((int(180/LATRES),int(360/LONGRES)))
E2A = np.empty((int(180/LATRES),int(360/LONGRES)))
E2S = np.empty((int(180/LATRES),int(360/LONGRES)))
A2S = np.empty((int(180/LATRES),int(360/LONGRES)))


SIMAT = np.empty((int(180/LATRES),int(360/LONGRES))) #Solar Insolation
SIALBMAT = np.empty((int(180/LATRES),int(360/LONGRES))) #Solar Insolation * (1-AlBEDO)
PERCMAT = np.empty((int(180/LATRES),int(360/LONGRES))) #Solar Insolation Percentage


for i in range(0,len(LATMAT[0])):
    LATMAT[:,i]=np.arange(90-LATRES/2,-90+LATRES/2-1,-LATRES)
    LATMAT
    
for i in range(0,len(LONGMAT)):
    LONGMAT[i,:]=np.arange(-180+LONGRES/2,180-LONGRES/2+1,LONGRES)
    LONGMAT
    
for i in range(0,len(LATDX[0])):
    LATDX[:,i]=(R*np.pi)/int(180/LATRES)
    
    
for i in range(0,len(LONGDX)):
    LONGDX[i,:]=((R*np.cos(np.deg2rad(LATMAT[i,0])))*2*np.pi)/int(360/LONGRES)
    

STABILITY=(0.5*np.minimum(LONGDX*LONGDX,LATDX*LATDX))/Ts

ALBMAT[:] = ALB


###############################################################################       
#calculate the area of a grid cell
#using HP 50g GLOBEARE prog methodology
###############################################################################   
    
A=np.empty((int(180/LATRES),1))
B=np.empty((int(180/LATRES),1))
C=np.empty((int(180/LATRES),1))
D=np.empty((int(180/LATRES),1))
E=np.empty((int(180/LATRES),1))
 
for i in range(0,int(U*2)):   
    A[i,0]=(i*LATRES)-90 #lower limit, further S, more negative
    B[i,0]=((i+1)*LATRES)-90; #upper limit, further N, more positive
    C[i,0]=(2*np.pi*(R**2))-(2*np.pi*(R**2)*(np.sin(np.deg2rad(B[i,0]))))
    D[i,0]=(2*np.pi*(R**2))-(2*np.pi*(R**2)*(np.sin(np.deg2rad(A[i,0]))))

E=D-C

AreaA=np.empty((int(180/LATRES),int(360/LONGRES)))
###############################################################################   

for i in range(0,int(360/LONGRES)):
    AreaA[:,i]=np.reshape((E/(360/LONGRES)),(len(LATMAT),)) #(Actual) area in m^2 per grid cell
    
del A,B,C,D,E

AreaAReShaped=AreaA.reshape(SIMAT.shape[0],SIMAT.shape[1],1) #reshaped to aid with a 3D array division later
    
#FOR THE FOLLOWING sum(sum(FOLLOWING)) to get global (as opposed to cell values)
FracAreaA = AreaA/(sum(sum(AreaA))) #sum of this MAT should be 1
MassAtmA = FracAreaA*5.14E18
VolumeA = AreaA*OD #*m depth to get m^3

abc4=np.sum(FracAreaA)

MgrA = VolumeA*1000000 #get mass (gr) of water per cell
OceJoulINI = MgrA*4.186*(OIT+273.15) #initial joules per cell %OIT=degC

if opt1 ==3:
    OceJoulTOT1_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1)) #evolving joules per cell PRE diffusion
    OceJoulTOT1_3D[:] = np.nan
    OceJoulTOT2_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1)) #evolving joules per cell POST diffusion
    OceJoulTOT2_3D[:] = np.nan
    OceJoulPAreaTOT2_3D = OceJoulTOT2_3D #ocean joules per unit area
    OceJoulTOT2_3D_PREDIFF = OceJoulTOT2_3D #just for checking with the diffusion with HP50g
    OceJoulTOT1_3D[:,:,0] = OceJoulINI #evolving joules per cell PRE diffusion
    OceJoulTOT2_3D[:,:,0] = OceJoulINI #evolving joules per cell POST diffusion

OceTempINI = (OceJoulINI/(4.186*MgrA))-273.15 #initial temp (degC)

if opt1 == 3:
    OceTempTOT1_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1)) #evolving temp (degC)
    OceTempTOT1_3D[:] = np.nan
    OceTempTOT2_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1)) #evolving temp (degC)
    OceTempTOT2_3D[:] = np.nan
    OceTempTOT1_3D[:,:,0] = OceTempINI #evolving temp (degC)
    OceTempTOT2_3D[:,:,0] = OceTempINI #evolving temp (degC)

AtmJoulINI = (AIT+273.15)*1004*MassAtmA

if opt1 == 3:
    AtmJoulTOT1_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1))
    AtmJoulTOT1_3D[:] = np.nan
    AtmJoulTOT2_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1))
    AtmJoulTOT2_3D[:] = np.nan
    AtmJoulPAreaTOT2_3D = AtmJoulTOT2_3D #atm joules per unit area
    AtmJoulTOT2_3D_PREDIFF = AtmJoulTOT2_3D #just for checking with the diffusion with HP50g
    AtmJoulTOT1_3D[:,:,0] = AtmJoulINI
    AtmJoulTOT2_3D[:,:,0] = AtmJoulINI

AtmTempINI = (AtmJoulINI/(MassAtmA*1004))-273.15

if opt1 == 3:
    AtmTempTOT1_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1))
    AtmTempTOT1_3D[:] = np.nan
    AtmTempTOT2_3D = np.empty((int(180/LATRES),int(360/LONGRES),NTs+1))
    AtmTempTOT2_3D[:] = np.nan
    AtmTempTOT1_3D[:,:,0] = AtmTempINI
    AtmTempTOT2_3D[:,:,0] = AtmTempINI


    
month = 9  
monthlist = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
monthstr = (monthlist[month-1])

MydateLON = datetime.datetime(2001, month, 21, 12, 0, 0) #maxNH %92days.*24

tend=int(NTs)    

AvSI=np.empty((tend,1)) #to get the average SI across the planet
OceAtmTempEvo=np.empty((tend,4))
OceAtmTempEvo[:]=np.nan

SIMATav=np.empty((100000,1))
SIMATav[:]=np.nan

a=0

###############################################################################   
############# MAIN TIME LOOP BEGINS ###########################################
###############################################################################   

for t in range(0,(tend)):
    print(t)
    
    
    MydateLON = MydateLON + timedelta(hours=1)
         
    print(MydateLON)
    for j in range(0,len(SIMAT[0])):    
        for i in range(0,len(SIMAT)):
        
            LAT = LATMAT[i,j]
                
            LONG = LONGMAT[i,j]
            TIMEadj = LONG/180*12*3600
            MydateLOC=MydateLON + datetime.timedelta(0,TIMEadj) #datetimeobj
            MydateLOC = MydateLOC.timetuple() #structdateobj
        
            Tsec=((MydateLOC[3]*3600) + MydateLOC[4]*60 + MydateLOC[5]) - (12*3600);
            LAT = math.radians(LAT)
            LONG = math.radians(LONG)
    
            DJ = MydateLOC[7]
            #DL FAM p318 eq 9.7
            if MydateLOC[0] >= 2001:
                DL = (MydateLOC[0] - 2001)/4
            else:
                DL = ((MydateLOC[0] - 2000)/4) - 1
    
            NJD = 364.5 + ((MydateLOC[0]-2001)*365) + DJ + DL
    
            GM = 357.528 + 0.9856003*NJD; #DEG
            LM = 280.460 + 0.9856474*NJD; #DEG
            LAMec = LM + 1.915*math.sin(math.radians(GM)) + 0.020*math.sin(math.radians(2*GM)) #in degrees?
            EPSob = 23.439 - 0.0000004*NJD #DEG
            DELTA = math.degrees(math.asin(math.sin(math.radians(EPSob))*math.sin(math.radians(LAMec)))) #Solar Declination Angle (DEG)
            Ha = math.degrees((2*math.pi*Tsec)/86400) #DEG
            THETAs = math.degrees(math.acos(math.sin(LAT)*math.sin(math.radians(DELTA)) + math.cos(LAT)*math.cos(math.radians(DELTA))*math.cos(math.radians(Ha)))) #Solar Zenith Angle (DEG)
    
    
            if math.cos(math.radians(THETAs)) < 0:
                INSOL = 0
            else:
                INSOL = SC*math.cos(math.radians(THETAs)) #INSOL calculated!!
    
            SIMAT[i,j]=INSOL
            SIMATav[t,0]=np.mean(SIMAT)
            SIALBMAT[i,j]=SIMAT[i,j]*(1-ALBMAT[i,j])
            PERCMAT[i,j]=INSOL/SC*100
            
    AvSI[t,0]=np.mean(SIMAT)  

    if opt1 == 3:
        OceAtmTempEvo[t,0]=np.mean(OceTempTOT1_3D[:,:,t]) #ocean temp without diff
        OceAtmTempEvo[t,1]=np.mean(OceTempTOT2_3D[:,:,t]) #ocean temp + DIFF
        OceAtmTempEvo[t,2]=np.mean(AtmTempTOT1_3D[:,:,t]) #atm temp without diff
        OceAtmTempEvo[t,3]=np.mean(AtmTempTOT2_3D[:,:,t]) #atm temp + DIFF
        
        AreaProp=np.sum(AreaA, axis=1)/np.sum(AreaA) #latitude band as a proportion of total area
        AreaProp=AreaProp.reshape(len(AreaProp),1)
        
        

###############################################################################      
####### phase 1 add/subtract all the fluxes ###################################
###############################################################################    
        
    #these are all fluxes per second
    S2E = SIALBMAT*(AreaA)

    if opt1 == 3:
        A2E = AreaA*SB*AA*((AtmTempTOT1_3D[:,:,t]+273.15)**4)*0.5
        SR = AreaA*SB*((OceTempTOT1_3D[:,:,t]+273.15)**4)
    E2A = SR*AA
    E2S = SR*(1-AA)
    A2S = A2E
    

    if opt1 == 3:
        OceJoulTOT1_3D[:,:,t+1] = OceJoulTOT1_3D[:,:,t] + S2E*Ts + A2E*Ts - E2S*Ts - E2A*Ts
        OceTempTOT1_3D[:,:,t+1] = (OceJoulTOT1_3D[:,:,t]/(4.186*MgrA))-273.15
        AtmJoulTOT1_3D[:,:,t+1] = AtmJoulTOT1_3D[:,:,t] + E2A*Ts - A2E*Ts - A2S*Ts
        AtmTempTOT1_3D[:,:,t+1] = (AtmJoulTOT1_3D[:,:,t]/(MassAtmA*1004))-273.15


    if opt1 == 3:
        OceJoulTOT2_3D[:,:,t+1] = OceJoulTOT2_3D[:,:,t] + S2E*Ts + A2E*Ts - E2S*Ts - E2A*Ts
        OceJoulTOT2_3D_PREDIFF[:,:,t+1] = OceJoulTOT2_3D[:,:,t+1]
        AtmJoulTOT2_3D[:,:,t+1] = AtmJoulTOT2_3D[:,:,t] + E2A*Ts - A2E*Ts - A2S*Ts   
        AtmJoulTOT2_3D_PREDIFF[:,:,t+1] = AtmJoulTOT2_3D[:,:,t+1]   
       
  
        OceJoulPAreaTOT2_3D[:,:,t+1] = OceJoulTOT2_3D[:,:,t+1]/AreaA#ReShaped #ocean joules per m^2
        AtmJoulPAreaTOT2_3D[:,:,t+1] = AtmJoulTOT2_3D[:,:,t+1]/AreaA#ReShaped #atm joules per m^2
       
    
 
    if (opt2 == 1) or (opt2 == 3):    
###############################################################################           
###### phase 2a include diffusion in Ocean ####################################
###############################################################################       
    
        if opt1 == 3:
            OceJoulTempEneDens=np.empty((int(180/LATRES),int(360/LONGRES)))
            OceJoulTempEneDens[:,:]=np.nan
            for j in range(0,len(SIMAT[0])):  
                for i in range(0,len(SIMAT)):
                    if i==0: #TOP ROW
                        OceJoulTempEneDens[i,j]=(((OceJoulPAreaTOT2_3D[i,int(j+len(LATMAT[0])/2)%len(LATMAT[0]),t+1]-2*OceJoulPAreaTOT2_3D[i,j,t+1]+OceJoulPAreaTOT2_3D[i+1,j,t+1])*KY)/(LATDX[i,j]**2)\
                                    + ((OceJoulPAreaTOT2_3D[i,j-1,t+1]-2*OceJoulPAreaTOT2_3D[i,j,t+1]+OceJoulPAreaTOT2_3D[i,(j+1)%len(LATMAT[0]),t+1])*KX)/(LONGDX[i,j]**2))*Ts\
                                    + OceJoulPAreaTOT2_3D[i,j,t+1]
                    
                    elif i==len(LATMAT)-1: #BOTTOM ROW
                        OceJoulTempEneDens[i,j]=(((OceJoulPAreaTOT2_3D[i-1,j,t+1]-2*OceJoulPAreaTOT2_3D[i,j,t+1]+OceJoulPAreaTOT2_3D[i,int(j+len(LATMAT[0])/2)%len(LATMAT[0]),t+1])*KY)/(LATDX[i,j]**2)\
                                    + ((OceJoulPAreaTOT2_3D[i,j-1,t+1]-2*OceJoulPAreaTOT2_3D[i,j,t+1]+OceJoulPAreaTOT2_3D[i,(j+1)%len(LATMAT[0]),t+1])*KX)/(LONGDX[i,j]**2))*Ts\
                                    + OceJoulPAreaTOT2_3D[i,j,t+1]

                    else: #non bottom non top rows
                        OceJoulTempEneDens[i,j]=(((OceJoulPAreaTOT2_3D[i-1,j,t+1]-2*OceJoulPAreaTOT2_3D[i,j,t+1]+OceJoulPAreaTOT2_3D[i+1,j,t+1])*KY)/(LATDX[i,j]**2)\
                                    + ((OceJoulPAreaTOT2_3D[i,j-1,t+1]-2*OceJoulPAreaTOT2_3D[i,j,t+1]+OceJoulPAreaTOT2_3D[i,(j+1)%len(LATMAT[0]),t+1])*KX)/(LONGDX[i,j]**2))*Ts\
                                    + OceJoulPAreaTOT2_3D[i,j,t+1]
            OceJoulPAreaTOT2_3D[:,:,t+1] = OceJoulTempEneDens #transfers back
            OceJoulTOT2_3D[:,:,t+1] = OceJoulPAreaTOT2_3D[:,:,t+1]*AreaA #converting back from j/m^2 to j
            OceTempTOT2_3D[:,:,t+1] = (OceJoulTOT2_3D[:,:,t+1]/(4.186*MgrA))-273.15


    if (opt2 == 2) or (opt2 == 3):

###############################################################################           
###### phase 2b include diffusion in Atmosphere ############################### 
###############################################################################       
    
        if opt1 == 3:
            AtmJoulTempEneDens=np.empty((int(180/LATRES),int(360/LONGRES)))
            AtmJoulTempEneDens[:,:]=np.nan
            for j in range(0,len(SIMAT[0])):  
                for i in range(0,len(SIMAT)):
                    if i==0: #TOP ROW
                        AtmJoulTempEneDens[i,j]=(((AtmJoulPAreaTOT2_3D[i,int(j+len(LATMAT[0])/2)%len(LATMAT[0]),t+1]-2*AtmJoulPAreaTOT2_3D[i,j,t+1]+AtmJoulPAreaTOT2_3D[i+1,j,t+1])*KY)/(LATDX[i,j]**2)\
                                    + ((AtmJoulPAreaTOT2_3D[i,j-1,t+1]-2*AtmJoulPAreaTOT2_3D[i,j,t+1]+AtmJoulPAreaTOT2_3D[i,(j+1)%len(LATMAT[0]),t+1])*KX)/(LONGDX[i,j]**2))*Ts\
                                    + AtmJoulPAreaTOT2_3D[i,j,t+1]

                    elif i==len(LATMAT)-1: #BOTTOM ROW
                        AtmJoulTempEneDens[i,j]=(((AtmJoulPAreaTOT2_3D[i-1,j,t+1]-2*AtmJoulPAreaTOT2_3D[i,j,t+1]+AtmJoulPAreaTOT2_3D[i,int(j+len(LATMAT[0])/2)%len(LATMAT[0]),t+1])*KY)/(LATDX[i,j]**2)\
                                    + ((AtmJoulPAreaTOT2_3D[i,j-1,t+1]-2*AtmJoulPAreaTOT2_3D[i,j,t+1]+AtmJoulPAreaTOT2_3D[i,(j+1)%len(LATMAT[0]),t+1])*KX)/(LONGDX[i,j]**2))*Ts\
                                    + AtmJoulPAreaTOT2_3D[i,j,t+1]

                    else: #non bottom non top rows
                        AtmJoulTempEneDens[i,j]=(((AtmJoulPAreaTOT2_3D[i-1,j,t+1]-2*AtmJoulPAreaTOT2_3D[i,j,t+1]+AtmJoulPAreaTOT2_3D[i+1,j,t+1])*KY)/(LATDX[i,j]**2)\
                                    + ((AtmJoulPAreaTOT2_3D[i,j-1,t+1]-2*AtmJoulPAreaTOT2_3D[i,j,t+1]+AtmJoulPAreaTOT2_3D[i,(j+1)%len(LATMAT[0]),t+1])*KX)/(LONGDX[i,j]**2))*Ts\
                                    + AtmJoulPAreaTOT2_3D[i,j,t+1]
            AtmJoulPAreaTOT2_3D[:,:,t+1] = AtmJoulTempEneDens #transfers back
            AtmJoulTOT2_3D[:,:,t+1] = AtmJoulPAreaTOT2_3D[:,:,t+1]*AreaA #converting back from j/m^2 to j
            AtmTempTOT2_3D[:,:,t+1] = (AtmJoulTOT2_3D[:,:,t+1]/(1004*MassAtmA))-273.15
            

##### end #####



        
    if t%(24*30)==0:
        month=month+1
        if month>12:
            month=month%12
        
        plt.clf()

        monthstr = (monthlist[month-1])

        
        plt.figure(10,figsize=(15,10))
        plt.subplot(3, 2, 1)
        plt.text(0.05,18.5,'A',fontsize=12)

        if opt1 == 3:  
            plt.pcolor(np.flipud(OceTempTOT1_3D[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("No diffusion ocean-" + str(monthstr) + "-" + str(t))
        plt.ylabel('Degrees Latitude')
        
        
        plt.subplot(3, 2, 2)
        plt.ylabel('Degrees Latitude')
        plt.text(0.05,18.5,'D',fontsize=12)

        if opt1 == 3:  
            plt.pcolor(np.flipud(OceTempTOT2_3D[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        plt.show()
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("Diffusion ocean-" + str(monthstr) + "-" + str(t))
        
        
        plt.subplot(3, 2, 3)
        plt.text(0.05,18.5,'B',fontsize=12)
        plt.ylabel('Degrees Latitude')

        if opt1 == 3:  
            plt.pcolor(np.flipud(AtmTempTOT1_3D[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("No diffusion atmosphere-" + str(monthstr) + "-" + str(t))
        
        
        plt.subplot(3, 2, 4)
        plt.ylabel('Degrees Latitude')
        plt.text(0.05,18.5,'E',fontsize=12)

        if opt1 == 3:  
            plt.pcolor(np.flipud(AtmTempTOT2_3D[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        plt.show()
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("Diffusion atmosphere-" + str(monthstr) + "-" + str(t))
        
        
        plt.subplot(3, 2, 5)
        plt.ylabel('Degrees Latitude')
        plt.xlabel('Temperature (Degrees C)')

        if opt1 == 3:  
            LatMeanOX1=OceTempTOT1_3D[:,:,t]
            LatMeanOX1=np.mean(LatMeanOX1, axis=1)
            LatMeanAX1=AtmTempTOT1_3D[:,:,t]
            LatMeanAX1=np.mean(LatMeanAX1, axis=1)
            LatMeanY1 = np.arange(LATMAT[-1,0],LATMAT[0,0]+(LATMAT[0,0]-LATMAT[1,0]),180/LATMAT.shape[0])
            LatMeanY1 = np.flipud(LatMeanY1)
            plt.plot(LatMeanOX1,LatMeanY1,'b',label='ocean')
            plt.plot(LatMeanAX1,LatMeanY1,'r',label='atmos')
            plt.legend(loc='upper right',prop={'size':10})
        plt.clim(-273.15, 273.15)
        plt.title("C No diffusion lat-temp-profile-" + str(monthstr) + "-" + str(t))
        
        
        plt.subplot(3, 2, 6)
        plt.ylabel('Degrees Latitude')
        plt.xlabel('Degrees C')

        if opt1 == 3:
            LatMeanOX2=OceTempTOT2_3D[:,:,t]
            LatMeanOX2=np.mean(LatMeanOX2, axis=1)
            LatMeanAX2=AtmTempTOT2_3D[:,:,t]
            LatMeanAX2=np.mean(LatMeanAX2, axis=1)
            LatMeanY2 = np.arange(LATMAT[-1,0],LATMAT[0,0]+(LATMAT[0,0]-LATMAT[1,0]),180/LATMAT.shape[0])
            LatMeanY2 = np.flipud(LatMeanY2)
            plt.plot(LatMeanOX2,LatMeanY2,'b',label='ocean')
            plt.plot(LatMeanAX2,LatMeanY2,'r',label='atmos')
            plt.legend(loc='upper right',prop={'size':10})
        plt.show()
        plt.clim(-273.15, 273.15)
        plt.title("F diffusion lat-temp-profile-" + str(monthstr) + "-" + str(t))
        
        
        plt.pause(0.1)
        show_plot()


OceAtmTempEvo2=np.empty((OceJoulTOT1_3D.shape[2],4))
OceAtmTempEvo2[:]=np.nan
ONDtemp=np.mean(OceTempTOT1_3D, axis=1)
ONDtemp=ONDtemp*AreaProp
ONDtemp=np.sum(ONDtemp, axis=0)
OceAtmTempEvo2[:,0]=ONDtemp

OYDtemp=np.mean(OceTempTOT2_3D, axis=1)
OYDtemp=OYDtemp*AreaProp
OYDtemp=np.sum(OYDtemp, axis=0)
OceAtmTempEvo2[:,1]=OYDtemp

ANDtemp=np.mean(AtmTempTOT1_3D, axis=1)
ANDtemp=ANDtemp*AreaProp
ANDtemp=np.sum(ANDtemp, axis=0)
OceAtmTempEvo2[:,2]=ANDtemp

AYDtemp=np.mean(AtmTempTOT2_3D, axis=1)
AYDtemp=AYDtemp*AreaProp
AYDtemp=np.sum(AYDtemp, axis=0)
OceAtmTempEvo2[:,3]=AYDtemp

print(OceAtmTempEvo[tend-1,:])
print(OceAtmTempEvo2[tend-1,:]) #This 2nd row gives you the more realistic average temp as weights the boxes according to area


plt.figure(2)
plt.title('Model ocean and atmosphere temperature evolution over time')
x= np.arange('2001-09-21T12:00:00.0', MydateLON, dtype='datetime64[h]')

y1=OceAtmTempEvo[:,0]
y2=OceAtmTempEvo[:,1]
y3=OceAtmTempEvo[:,2]
y4=OceAtmTempEvo[:,3]

y5=OceAtmTempEvo2[0:-1,0]
y6=OceAtmTempEvo2[0:-1,1]
y7=OceAtmTempEvo2[0:-1,2]
y8=OceAtmTempEvo2[0:-1,3]


plt.plot(x,y5,'b',linewidth=1.0,label='oceanNODIFF')
plt.plot(x,y6,'b--',linewidth=1.0,label='oceanDIFF')
plt.plot(x,y7,'r',linewidth=1.0,label='atmosNODIFF')
plt.plot(x,y8,'r--',linewidth=1.0,label='atmosDIFF')
plt.legend(loc='lower right',prop={'size':14})
plt.ylabel('Temperature (Degrees C)')
plt.xlabel('Time')
plt.show()
show_plot() #this is your function mcp!
plt.close(1)
