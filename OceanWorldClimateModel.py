
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:05:47 2016
@author: markprosser
"""

from IPython import get_ipython
get_ipython().magic('reset -f')  # NOQA
# sys.path.append('PythFunctions')
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import timedelta
# from Function import show_plot
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


opt2 = 3
# 1=Oceandiffusion only
# 2=Atmsopherediffusion only
# 3=both

# PoleFrac=0.99
SOLAR_CONSTANT = 1361
ALBEDO = 0.3
OCEAN_INITIAL_TEMP = -273.15  # initial temp of water degC
ATMOSPHERE_INITIAL_TEMP = -273.15
EARTH_RADIUS_M = 6371000  # m
EARTH_CROSS_AREA = 2 * np.pi * EARTH_RADIUS_M ** 2
EARTH_AREA = 5.10E14 
N_LAT = 5
N_LONG = 3
STEFAN_BOLTZMANN_CONSTANT = 5.67E-8
ATMOSPHERIC_ABSORPTION_COEFFICIENT = 0.7814
WATER_HEAT_CAPACITY = 4.186  # CHECK
DELTA_TIME_SECS = 3600
N_TIME_STEPS = 40 * 24 * 1
DIFFUSION_X_CONSTANT = 800000  # diffusion constant in X #90000
DIFFUSION_Y_CONSTANT = 800000  # 2500000 #diffusion constant in Y
LASER_FROM_SPACE = 0  # 9999#99999
OCEAN_DEPTH_M = 1
START_MONTH = 9
MONTH_LIST = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MY_DATE_LONDON = datetime.datetime(2001, START_MONTH, 21, 12, 0, 0)
# maxNH %92days.*24

# ********************** END OF INITIAL CONDITIONS INPUT **********************
lat_res_deg = 180 / N_LAT
long_res_deg = 360 / N_LONG
month = START_MONTH
month_str = MONTH_LIST[month-1]

xticks = np.arange(-180, 181, long_res_deg)
yticks = np.arange(-90, 91, lat_res_deg)
midcell_lat_mat = np.full((N_LAT, N_LONG), np.nan)
midcell_long_mat = np.full((N_LAT, N_LONG), np.nan)
cell_dy_m_mat = np.full((N_LAT, N_LONG), np.nan)
cell_dx_m_mat = np.full((N_LAT, N_LONG), np.nan)
albedo_mat = np.full((N_LAT, N_LONG), np.nan)
stability = np.full((N_LAT, N_LONG), np.nan)  # for stability analysis
surf_radiation = np.full((N_LAT, N_LONG), np.nan)
space2_ocean_flux = np.full((N_LAT, N_LONG), np.nan)
atmos2_ocean_flux = np.full((N_LAT, N_LONG), np.nan)
ocean2_atmos_flux = np.full((N_LAT, N_LONG), np.nan)
ocean2_space_flux = np.full((N_LAT, N_LONG), np.nan)
atmos2_space_flux = np.full((N_LAT, N_LONG), np.nan)
ocean_cell_area_m2_mat = np.full((N_LAT, N_LONG), np.nan)
# evolving joules per cell PRE diffusion
ocean_cell_joules_prediff_3dmat = np.full((N_LAT, N_LONG, N_TIME_STEPS+1), np.nan)
# evolving joules per cell POST diffusion
ocean_cell_joules_postdiff_3dmat = np.full((N_LAT, N_LONG, N_TIME_STEPS+1), np.nan)

# TODO: remove unnessesary declarations, e.g. toa_solar_insol_mat
# toa_solar_insol_mat = np.full((N_LAT, N_LONG), np.nan) #Solar Insolation
solar_insol = np.full((N_LAT, N_LONG), np.nan)  # Solar Insolation * (1-AlBEDO)
toa_solar_insol_perc = np.full((N_LAT, N_LONG), np.nan)  # Solar Insol Percent

midcell_lat_mat[:] = np.arange(90-lat_res_deg/2,
                               -90+lat_res_deg/2-1,
                               -lat_res_deg)[:, np.newaxis]
midcell_long_mat[:] = np.arange(-180+long_res_deg/2,
                                180-long_res_deg/2+1,
                                long_res_deg)[np.newaxis, :]

cell_dy_m_mat[:] = (EARTH_RADIUS_M * np.pi) / N_LAT

cell_dx_m_mat[:] = (EARTH_RADIUS_M * np.cos(np.deg2rad(midcell_lat_mat))
                    * 2 * np.pi) / N_LONG

stability = 0.5 * np.minimum(cell_dx_m_mat * cell_dx_m_mat,
                             cell_dy_m_mat * cell_dy_m_mat) / DELTA_TIME_SECS

albedo_mat[:] = ALBEDO

# calculate the area of a grid cell
# using HP 50g GLOBEARE prog methodology
A = np.arange(N_LAT) * lat_res_deg - 90
B = np.arange(1, N_LAT+1) * lat_res_deg - 90
C = 0.5 * EARTH_AREA * (1 - np.sin(np.deg2rad(B)))
D = 0.5 * EARTH_AREA * (1 - np.sin(np.deg2rad(A)))
E = D - C

ocean_cell_area_m2_mat[:] = E[:, np.newaxis] / N_LONG

# TODO: remove in the next commit
# del A, B, C, D, E <- not really necessary

# reshape to aid with a 3D array division later
ocean_cell_area_m2_mat_3d = ocean_cell_area_m2_mat[..., np.newaxis]

# FOR THE FOLLOWING sum(sum(FOLLOWING)) to get global
# (as opposed to cell values)
frac_ocean_cell_area_m2_mat = (ocean_cell_area_m2_mat /
                               ocean_cell_area_m2_mat.sum())
assert frac_ocean_cell_area_m2_mat.sum() == 1, 'sum of this ARR should be 1'
atmos_cell_mass_kg_mat = frac_ocean_cell_area_m2_mat * EARTH_AREA
ocean_cell_vol_m3_mat = ocean_cell_area_m2_mat * OCEAN_DEPTH_M  # *m depth m^3
# FIXME: multiply by density? - multiplying by 1 million is effectively doing that
# get mass (gr) of water per cell
ocean_cell_mass_gr_mat = ocean_cell_vol_m3_mat * 1000000
ocean_cell_initjoules_mat = (ocean_cell_mass_gr_mat
                             * WATER_HEAT_CAPACITY
                             * (OCEAN_INITIAL_TEMP + 273.15))  # initial joules

# ocean joules per unit area
ocean_cell_joulesperunitarea_postdiff_3dmat = ocean_cell_joules_postdiff_3dmat
# just for checking with the diffusion with HP50g
ocean_cell_joules_postdiff_3dmat_check = ocean_cell_joules_postdiff_3dmat
# evolving joules per cell PRE diffusion
ocean_cell_joules_prediff_3dmat[:,:,0] = ocean_cell_initjoules_mat
# evolving joules per cell POST diffusion
ocean_cell_joules_postdiff_3dmat[:,:,0] = ocean_cell_initjoules_mat

oce_temp_ini = (ocean_cell_initjoules_mat
                / (WATER_HEAT_CAPACITY * ocean_cell_mass_gr_mat)) - 273.15

ocean_cell_tempdeg_prediff_3dmat = np.full((N_LAT, N_LONG, N_TIME_STEPS+1), np.nan)
ocean_cell_tempdeg_postdiff_3dmat = np.full((N_LAT, N_LONG, N_TIME_STEPS+1), np.nan)
ocean_cell_tempdeg_prediff_3dmat[:,:,0] = oce_temp_ini  # evolving temp (degC)
ocean_cell_tempdeg_postdiff_3dmat[:,:,0] = oce_temp_ini  # evolving temp (degC)

atmos_cell_initjoules_mat = (ATMOSPHERE_INITIAL_TEMP+273.15)*1004*atmos_cell_mass_kg_mat

atmos_cell_joules_prediff_3dmat = np.full((N_LAT, N_LONG,N_TIME_STEPS+1), np.nan)
atmos_cell_joules_postdiff_3dmat = np.full((N_LAT, N_LONG,N_TIME_STEPS+1), np.nan)
atmos_cell_joulesperunitarea_postdiff_3dmat = atmos_cell_joules_postdiff_3dmat #atm joules per unit area
atmos_cell_joules_postdiff_3dmat_check = atmos_cell_joules_postdiff_3dmat #just for checking with the diffusion with HP50g
atmos_cell_joules_prediff_3dmat[:,:,0] = atmos_cell_initjoules_mat
atmos_cell_joules_postdiff_3dmat[:,:,0] = atmos_cell_initjoules_mat

atmos_cell_inittemp_deg_mat = (atmos_cell_initjoules_mat/(atmos_cell_mass_kg_mat*1004))-273.15


atmos_cell_tempdeg_prediff_3dmat = np.full((N_LAT, N_LONG,N_TIME_STEPS+1), np.nan)
atmos_cell_tempdeg_postdiff_3dmat = np.full((N_LAT, N_LONG,N_TIME_STEPS+1), np.nan)
atmos_cell_tempdeg_prediff_3dmat[:,:,0] = atmos_cell_inittemp_deg_mat
atmos_cell_tempdeg_postdiff_3dmat[:,:,0] = atmos_cell_inittemp_deg_mat


t_end=int(N_TIME_STEPS)

tseries_mean_toa_solar_insol=np.full((t_end,1), np.nan) #to get the average SI across the planet
tseries_ocean_atmos_mean_temp=np.full((t_end,4), np.nan)

# Utility functions used to calculate insolation
_get_year = np.vectorize(lambda x: x.year)
_get_doy = np.vectorize(lambda x: x.timetuple().tm_yday)
_get_sec = np.vectorize(lambda x: x.hour*3600 + x.minute*60 + x.second - 12*3600)

###############################################################################   
############# MAIN TIME LOOP BEGINS ###########################################
###############################################################################   

for t in range(t_end):
    MY_DATE_LONDON = MY_DATE_LONDON + timedelta(hours=1)
    print(t, MY_DATE_LONDON)
    tadj = midcell_long_mat / 180 * 12 * 3600

    dts = MY_DATE_LONDON + tadj * timedelta(seconds=1)

    tadj_sec = midcell_long_mat / 180 * 12 * 3600
    datetime_adj = MY_DATE_LONDON + tadj_sec * timedelta(seconds=1)

    t_sec = _get_sec(datetime_adj)
    years = _get_year(datetime_adj)
    dj = _get_doy(datetime_adj)

    lat = np.deg2rad(midcell_lat_mat)
    long = np.deg2rad(midcell_long_mat)

    # Use eq 9.7 from FAM (p318)
    dl = np.full(years.shape, np.nan)
    dl[years>=2001] = (years[years>=2001] - 2001) / 4
    dl[years<2001] = (years[years<2001] - 2001) / 4 - 1

    njd = 364.5 + (years - 2001) * 365 + dj + dl

    gm = 357.528 + 0.9856003 * njd  # DEG
    lm = 280.460 + 0.9856474 * njd  # DEG
    lam_ec = lm + 1.915 * np.sin(np.deg2rad(gm)) + 0.020*np.sin(np.deg2rad(2*gm)) #in degrees?
    eps_ob = 23.439 - 0.0000004 * njd  # DEG
    delta = np.arcsin(np.sin(np.deg2rad(eps_ob))
                      *np.sin(np.deg2rad(lam_ec)))  # Solar Declination Angle (RAD)
    ha = (2 * np.pi * t_sec) / 86400  # RAD
    theta_s = np.degrees(np.arccos(np.sin(lat) * np.sin(delta)
                                   + np.cos(lat) * np.cos(delta) * np.cos(ha)
                                  )
                        ) #Solar Zenith Angle (DEG)
    toa_solar_insol_mat = SOLAR_CONSTANT*np.cos(np.deg2rad(theta_s))
    toa_solar_insol_mat[np.cos(np.deg2rad(theta_s)) < 0] = 0
    # toa_solar_insol_matav[t,0]=np.mean(toa_solar_insol_mat)
    solar_insol = toa_solar_insol_mat * (1 - albedo_mat)
    toa_solar_insol_perc = toa_solar_insol_mat / SOLAR_CONSTANT * 100
 
    tseries_mean_toa_solar_insol[t, 0] = np.mean(toa_solar_insol_mat)  

   
    tseries_ocean_atmos_mean_temp[t,0]=np.mean(ocean_cell_tempdeg_prediff_3dmat[:,:,t]) #ocean temp without diff
    tseries_ocean_atmos_mean_temp[t,1]=np.mean(ocean_cell_tempdeg_postdiff_3dmat[:,:,t]) #ocean temp + DIFF
    tseries_ocean_atmos_mean_temp[t,2]=np.mean(atmos_cell_tempdeg_prediff_3dmat[:,:,t]) #atm temp without diff
    tseries_ocean_atmos_mean_temp[t,3]=np.mean(atmos_cell_tempdeg_postdiff_3dmat[:,:,t]) #atm temp + DIFF
    
    lat_band_area_prop=np.sum(ocean_cell_area_m2_mat, axis=1)/np.sum(ocean_cell_area_m2_mat) #latitude band as a proportion of total area
    lat_band_area_prop=lat_band_area_prop.reshape(len(lat_band_area_prop),1)
        
        

###############################################################################      
####### phase 1 add/subtract all the fluxes ###################################
###############################################################################    
        
    #these are all fluxes per second
    space2_ocean_flux = solar_insol*(ocean_cell_area_m2_mat)

    
    atmos2_ocean_flux = ocean_cell_area_m2_mat*STEFAN_BOLTZMANN_CONSTANT*ATMOSPHERIC_ABSORPTION_COEFFICIENT*((atmos_cell_tempdeg_prediff_3dmat[:,:,t]+273.15)**4)*0.5
    surf_radiation = ocean_cell_area_m2_mat*STEFAN_BOLTZMANN_CONSTANT*((ocean_cell_tempdeg_prediff_3dmat[:,:,t]+273.15)**4)
    ocean2_atmos_flux = surf_radiation*ATMOSPHERIC_ABSORPTION_COEFFICIENT
    ocean2_space_flux = surf_radiation*(1-ATMOSPHERIC_ABSORPTION_COEFFICIENT)
    atmos2_space_flux = atmos2_ocean_flux
    

    
    ocean_cell_joules_prediff_3dmat[:,:,t+1] = ocean_cell_joules_prediff_3dmat[:,:,t] + space2_ocean_flux*DELTA_TIME_SECS + atmos2_ocean_flux*DELTA_TIME_SECS - ocean2_space_flux*DELTA_TIME_SECS - ocean2_atmos_flux*DELTA_TIME_SECS
    ocean_cell_tempdeg_prediff_3dmat[:,:,t+1] = (ocean_cell_joules_prediff_3dmat[:,:,t]/(4.186*ocean_cell_mass_gr_mat))-273.15
    atmos_cell_joules_prediff_3dmat[:,:,t+1] = atmos_cell_joules_prediff_3dmat[:,:,t] + ocean2_atmos_flux*DELTA_TIME_SECS - atmos2_ocean_flux*DELTA_TIME_SECS - atmos2_space_flux*DELTA_TIME_SECS
    atmos_cell_tempdeg_prediff_3dmat[:,:,t+1] = (atmos_cell_joules_prediff_3dmat[:,:,t]/(atmos_cell_mass_kg_mat*1004))-273.15


    
    ocean_cell_joules_postdiff_3dmat[:,:,t+1] = ocean_cell_joules_postdiff_3dmat[:,:,t] + space2_ocean_flux*DELTA_TIME_SECS + atmos2_ocean_flux*DELTA_TIME_SECS - ocean2_space_flux*DELTA_TIME_SECS - ocean2_atmos_flux*DELTA_TIME_SECS
    ocean_cell_joules_postdiff_3dmat_check[:,:,t+1] = ocean_cell_joules_postdiff_3dmat[:,:,t+1]
    atmos_cell_joules_postdiff_3dmat[:,:,t+1] = atmos_cell_joules_postdiff_3dmat[:,:,t] + ocean2_atmos_flux*DELTA_TIME_SECS - atmos2_ocean_flux*DELTA_TIME_SECS - atmos2_space_flux*DELTA_TIME_SECS   
    atmos_cell_joules_postdiff_3dmat_check[:,:,t+1] = atmos_cell_joules_postdiff_3dmat[:,:,t+1]   
   
  
    ocean_cell_joulesperunitarea_postdiff_3dmat[:,:,t+1] = ocean_cell_joules_postdiff_3dmat[:,:,t+1]/ocean_cell_area_m2_mat#ReShaped #ocean joules per m^2
    atmos_cell_joulesperunitarea_postdiff_3dmat[:,:,t+1] = atmos_cell_joules_postdiff_3dmat[:,:,t+1]/ocean_cell_area_m2_mat#ReShaped #atm joules per m^2
   
    
 
    if (opt2 == 1) or (opt2 == 3):    
###############################################################################           
###### phase 2a include diffusion in Ocean ####################################
###############################################################################       
    
        
        ocean_cell_joulesperunitarea_postdiff_mat=np.full((N_LAT, N_LONG), np.nan)
        for j in range(0,len(toa_solar_insol_mat[0])):  
            for i in range(0,len(toa_solar_insol_mat)):
                if i==0: #TOP ROW
                    ocean_cell_joulesperunitarea_postdiff_mat[i,j]=(((ocean_cell_joulesperunitarea_postdiff_3dmat[i,int(j+len(midcell_lat_mat[0])/2)%len(midcell_lat_mat[0]),t+1]-2*ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+ocean_cell_joulesperunitarea_postdiff_3dmat[i+1,j,t+1])*DIFFUSION_Y_CONSTANT)/(cell_dy_m_mat[i,j]**2)\
                                + ((ocean_cell_joulesperunitarea_postdiff_3dmat[i,j-1,t+1]-2*ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+ocean_cell_joulesperunitarea_postdiff_3dmat[i,(j+1)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_X_CONSTANT)/(cell_dx_m_mat[i,j]**2))*DELTA_TIME_SECS\
                                + ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]
                
                elif i==len(midcell_lat_mat)-1: #BOTTOM ROW
                    ocean_cell_joulesperunitarea_postdiff_mat[i,j]=(((ocean_cell_joulesperunitarea_postdiff_3dmat[i-1,j,t+1]-2*ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+ocean_cell_joulesperunitarea_postdiff_3dmat[i,int(j+len(midcell_lat_mat[0])/2)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_Y_CONSTANT)/(cell_dy_m_mat[i,j]**2)\
                                + ((ocean_cell_joulesperunitarea_postdiff_3dmat[i,j-1,t+1]-2*ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+ocean_cell_joulesperunitarea_postdiff_3dmat[i,(j+1)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_X_CONSTANT)/(cell_dx_m_mat[i,j]**2))*DELTA_TIME_SECS\
                                + ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]

                else: #non bottom non top rows
                    ocean_cell_joulesperunitarea_postdiff_mat[i,j]=(((ocean_cell_joulesperunitarea_postdiff_3dmat[i-1,j,t+1]-2*ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+ocean_cell_joulesperunitarea_postdiff_3dmat[i+1,j,t+1])*DIFFUSION_Y_CONSTANT)/(cell_dy_m_mat[i,j]**2)\
                                + ((ocean_cell_joulesperunitarea_postdiff_3dmat[i,j-1,t+1]-2*ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+ocean_cell_joulesperunitarea_postdiff_3dmat[i,(j+1)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_X_CONSTANT)/(cell_dx_m_mat[i,j]**2))*DELTA_TIME_SECS\
                                + ocean_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]
        ocean_cell_joulesperunitarea_postdiff_3dmat[:,:,t+1] = ocean_cell_joulesperunitarea_postdiff_mat #transfers back
        ocean_cell_joules_postdiff_3dmat[:,:,t+1] = ocean_cell_joulesperunitarea_postdiff_3dmat[:,:,t+1]*ocean_cell_area_m2_mat #converting back from j/m^2 to j
        ocean_cell_tempdeg_postdiff_3dmat[:,:,t+1] = (ocean_cell_joules_postdiff_3dmat[:,:,t+1]/(4.186*ocean_cell_mass_gr_mat))-273.15


    if (opt2 == 2) or (opt2 == 3):

###############################################################################           
###### phase 2b include diffusion in Atmosphere ############################### 
###############################################################################       
    
       
        atmos_cell_joulesperunitarea_postdiff_mat=np.full((N_LAT, N_LONG), np.nan)
        for j in range(0,len(toa_solar_insol_mat[0])):  
            for i in range(0,len(toa_solar_insol_mat)):
                if i==0: #TOP ROW
                    atmos_cell_joulesperunitarea_postdiff_mat[i,j]=(((atmos_cell_joulesperunitarea_postdiff_3dmat[i,int(j+len(midcell_lat_mat[0])/2)%len(midcell_lat_mat[0]),t+1]-2*atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+atmos_cell_joulesperunitarea_postdiff_3dmat[i+1,j,t+1])*DIFFUSION_Y_CONSTANT)/(cell_dy_m_mat[i,j]**2)\
                                + ((atmos_cell_joulesperunitarea_postdiff_3dmat[i,j-1,t+1]-2*atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+atmos_cell_joulesperunitarea_postdiff_3dmat[i,(j+1)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_X_CONSTANT)/(cell_dx_m_mat[i,j]**2))*DELTA_TIME_SECS\
                                + atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]

                elif i==len(midcell_lat_mat)-1: #BOTTOM ROW
                    atmos_cell_joulesperunitarea_postdiff_mat[i,j]=(((atmos_cell_joulesperunitarea_postdiff_3dmat[i-1,j,t+1]-2*atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+atmos_cell_joulesperunitarea_postdiff_3dmat[i,int(j+len(midcell_lat_mat[0])/2)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_Y_CONSTANT)/(cell_dy_m_mat[i,j]**2)\
                                + ((atmos_cell_joulesperunitarea_postdiff_3dmat[i,j-1,t+1]-2*atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+atmos_cell_joulesperunitarea_postdiff_3dmat[i,(j+1)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_X_CONSTANT)/(cell_dx_m_mat[i,j]**2))*DELTA_TIME_SECS\
                                + atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]

                else: #non bottom non top rows
                    atmos_cell_joulesperunitarea_postdiff_mat[i,j]=(((atmos_cell_joulesperunitarea_postdiff_3dmat[i-1,j,t+1]-2*atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+atmos_cell_joulesperunitarea_postdiff_3dmat[i+1,j,t+1])*DIFFUSION_Y_CONSTANT)/(cell_dy_m_mat[i,j]**2)\
                                + ((atmos_cell_joulesperunitarea_postdiff_3dmat[i,j-1,t+1]-2*atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]+atmos_cell_joulesperunitarea_postdiff_3dmat[i,(j+1)%len(midcell_lat_mat[0]),t+1])*DIFFUSION_X_CONSTANT)/(cell_dx_m_mat[i,j]**2))*DELTA_TIME_SECS\
                                + atmos_cell_joulesperunitarea_postdiff_3dmat[i,j,t+1]
        atmos_cell_joulesperunitarea_postdiff_3dmat[:,:,t+1] = atmos_cell_joulesperunitarea_postdiff_mat #transfers back
        atmos_cell_joules_postdiff_3dmat[:,:,t+1] = atmos_cell_joulesperunitarea_postdiff_3dmat[:,:,t+1]*ocean_cell_area_m2_mat #converting back from j/m^2 to j
        atmos_cell_tempdeg_postdiff_3dmat[:,:,t+1] = (atmos_cell_joules_postdiff_3dmat[:,:,t+1]/(1004*atmos_cell_mass_kg_mat))-273.15
        

##### end #####



        
    if t%(24*30)==0:
        month=month+1
        if month>12:
            month=month%12
        
        plt.clf()

        month_str = (MONTH_LIST[month-1])

        
        plt.figure(10,figsize=(15,10))
        plt.subplot(3, 2, 1)
        plt.text(0.05,18.5,'A',fontsize=12)

        
        plt.pcolor(np.flipud(ocean_cell_tempdeg_prediff_3dmat[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("No diffusion ocean-" + str(month_str) + "-" + str(t))
        plt.ylabel('Degrees Latitude')
        
        
        plt.subplot(3, 2, 2)
        plt.ylabel('Degrees Latitude')
        plt.text(0.05,18.5,'D',fontsize=12)

        
        plt.pcolor(np.flipud(ocean_cell_tempdeg_postdiff_3dmat[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        plt.show()
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("Diffusion ocean-" + str(month_str) + "-" + str(t))
        
        
        plt.subplot(3, 2, 3)
        plt.text(0.05,18.5,'B',fontsize=12)
        plt.ylabel('Degrees Latitude')

         
        plt.pcolor(np.flipud(atmos_cell_tempdeg_prediff_3dmat[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("No diffusion atmosphere-" + str(month_str) + "-" + str(t))
        
        
        plt.subplot(3, 2, 4)
        plt.ylabel('Degrees Latitude')
        plt.text(0.05,18.5,'E',fontsize=12)

         
        plt.pcolor(np.flipud(atmos_cell_tempdeg_postdiff_3dmat[:,:,t]), cmap='bwr')
        plt.xticks( np.arange(0,xticks.shape[0]), xticks )
        plt.yticks( np.arange(0,yticks.shape[0]), yticks )
        plt.show()
        clb=plt.colorbar()
        clb.set_label('DegC',rotation=270)
        plt.clim(-273.15, 273.15)
        plt.title("Diffusion atmosphere-" + str(month_str) + "-" + str(t))
        
        
        plt.subplot(3, 2, 5)
        plt.ylabel('Degrees Latitude')
        plt.xlabel('Temperature (Degrees C)')

      
        ocean_cell_tempdeg_prediff_latmean=ocean_cell_tempdeg_prediff_3dmat[:,:,t]
        ocean_cell_tempdeg_prediff_latmean=np.mean(ocean_cell_tempdeg_prediff_latmean, axis=1)
        atmos_cell_tempdeg_prediff_latmean=atmos_cell_tempdeg_prediff_3dmat[:,:,t]
        atmos_cell_tempdeg_prediff_latmean=np.mean(atmos_cell_tempdeg_prediff_latmean, axis=1)
        midcell_lat_y1 = np.arange(midcell_lat_mat[-1,0],midcell_lat_mat[0,0]+(midcell_lat_mat[0,0]-midcell_lat_mat[1,0]),180/midcell_lat_mat.shape[0])
        midcell_lat_y1 = np.flipud(midcell_lat_y1)
        plt.plot(ocean_cell_tempdeg_prediff_latmean,midcell_lat_y1,'b',label='ocean')
        plt.plot(atmos_cell_tempdeg_prediff_latmean,midcell_lat_y1,'r',label='atmos')
        plt.legend(loc='upper right',prop={'size':10})
        plt.clim(-273.15, 273.15)
        plt.title("C No diffusion lat-temp-profile-" + str(month_str) + "-" + str(t))
        
        
        plt.subplot(3, 2, 6)
        plt.ylabel('Degrees Latitude')
        plt.xlabel('Degrees C')

        
        ocean_cell_tempdeg_postdiff_latmean=ocean_cell_tempdeg_postdiff_3dmat[:,:,t]
        ocean_cell_tempdeg_postdiff_latmean=np.mean(ocean_cell_tempdeg_postdiff_latmean, axis=1)
        atmos_cell_tempdeg_postdiff_latmean=atmos_cell_tempdeg_postdiff_3dmat[:,:,t]
        atmos_cell_tempdeg_postdiff_latmean=np.mean(atmos_cell_tempdeg_postdiff_latmean, axis=1)
        midcell_lat_y2 = np.arange(midcell_lat_mat[-1,0],midcell_lat_mat[0,0]+(midcell_lat_mat[0,0]-midcell_lat_mat[1,0]),180/midcell_lat_mat.shape[0])
        midcell_lat_y2 = np.flipud(midcell_lat_y2)
        plt.plot(ocean_cell_tempdeg_postdiff_latmean,midcell_lat_y2,'b',label='ocean')
        plt.plot(atmos_cell_tempdeg_postdiff_latmean,midcell_lat_y2,'r',label='atmos')
        plt.legend(loc='upper right',prop={'size':10})
        plt.show()
        plt.clim(-273.15, 273.15)
        plt.title("F diffusion lat-temp-profile-" + str(month_str) + "-" + str(t))
        
        
        plt.pause(0.1)
        show_plot()


tseries_ocean_atmos_mean_temp_area_weighted=np.full((ocean_cell_joules_prediff_3dmat.shape[2],4), np.nan)
tseries_ocean_nodiff_meantemp=np.mean(ocean_cell_tempdeg_prediff_3dmat, axis=1)
tseries_ocean_nodiff_meantemp=tseries_ocean_nodiff_meantemp*lat_band_area_prop
tseries_ocean_nodiff_meantemp=np.sum(tseries_ocean_nodiff_meantemp, axis=0)
tseries_ocean_atmos_mean_temp_area_weighted[:,0]=tseries_ocean_nodiff_meantemp

tseries_ocean_diff_meantemp=np.mean(ocean_cell_tempdeg_postdiff_3dmat, axis=1)
tseries_ocean_diff_meantemp=tseries_ocean_diff_meantemp*lat_band_area_prop
tseries_ocean_diff_meantemp=np.sum(tseries_ocean_diff_meantemp, axis=0)
tseries_ocean_atmos_mean_temp_area_weighted[:,1]=tseries_ocean_diff_meantemp

tseries_atmos_nodiff_meantemp=np.mean(atmos_cell_tempdeg_prediff_3dmat, axis=1)
tseries_atmos_nodiff_meantemp=tseries_atmos_nodiff_meantemp*lat_band_area_prop
tseries_atmos_nodiff_meantemp=np.sum(tseries_atmos_nodiff_meantemp, axis=0)
tseries_ocean_atmos_mean_temp_area_weighted[:,2]=tseries_atmos_nodiff_meantemp

tseries_atmos_diff_meantemp=np.mean(atmos_cell_tempdeg_postdiff_3dmat, axis=1)
tseries_atmos_diff_meantemp=tseries_atmos_diff_meantemp*lat_band_area_prop
tseries_atmos_diff_meantemp=np.sum(tseries_atmos_diff_meantemp, axis=0)
tseries_ocean_atmos_mean_temp_area_weighted[:,3]=tseries_atmos_diff_meantemp

print(tseries_ocean_atmos_mean_temp[t_end-1,:])
print(tseries_ocean_atmos_mean_temp_area_weighted[t_end-1,:]) #This 2nd row gives you the more realistic average temp as weights the boxes according to area


plt.figure(2)
plt.title('Model ocean and atmosphere temperature evolution over time')
x= np.arange('2001-09-21T12:00:00.0', MY_DATE_LONDON, dtype='datetime64[h]')

y1=tseries_ocean_atmos_mean_temp[:,0]
y2=tseries_ocean_atmos_mean_temp[:,1]
y3=tseries_ocean_atmos_mean_temp[:,2]
y4=tseries_ocean_atmos_mean_temp[:,3]

y5=tseries_ocean_atmos_mean_temp_area_weighted[0:-1,0]
y6=tseries_ocean_atmos_mean_temp_area_weighted[0:-1,1]
y7=tseries_ocean_atmos_mean_temp_area_weighted[0:-1,2]
y8=tseries_ocean_atmos_mean_temp_area_weighted[0:-1,3]


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