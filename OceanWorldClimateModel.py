#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 15:05:47 2016
@author: markprosser
"""
try:
    # To be run in Spyder
    from IPython import get_ipython
    get_ipython().magic('reset -f')  # NOQA
except:
    pass
# sys.path.append('PythFunctions')
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
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


def calc_diffusion(arr, dx, dy, x_const, y_const, dt):
    """
    Calculate 2D diffusion using centred differences

    Arguments
    ---------
    arr : numpy array of shape (M, N)
        array to calculate diffusion
    dx, dy : numpy arrays of shape (M, N)
        grid spacing
    x_const, y_const: numpy arrays of shape (M, N) or scalar
        diffusion coefficients
    dt : scalar
        time step in seconds

    Returns
    -------
    numpy array of shape (M, N)
    """
    nlat, nlon = arr.shape
    work = np.full((nlat, nlon), np.nan)
    for j in range(nlon):
        wtf1 = int(j + nlon / 2) % nlon
        wtf2 = (j + 1) % nlon
        # TOP ROW (i = 0)
        _x_diff = ((arr[0, wtf1] - 2 * arr[0, j] + arr[1, j])
                   * x_const) / (dx[0, j] ** 2)
        _y_diff = ((arr[0, j-1] - 2 * arr[0, j] + arr[0, wtf2])
                   * y_const) / (dy[0, j] ** 2)
        work[0, j] = ((_x_diff + _y_diff) * dt + arr[0, j])

        # BOTTOM ROW (i = last - 1)
        _x_diff = ((arr[nlat-1, wtf1] - 2 * arr[nlat-1, j] + arr[1, j])
                   * x_const) / (dx[nlat-1, j] ** 2)
        _y_diff = ((arr[nlat-1, j-1] - 2 * arr[nlat-1, j]
                    + arr[nlat-1, wtf2])
                   * y_const) / (dy[nlat-1, j] ** 2)
        work[nlat-1, j] = ((_x_diff + _y_diff) * dt + arr[nlat-1, j])

    sub_arr = arr[1:nlat-1, :]
    _x_diff = ((np.roll(sub_arr, 1, axis=1)  # roll right
                - 2 * sub_arr
                + np.roll(sub_arr, -1, axis=1))  # roll left
               * x_const / (dx[1:nlat - 1, :] ** 2))
    _y_diff = ((np.roll(sub_arr, 1, axis=0)  # roll down
                - 2 * sub_arr
                + np.roll(sub_arr, -1, axis=0))  # roll up
               * y_const / (dy[1:nlat - 1, :] ** 2))

    work[1:nlat-1, :] = ((_y_diff + _x_diff) * dt + sub_arr)
    return work


opt2 = 3
# 1=Oceandiffusion only
# 2=Atmsopherediffusion only
# 3=both

T0 = 273.15
# PoleFrac=0.99
SOLAR_CONST = 1361
ALBEDO = 0.3
OCEAN_INIT_TEMP = -T0  # initial temp of water degC
ATMOS_INIT_TEMP = -T0
ATMOS_KG = 5.14E18  # kg
EARTH_RAD_M = 6371000  # m
N_LAT = 5
N_LONG = 3
STEF_BOLTZ_CONST = 5.67E-8
ATM_ABSORP_COEF = 0.7814
WATER_HEAT_CAPAC = 4.186  # CHECK!
AIR_HEAT_CAPAC = 1004.5  # CHECK!
DELTA_SECS = 3600
N_TIME_STEPS = int(1 * 365 * 24 * 1.5)
DIFF_X_CONST = 800000  # diffusion constant in X #90000
DIFF_Y_CONST = 800000  # 2500000 #diffusion constant in Y
LASER_FROM_SPACE = 0  # 9999#99999
OCEAN_DEPTH_M = 1
START_MONTH = 9
MONTH_LIST = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# maxNH %92days.*24

# ********************** END OF INITIAL CONDITIONS INPUT **********************
earth_hemisph_area = 2 * np.pi * EARTH_RAD_M ** 2
earth_area = 4 * np.pi * EARTH_RAD_M ** 2
lat_res_deg = 180 / N_LAT
long_res_deg = 360 / N_LONG
month = START_MONTH
month_str = MONTH_LIST[month - 1]
my_date_london = datetime(2001, START_MONTH, 21, 12, 0, 0)

xticks = np.arange(-180, 181, long_res_deg)
yticks = np.arange(-90, 91, lat_res_deg)
midcell_lat_2d = np.full((N_LAT, N_LONG), np.nan)
midcell_long_2d = np.full((N_LAT, N_LONG), np.nan)
cell_dy_m_2d = np.full((N_LAT, N_LONG), np.nan)
cell_dx_m_2d = np.full((N_LAT, N_LONG), np.nan)
albedo_2d = np.full((N_LAT, N_LONG), np.nan)
stability = np.full((N_LAT, N_LONG), np.nan)  # for stability analysis
surf_radiation = np.full((N_LAT, N_LONG), np.nan)
space2_ocean_flux = np.full((N_LAT, N_LONG), np.nan)
atmos2_ocean_flux = np.full((N_LAT, N_LONG), np.nan)
ocean2_atmos_flux = np.full((N_LAT, N_LONG), np.nan)
ocean2_space_flux = np.full((N_LAT, N_LONG), np.nan)
atmos2_space_flux = np.full((N_LAT, N_LONG), np.nan)
ocean_cell_m2_2d = np.full((N_LAT, N_LONG), np.nan)
# evolving joules per cell PRE diffusion
ocean_cell_j_prediff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
# evolving joules per cell POST diffusion
ocean_cell_j_postdiff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)

# TODO: remove unnessesary declarations, e.g. toa_insol_2d
# toa_insol_2d = np.full((N_LAT, N_LONG), np.nan) #Solar Insolation
alb_insol_2d = np.full((N_LAT, N_LONG), np.nan)  # Solar Insolation * (1-ALB)
toa_insol_perc = np.full((N_LAT, N_LONG), np.nan)  # Solar Insol Percent

midcell_lat_2d[:] = np.arange(90 - lat_res_deg / 2,
                              -90 + lat_res_deg / 2 - 1,
                              -lat_res_deg)[:, np.newaxis]
midcell_long_2d[:] = np.arange(-180 + long_res_deg / 2,
                               180 - long_res_deg / 2 + 1,
                               long_res_deg)[np.newaxis, :]

cell_dy_m_2d[:] = (EARTH_RAD_M * np.pi) / N_LAT

cell_dx_m_2d[:] = (EARTH_RAD_M * np.cos(np.deg2rad(midcell_lat_2d))
                   * 2 * np.pi) / N_LONG

stability = 0.5 * np.minimum(cell_dx_m_2d * cell_dx_m_2d,
                             cell_dy_m_2d * cell_dy_m_2d) / DELTA_SECS

albedo_2d[:] = ALBEDO

# calculate the area of a grid cell
# using HP 50g GLOBEARE prog methodology
A = np.arange(N_LAT) * lat_res_deg - 90
B = np.arange(1, N_LAT + 1) * lat_res_deg - 90
C = 0.5 * earth_area * (1 - np.sin(np.deg2rad(B)))
D = 0.5 * earth_area * (1 - np.sin(np.deg2rad(A)))
E = D - C

ocean_cell_m2_2d[:] = E[:, np.newaxis] / N_LONG

# reshape to aid with a 3D array division later
ocean_cell_m2_2d_reshaped = ocean_cell_m2_2d[..., np.newaxis]

# FOR THE FOLLOWING sum(sum(FOLLOWING)) to get global
# (as opposed to cell values)
frac_ocean_cell_m2_2d = (ocean_cell_m2_2d /
                         ocean_cell_m2_2d.sum())
# assert frac_ocean_cell_m2_2d.sum() == 1, 'sum of this ARR should be 1'
atmos_cell_kg_2d = frac_ocean_cell_m2_2d * ATMOS_KG
ocean_cell_m3_2d = ocean_cell_m2_2d * OCEAN_DEPTH_M  # *m depth m^3
# FIXME: multiply by density? - mult by 1 million is effectively doing that
# get mass (gr) of water per cell
ocean_cell_gr_2d = ocean_cell_m3_2d * 1000000
ocean_cell_init_j_2d = (ocean_cell_gr_2d
                        * WATER_HEAT_CAPAC
                        * (OCEAN_INIT_TEMP + T0))  # initial joules

# ocean joules per unit area
ocean_cell_jperunitarea_postdiff_3d = ocean_cell_j_postdiff_3d
# just for checking with the diffusion with HP50g
ocean_cell_j_postdiff_3d_check = ocean_cell_j_postdiff_3d
# evolving joules per cell PRE diffusion
ocean_cell_j_prediff_3d[:, :, 0] = ocean_cell_init_j_2d
# evolving joules per cell POST diffusion
ocean_cell_j_postdiff_3d[:, :, 0] = ocean_cell_init_j_2d

oce_temp_ini = (ocean_cell_init_j_2d
                / (WATER_HEAT_CAPAC * ocean_cell_gr_2d)) - T0

ocean_cell_deg_prediff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
ocean_cell_deg_postdiff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
ocean_cell_deg_prediff_3d[:, :, 0] = oce_temp_ini  # evolving temp (degC)
ocean_cell_deg_postdiff_3d[:, :, 0] = oce_temp_ini  # evolving temp (degC)

atmos_cell_init_j_2d = (ATMOS_INIT_TEMP + T0) * \
                        AIR_HEAT_CAPAC * atmos_cell_kg_2d

atmos_cell_j_prediff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
atmos_cell_j_postdiff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
# atm joules per unit area
atmos_cell_jperunitarea_postdiff_3d = atmos_cell_j_postdiff_3d
# just for checking with the diffusion with HP50g
atmos_cell_j_postdiff_3d_check = atmos_cell_j_postdiff_3d.copy()
atmos_cell_j_prediff_3d[:, :, 0] = atmos_cell_init_j_2d
atmos_cell_j_postdiff_3d[:, :, 0] = atmos_cell_init_j_2d

atmos_cell_init_deg_2d = (atmos_cell_init_j_2d /
                          (atmos_cell_kg_2d * AIR_HEAT_CAPAC) - T0)


atmos_cell_deg_prediff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
atmos_cell_deg_postdiff_3d = np.full((N_LAT, N_LONG, N_TIME_STEPS + 1), np.nan)
atmos_cell_deg_prediff_3d[:, :, 0] = atmos_cell_init_deg_2d
atmos_cell_deg_postdiff_3d[:, :, 0] = atmos_cell_init_deg_2d

# to get the average SI across the planet
tseries_mean_toa_insol = np.full((N_TIME_STEPS, 1), np.nan)

# Utility functions that allow datetime operations on arrays
# (used to calculate insolation)
_get_year = np.vectorize(lambda x: x.year)
_get_doy = np.vectorize(lambda x: x.timetuple().tm_yday)
_get_sec = np.vectorize(lambda x: x.hour * 3600
                        + x.minute * 60
                        + x.second - 12 * 3600)

######################### NOQA
# MAIN TIME LOOP BEGINS #
######################### NOQA
for t in range(N_TIME_STEPS):
    my_date_london = my_date_london + timedelta(hours=1)
    print(t, my_date_london)
    tadj = midcell_long_2d / 180 * 12 * 3600

    dts = my_date_london + tadj * timedelta(seconds=1)

    tadj_sec = midcell_long_2d / 180 * 12 * 3600
    datetime_adj = my_date_london + tadj_sec * timedelta(seconds=1)

    t_sec = _get_sec(datetime_adj)
    years = _get_year(datetime_adj)
    dj = _get_doy(datetime_adj)

    lat = np.deg2rad(midcell_lat_2d)
    long = np.deg2rad(midcell_long_2d)

    # Use eq 9.7 from FAM (p318)
    dl = np.full(years.shape, np.nan)
    dl[years >= 2001] = (years[years >= 2001] - 2001) / 4
    dl[years < 2001] = (years[years < 2001] - 2001) / 4 - 1

    njd = 364.5 + (years - 2001) * 365 + dj + dl

    gm = 357.528 + 0.9856003 * njd  # DEG
    lm = 280.460 + 0.9856474 * njd  # DEG
    lam_ec = (lm + 1.915 * np.sin(np.deg2rad(gm))
              + 0.020 * np.sin(np.deg2rad(2 * gm)))  # in degrees?
    eps_ob = 23.439 - 0.0000004 * njd  # DEG
    delta = np.arcsin(np.sin(np.deg2rad(eps_ob))
                      * np.sin(np.deg2rad(lam_ec)))  # Solar Declination Angle
    ha = (2 * np.pi * t_sec) / 86400  # RAD
    theta_s = np.degrees(np.arccos(np.sin(lat) * np.sin(delta)
                                   + np.cos(lat) * np.cos(delta) * np.cos(ha)
                                   )
                         )  # Solar Zenith Angle (DEG)
    toa_insol_2d = SOLAR_CONST * np.cos(np.deg2rad(theta_s))
    toa_insol_2d[np.cos(np.deg2rad(theta_s)) < 0] = 0
    # toa_insol_2dav[t,0]=np.mean(toa_insol_2d)
    alb_insol_2d = toa_insol_2d * (1 - albedo_2d)
    toa_insol_perc = toa_insol_2d / SOLAR_CONST * 100

    tseries_mean_toa_insol[t, 0] = toa_insol_2d.mean()

    # latitude band as a proportion of total area
    lat_band_area_prop = ocean_cell_m2_2d.sum(axis=1) / ocean_cell_m2_2d.sum()

    ####################################### NOQA
    # phase 1 add/subtract all the fluxes #
    ####################################### NOQA
    # these are all fluxes per second
    space2_ocean_flux = alb_insol_2d * (ocean_cell_m2_2d)

    atmos2_ocean_flux = (ocean_cell_m2_2d * STEF_BOLTZ_CONST * ATM_ABSORP_COEF
                         * (atmos_cell_deg_prediff_3d[:, :, t] + T0) ** 4
                         * 0.5)
    surf_radiation = (ocean_cell_m2_2d * STEF_BOLTZ_CONST
                      * (ocean_cell_deg_prediff_3d[:, :, t] + T0) ** 4)
    ocean2_atmos_flux = surf_radiation * ATM_ABSORP_COEF
    ocean2_space_flux = surf_radiation * (1 - ATM_ABSORP_COEF)
    atmos2_space_flux = atmos2_ocean_flux

    ocean_cell_j_prediff_3d[:, :, t+1] = (ocean_cell_j_prediff_3d[:, :, t]
                                          + space2_ocean_flux * DELTA_SECS
                                          + atmos2_ocean_flux * DELTA_SECS
                                          - ocean2_space_flux * DELTA_SECS
                                          - ocean2_atmos_flux * DELTA_SECS)
    ocean_cell_deg_prediff_3d[:, :, t+1] = (ocean_cell_j_prediff_3d[:, :, t]
                                              / (WATER_HEAT_CAPAC * ocean_cell_gr_2d)) - T0  # NOQA
    atmos_cell_j_prediff_3d[:, :, t+1] = (atmos_cell_j_prediff_3d[:, :, t]
                                          + ocean2_atmos_flux * DELTA_SECS
                                          - atmos2_ocean_flux * DELTA_SECS
                                          - atmos2_space_flux * DELTA_SECS)
    atmos_cell_deg_prediff_3d[:, :, t+1] = atmos_cell_j_prediff_3d[:, :, t] / (atmos_cell_kg_2d * AIR_HEAT_CAPAC) - T0  # NOQA

    ocean_cell_j_postdiff_3d[:, :, t+1] = (ocean_cell_j_postdiff_3d[:, :, t]
                                           + space2_ocean_flux * DELTA_SECS
                                           + atmos2_ocean_flux * DELTA_SECS
                                           - ocean2_space_flux * DELTA_SECS
                                           - ocean2_atmos_flux * DELTA_SECS)
    ocean_cell_j_postdiff_3d_check[:, :, t+1] = ocean_cell_j_postdiff_3d[:, :, t+1]  # NOQA
    atmos_cell_j_postdiff_3d[:, :, t+1] = (atmos_cell_j_postdiff_3d[:, :, t]
                                           + ocean2_atmos_flux * DELTA_SECS
                                           - atmos2_ocean_flux * DELTA_SECS
                                           - atmos2_space_flux * DELTA_SECS)
    atmos_cell_j_postdiff_3d_check[
        :, :, t+1] = atmos_cell_j_postdiff_3d[:, :, t+1]  # why does it exist?

    ocean_cell_jperunitarea_postdiff_3d[:, :, t+1] = ocean_cell_j_postdiff_3d[:, :, t+1] / ocean_cell_m2_2d  # NOQA ReShaped #ocean joules per m^2
    atmos_cell_jperunitarea_postdiff_3d[:, :, t+1] = atmos_cell_j_postdiff_3d[:, :, t+1] / ocean_cell_m2_2d  # NOQA ReShaped #atm joules per m^2

    if (opt2 == 1) or (opt2 == 3):
        ####################################### NOQA
        # phase 2a include diffusion in Ocean #
        ####################################### NOQA
        ocean_cell_jperunitarea_postdiff_3d[:, :, t+1] = calc_diffusion(ocean_cell_jperunitarea_postdiff_3d[:, :, t+1],  # NOQA
                                                                        cell_dx_m_2d, cell_dy_m_2d,  # NOQA
                                                                        DIFF_X_CONST, DIFF_Y_CONST,  # NOQA
                                                                        DELTA_SECS)  # NOQA

        ocean_cell_j_postdiff_3d[:, :, t+1] = ocean_cell_jperunitarea_postdiff_3d[:, :, t+1] * ocean_cell_m2_2d  # converting back from j/m^2 to j  # NOQA
        ocean_cell_deg_postdiff_3d[:, :, t+1] = (ocean_cell_j_postdiff_3d[:, :, t+1] / (WATER_HEAT_CAPAC * ocean_cell_gr_2d)) - T0  # NOQA

    if (opt2 == 2) or (opt2 == 3):
        ############################################ NOQA
        # phase 2b include diffusion in Atmosphere #
        ############################################ NOQA
        atmos_cell_jperunitarea_postdiff_3d[:, :, t+1] = calc_diffusion(atmos_cell_jperunitarea_postdiff_3d[:, :, t+1],  # NOQA
                                                                        cell_dx_m_2d, cell_dy_m_2d,  # NOQA
                                                                        DIFF_X_CONST, DIFF_Y_CONST,  # NOQA
                                                                        DELTA_SECS)  # NOQA
        # converting back from j/m^2 to j
        atmos_cell_j_postdiff_3d[:, :, t+1] = atmos_cell_jperunitarea_postdiff_3d[:, :, t+1] * ocean_cell_m2_2d  # NOQA
        atmos_cell_deg_postdiff_3d[:, :, t+1] = (atmos_cell_j_postdiff_3d[:, :, t+1] / (AIR_HEAT_CAPAC * atmos_cell_kg_2d)) - T0  # NOQA

    # END OF CALCULATIONS

    if t % (24 * 30) == 0:
        month = month + 1
        if month > 12:
            month = month % 12

        plt.clf()

        month_str = (MONTH_LIST[month - 1])

        plt.figure(10, figsize=(15, 10))
        plt.subplot(3, 2, 1)
        plt.text(0.05, 18.5, 'A', fontsize=12)

        plt.pcolor(np.flipud(ocean_cell_deg_prediff_3d[:, :, t]),
                   cmap='bwr', vmin=-T0, vmax=T0)
        plt.xticks(np.arange(0, xticks.shape[0]), xticks)
        plt.yticks(np.arange(0, yticks.shape[0]), yticks)
        clb = plt.colorbar()
        clb.set_label('DegC', rotation=270)
        plt.title("No diffusion ocean-" + str(month_str) + "-" + str(t))
        plt.ylabel('Degrees Latitude')

        plt.subplot(3, 2, 2)
        plt.ylabel('Degrees Latitude')
        plt.text(0.05, 18.5, 'D', fontsize=12)

        plt.pcolor(np.flipud(ocean_cell_deg_postdiff_3d[:, :, t]),
                   cmap='bwr', vmin=-T0, vmax=T0)
        plt.xticks(np.arange(0, xticks.shape[0]), xticks)
        plt.yticks(np.arange(0, yticks.shape[0]), yticks)
        # plt.show()
        clb = plt.colorbar()
        clb.set_label('DegC', rotation=270)
        plt.title("Diffusion ocean-" + str(month_str) + "-" + str(t))

        plt.subplot(3, 2, 3)
        plt.text(0.05, 18.5, 'B', fontsize=12)
        plt.ylabel('Degrees Latitude')

        plt.pcolor(np.flipud(atmos_cell_deg_prediff_3d[:, :, t]),
                   cmap='bwr', vmin=-T0, vmax=T0)
        plt.xticks(np.arange(0, xticks.shape[0]), xticks)
        plt.yticks(np.arange(0, yticks.shape[0]), yticks)
        clb = plt.colorbar()
        clb.set_label('DegC', rotation=270)
        plt.title("No diffusion atmosphere-" + str(month_str) + "-" + str(t))

        plt.subplot(3, 2, 4)
        plt.ylabel('Degrees Latitude')
        plt.text(0.05, 18.5, 'E', fontsize=12)

        plt.pcolor(np.flipud(atmos_cell_deg_postdiff_3d[:, :, t]),
                   cmap='bwr', vmin=-T0, vmax=T0)
        plt.xticks(np.arange(0, xticks.shape[0]), xticks)
        plt.yticks(np.arange(0, yticks.shape[0]), yticks)
        # plt.show()
        clb = plt.colorbar()
        clb.set_label('DegC', rotation=270)
        plt.title("Diffusion atmosphere-" + str(month_str) + "-" + str(t))

        plt.subplot(3, 2, 5)
        plt.ylabel('Degrees Latitude')
        plt.xlabel('Temperature (Degrees C)')

        ocean_cell_deg_prediff_latmean = ocean_cell_deg_prediff_3d[:, :, t]
        ocean_cell_deg_prediff_latmean = np.mean(
            ocean_cell_deg_prediff_latmean, axis=1)
        atmos_cell_deg_prediff_latmean = atmos_cell_deg_prediff_3d[:, :, t]
        atmos_cell_deg_prediff_latmean = np.mean(
            atmos_cell_deg_prediff_latmean, axis=1)
        midcell_lat_y1 = np.arange(midcell_lat_2d[-1, 0],
                                   midcell_lat_2d[0, 0]
                                   + midcell_lat_2d[0, 0]
                                   - midcell_lat_2d[1, 0],
                                   180 / midcell_lat_2d.shape[0])
        midcell_lat_y1 = np.flipud(midcell_lat_y1)
        plt.plot(ocean_cell_deg_prediff_latmean,
                 midcell_lat_y1, 'b', label='ocean')
        plt.plot(atmos_cell_deg_prediff_latmean,
                 midcell_lat_y1, 'r', label='atmos')
        plt.legend(loc='upper right', prop={'size': 10})
        # plt.clim(-T0, T0)
        plt.title("C No diffusion lat-temp-profile-" +
                  str(month_str) + "-" + str(t))

        plt.subplot(3, 2, 6)
        plt.ylabel('Degrees Latitude')
        plt.xlabel('Degrees C')

        ocean_cell_deg_postdiff_latmean = ocean_cell_deg_postdiff_3d[:, :, t]
        ocean_cell_deg_postdiff_latmean = np.mean(
            ocean_cell_deg_postdiff_latmean, axis=1)
        atmos_cell_deg_postdiff_latmean = atmos_cell_deg_postdiff_3d[:, :, t]
        atmos_cell_deg_postdiff_latmean = np.mean(
            atmos_cell_deg_postdiff_latmean, axis=1)
        midcell_lat_y2 = np.arange(midcell_lat_2d[-1, 0],
                                   midcell_lat_2d[0, 0]
                                   + midcell_lat_2d[0, 0]
                                   - midcell_lat_2d[1, 0],
                                   180 / midcell_lat_2d.shape[0])
        midcell_lat_y2 = np.flipud(midcell_lat_y2)
        plt.plot(ocean_cell_deg_postdiff_latmean,
                 midcell_lat_y2, 'b', label='ocean')
        plt.plot(atmos_cell_deg_postdiff_latmean,
                 midcell_lat_y2, 'r', label='atmos')
        plt.legend(loc='upper right', prop={'size': 10})
        # plt.show()
        # plt.clim(-T0, T0)
        plt.title("F diffusion lat-temp-profile-" +
                  str(month_str) + "-" + str(t))

        plt.pause(0.1)
        show_plot()


tseries_oa_mean_temp_area_weighted = np.full(((ocean_cell_j_prediff_3d
                                               .shape[2]), 4),
                                             np.nan)
_arrs = (ocean_cell_deg_prediff_3d,
         ocean_cell_deg_postdiff_3d,
         atmos_cell_deg_prediff_3d,
         atmos_cell_deg_postdiff_3d)
for i, arr in enumerate(_arrs):
    mean_arr = arr.mean(axis=1)
    mean_arr *= lat_band_area_prop[:, np.newaxis]
    tseries_oa_mean_temp_area_weighted[:, i] = mean_arr.sum(axis=0)

y1 = ocean_cell_deg_prediff_3d.mean(axis=(0, 1))  # ocean temp without diff
y2 = ocean_cell_deg_postdiff_3d.mean(axis=(0, 1))  # ocean temp + DIFF
y3 = atmos_cell_deg_prediff_3d.mean(axis=(0, 1))  # atm temp without diff
y4 = atmos_cell_deg_postdiff_3d.mean(axis=(0, 1))  # atm temp + DIFF

print('ocean temp without diff: {}'.format(y1[N_TIME_STEPS - 1]))
print('ocean temp + DIFF      : {}'.format(y2[N_TIME_STEPS - 1]))
print('atm temp without diff  : {}'.format(y3[N_TIME_STEPS - 1]))
print('atm temp + DIFF        : {}'.format(y4[N_TIME_STEPS - 1]))
# This 2nd row gives you the more realistic
# average temp as weights the boxes according to area
print(tseries_oa_mean_temp_area_weighted[N_TIME_STEPS - 1, :])


plt.figure(2)
plt.title('Model ocean and atmosphere temperature evolution over time')
x = np.arange('2001-09-21T12:00:00.0', my_date_london, dtype='datetime64[h]')

y5 = tseries_oa_mean_temp_area_weighted[0:-1, 0]
y6 = tseries_oa_mean_temp_area_weighted[0:-1, 1]
y7 = tseries_oa_mean_temp_area_weighted[0:-1, 2]
y8 = tseries_oa_mean_temp_area_weighted[0:-1, 3]


plt.plot(x, y5, 'b', linewidth=1.0, label='oceanNODIFF')
plt.plot(x, y6, 'b--', linewidth=1.0, label='oceanDIFF')
plt.plot(x, y7, 'r', linewidth=1.0, label='atmosNODIFF')
plt.plot(x, y8, 'r--', linewidth=1.0, label='atmosDIFF')
plt.legend(loc='lower right', prop={'size': 14})
plt.ylabel('Temperature (Degrees C)')
plt.xlabel('Time')
# plt.show()
show_plot()  # this is your function mcp!
plt.close(1)

# AAA= np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]])
# AAA=AAA[1:3,1:3]
# AAAA=   np.roll(AAA,1,axis=1)[1:N_LAT-1,0:N_LONG]
