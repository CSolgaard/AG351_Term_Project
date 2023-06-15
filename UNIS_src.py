
from pyproj import Transformer
import numpy as np 
from pathlib import Path
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from dataclasses import dataclass
import rasterio
import chaosmagpy as cp
from parula import parula_map
from datetime import datetime
from scipy.interpolate import interp1d,interp2d

def ll2utm(longitude, latitude, height, utmZone): 
    """ 
    Method using the pyproj python package, for transforming geodetic coordinates to UTM 

    Input: 
        latitude:               array of latitude values
        longitude:              array of longitude values 
        height:                 array of height (meters above ellipsoid)
        utmZone:                str ex: "31"(Svalbard) or "33"(Greenland) 


    Output: 
        UTM dataclass:          Easting, Northing [m]
    -----------------------
    Solgaard, 10/05-2023 
    """
    pipeline = "+ellps=GRS80 +proj=pipeline +step +proj=utm +zone=" + utmZone
    transform_object = Transformer.from_pipeline(pipeline)
    geodetic_corr = [longitude, latitude, height]
    UTM_corr = transform_object.transform(*geodetic_corr)

    @dataclass
    class UTM: 
        Northing: np.ndarray
        Easting: np.ndarray
        Elevation: np.ndarray
    UTM = UTM(UTM_corr[1], UTM_corr[0], UTM_corr[2])
    return UTM



def create_df(lowerlim, upperlim, dat4): 
    frame = {"Easting":dat4.Easting[lowerlim:upperlim], "Northing":dat4.Northing[lowerlim:upperlim], 
             "nT":dat4.nT[lowerlim:upperlim], "time":dat4.time[lowerlim:upperlim], 
             "elevation":dat4.UTM_elevation[lowerlim:upperlim]}

    df = pd.DataFrame(frame)
    return df


def dist_from_ref(easting_mes, easting_ref, northing_mes, northing_ref): 
    dist = np.sqrt((easting_ref - easting_mes)**2 + (northing_ref - northing_mes)**2)
    return dist


def find_intersection(p1, p2):
    # p1 and p2 are the polynomial functions of the two lines
    return np.roots(p1 - p2)


def get_sec(time): 
    """
    hh:mm:ss.s => SOD (Seconds Of Day)
    """
    SOD = []
    for i in range(len(time)): 
        SOD_value = int(time[i].split(":")[0])*3600+ int(time[i].split(":")[1])*60 + float(time[i].split(":")[2])
        SOD.append(SOD_value)

    return SOD


def dist_from_0(easting_mes, easting_ref, northing_mes, northing_ref): 
    dist = np.sqrt((easting_ref - easting_mes)**2 + (northing_ref - northing_mes)**2)

    if np.mean(np.diff(northing_mes)) < 0: 
        idx = (northing_mes > northing_ref)
        dist[idx] = dist * (-1)
    else: 
        idx = (northing_mes < northing_ref)
        dist[idx] = dist * (-1)
    return dist



def spatial_cut(df, X, spec): 
    """
    X:          Hyperparameter for optimizing spatial cutoff
    df:         Dataframe, needs to include a Easting, Northing and Time column 
    spec:       int: 0 if data are removed from start, 1 if data should be removed from back
    return:     
    df_sorted:  Dataframe with a cutoff at X value. 
    """
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    if spec == 0: 
        axs[0].plot(df.time, df.Easting, zorder = 1, label="Raw Data")
        axs[0].plot(df.time[X:], df.Easting[X:], zorder = 10, label="Spatial Sorted")
        axs[1].plot(df.time, df.Northing, zorder = 1, label="Raw Data")
        axs[1].plot(df.time[X:], df.Northing[X:], zorder = 10, label="Spatial Sorted")
        df_sorted = df.drop(df.index[:X])
    else: 
        axs[0].plot(df.time, df.Easting, zorder = 1, label="Raw Data")
        axs[0].plot(df.time[:X], df.Easting[:X], zorder = 10, label="Spatial Sorted")
        axs[1].plot(df.time, df.Northing, zorder = 1, label="Raw Data")
        axs[1].plot(df.time[:X], df.Northing[:X], zorder = 10, label="Spatial Sorted")
        df_sorted = df.drop(df.index[X:])
    
    axs[0].set_title('Easting VS. Time')
    axs[0].set_ylabel("Easting [m]")
    axs[0].set_xlabel("Time")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title('Northing VS. Time')
    axs[1].set_ylabel("Northing [m]")
    axs[1].set_xlabel("Time")
    axs[1].legend()
    axs[1].grid()
    plt.show()

    df_sorted.reset_index(drop=True, inplace=True)
    return df_sorted




def Import_tiff(file, ax): 
    src = rasterio.open(file)

    # Read the individual bands of the image
    band1 = src.read(1)  # Assuming you want to plot the first band
    band2 = src.read(2)  # Assuming you want to plot the second band
    band3 = src.read(3)  # Assuming you want to plot the third band

    # Create the RGB image by stacking the bands
    rgb_image = np.stack((band1, band2, band3), axis=-1)

    # Define the extent of the image using the source bounds
    extent = (src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top)

    ax.imshow(rgb_image, extent=extent)

    return rgb_image, extent


def time_convert(time_series, date_str): 
    """
    Input:          time_series: '%H:%M:%S.%f'
                    date_str:    'dd/mm/year'
    """
    # Define the date
    # date_str = '24/08/2018'
    date_format = '%d/%m/%Y'
    date = datetime.strptime(date_str, date_format).date()

    # Convert time series to formatted time strings
    formatted_times = []
    for time_value in time_series:
        time_str = str(int(time_value)).zfill(6)  # Convert to 6-digit string format
        hours = int(time_str[:2])
        minutes = int(time_str[2:4])
        seconds = int(time_str[4:6])
        microseconds = int((time_value % 1) * 1e6)  # Extract microseconds
        datetime_obj = datetime.combine(date, datetime.min.time())  # Create datetime object with date and midnight time
        datetime_obj = datetime_obj.replace(hour=hours, minute=minutes, second=seconds, microsecond=microseconds)
        formatted_time = datetime_obj.time().strftime('%H:%M:%S.%f')
        formatted_times.append(formatted_time)

    return formatted_times



def Calc_CHAOS(Chaos_path, df, date): 
    """
    Input: 
        Chaos_path:             Path to CHAOS.mat file, containing model parameters 
        df:                     Dataframe containing the lat and longitude of survey, needs to contain a 
                                column of Internal values (df.Internal)
        date:                   Date in format "dd/mm-year"

    Output: 
        Lithospheric:           Calculated Crustal anomaly. 
    """

    df["colat"] = 90 - df.lat

    radius = 6371 # km, earth radius
    theta = np.linspace(df.colat.min(), df.colat.max(), 1000)  # colatitude in degrees
    phi = np.linspace(df.lon.min(), df.lon.max(), 1000)  # longitude in degrees

    phi_grid, theta_grid = np.meshgrid(phi, theta)
    radius_grid = radius*np.ones(phi_grid.shape)

    year = int((date.split("/")[1]).split("-")[1])
    month = int((date.split("/")[1]).split("-")[0])
    day = int(str.split("/")[0])

    time = cp.data_utils.mjd2000(year, month, day)  # modified Julian date

    # load the CHAOS model
    model = cp.load_CHAOS_matfile(Chaos_path)

    # compute field components on the grid using the Gauss coefficients
    B_radius, B_theta, B_phi = model.synth_values_tdep(time, radius, theta, phi, grid=True,nmax=13)
    X = -B_theta
    Y = B_phi
    Z = -B_radius

    F_chaos = np.sqrt(X**2 + Y**2 + Z**2)

    plt.figure(figsize=(5,5))
    sc = plt.contourf(theta, phi, F_chaos, cmap=parula_map)
    plt.plot(df.colat, df.lon, "black", label="Survey Line")
    plt.xlabel(r'Longitude [$^{\circ}$]')
    plt.ylabel(r'Colatitude [$^{\circ}$]')
    plt.title('CHAOS Model, F [nT]')
    plt.grid()
    cbar = plt.colorbar(sc)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.set_label("[nT]")

    cbar.ax.yaxis.set_major_formatter(tick.FormatStrFormatter('%.1f'))
    plt.xticks(rotation=45)
    plt.gca().ticklabel_format(axis='both', style='plain', useOffset=False)
    plt.legend()
    plt.show()

    f_chaos = interp2d(theta,phi+90,F_chaos,'cubic')
    interp_chaos = []
    for lat,lon in zip(df.lat,df.lon):
        interp_chaos.extend(f_chaos(lon,lat))
    CHAOS = interp_chaos
    lithospheric = df.Internal - CHAOS

    return lithospheric
        


