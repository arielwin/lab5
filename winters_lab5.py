import os
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from rasterio.plot import show
from matplotlib import pyplot
import glob
import scipy

def slopeAspect(dem, cs):    
    """Calculates slope and aspect using the 3rd-order finite difference method    
    Parameters    
    ----------    
    dem : numpy array        
        A numpy array of a DEM    
    cs : float
        The cell size of the original DEM    
    Returns    
    -------    
    numpy arrays        
        Slope and Aspect arrays    
    """    
    from math import pi    
    from scipy import ndimage    
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])    
    dzdx = ndimage.convolve(dem, kernel, mode='mirror') / (8 * cs)    
    dzdy = ndimage.convolve(dem, kernel.T, mode='mirror') / (8 * cs)    
    slp = np.arctan((dzdx ** 2 + dzdy ** 2) ** 0.5) * 180 / pi    
    ang = np.arctan2(-dzdy, dzdx) * 180 / pi    
    aspect = np.where(ang > 90, 450 - ang, 90 - ang)    
    return slp, aspect

def reclassAspect(npArray):    
    """Reclassify aspect array to 8 cardinal directions (N,NE,E,SE,S,SW,W,NW),    
    encoded 1 to 8, respectively (same as ArcGIS aspect classes).    
    Parameters    
    ----------    
    npArray : numpy array        
    numpy array with aspect values 0 to 360    
    Returns    
    -------    
    numpy array
    numpy array with cardinal directions    
    """    
    return np.where((npArray > 22.5) & (npArray <= 67.5), 2,    
    np.where((npArray > 67.5) & (npArray <= 112.5), 3,   
    np.where((npArray > 112.5) & (npArray <= 157.5), 4,    
    np.where((npArray > 157.5) & (npArray <= 202.5), 5,
    np.where((npArray > 202.5) & (npArray <= 247.5), 6,  
    np.where((npArray > 247.5) & (npArray <= 292.5), 7,    
    np.where((npArray > 292.5) & (npArray <= 337.5), 8, 1)))))))

def reclassByHisto(npArray, bins):    
    """Reclassify np array based on a histogram approach using a specified   
    number of bins. Returns the reclassified numpy array and the classes from  
    the histogram.  
    
    Parameters  
    ---------- 
    npArray : numpy array    
        Array to be reclassified   
    bins : int     
        Number of bins  
    Returns  
    -------   
    numpy array   
    umpy array with reclassified values    
    """    
    histo = np.histogram(npArray, bins)[1]    
    rClss = np.zeros_like(npArray)    
    for i in range(bins):     
        rClss = np.where((npArray >= histo[i]) & (npArray <= histo[i + 1]),  
                         i + 1, rClss)    
    return rClss

dem_array = rasterio.open(r'data/bigElk_dem.tif').read(1) #open and read dem
slope, aspect = slopeAspect(dem_array, 30)                #get slope and aspect from slopeAspect function

aspect_reclass = reclassAspect(aspect)                    #reclassify the aspect
slope_reclass = reclassByHisto(slope,10)                  #reclassify the slope with 10 bins

fire_array = rasterio.open(r'data/fire_perimeter.tif').read(1) #open fire array
reds = glob.glob(r'data/L5_big_elk/*B3.tif')                   #get B3, red bands
nirs = glob.glob(r'data/L5_big_elk/*B4.tif')                   #get B4, Near Infrared Bands
h = np.where(fire_array == 2)                                  #get subset of fire area that is the healthy area
b = np.where(fire_array == 1)                                  #same but for burnded areas

means = [] #empty list for means
rr = []    #empty list for recovery ratios

years = ['2002', '2003', '2004', '2005', '2006',
         '2007', '2008', '2009', '2010', '2011', ] #list of years

for x,y in zip(reds,nirs):                 #for each tif in red,nir
    red = rasterio.open(x,'r').read(1)     #open and read red
    nir = rasterio.open(y, 'r').read(1)    #open and read nir
    ndvi = ((nir-red)/(nir+red))           #get ndvi using ndvi function
    ndvi_mean = ndvi[h].mean()             #get only the ndvi array for healthy areas
    recovery_ratio = ndvi/ndvi_mean        #get recovery ration
    burned_mean = recovery_ratio[b].mean() #get mean array of burned areas
    means.append(burned_mean)              #append
    flat = recovery_ratio.ravel()          #flatten recovery ratios for use in polyfit
    rr.append(flat)                        #append

stacked = np.vstack(rr)                       #stack all ten arrays
line = np.polyfit(range(10), stacked, 1)[0]   #run stacked arrays through polyfit to get line
lines = line.reshape(280,459)                 #reshape to same shape as original tif shapes
mc = np.where(fire_array == 1, lines, np.nan) #mean coefficient of only bunred areas

print('The mean coefficient of revoery for the period was', round(np.nanmean(mc),4)) #print mean of mc, round to 4 digits
print('')
for y,z in zip(years, means):
    print('For', y, 'the recovery ratio was', round(np.nanmean(z),5))#print rr for each year
    
def zonal_stats_table(zones, value_raster, csv_name): #function: YeeHaw!
    '''
    Function that takes an array of zones: like slope classes, or aspect classes and gets
    zonal stats based on the value raster.
    Outputs a csv
    
    '''
    mean_stats = [] #empty lists for collection
    max_stats = []
    min_stats = []
    count_stats = []
    std_stats = []
    zone = []
    
    for u in np.unique(zones):
        ras = np.where(zones==u, u, np.nan)                       #zone raster for only zones of number u
        min_stats.append(round(np.nanmin(ras * value_raster),5))  #minimum number in overlap
        max_stats.append(round(np.nanmax(ras * value_raster),5))  #maximum number in overlap
        mean_stats.append(round(np.nanmean(ras * value_raster),5))#mean of overlap
        std_stats.append(round(np.nanstd(ras * value_raster),5))  #standard deviation of overlap
        count_stats.append(np.where(zones == u, 1, 0).sum())      #get of cells equal to u
        zone.append(int(u))                                       #get a zone number
    
    stats = {'zone' : zone, 'min': min_stats, 'max': max_stats, 'mean': mean_stats, 'std': std_stats, 'count': count_stats} #diction for the data
    df = pd.DataFrame(stats) #make a dataframe
    df.to_csv(csv_name)      #turn df to csv
    return df                #return for later use
print('')
print(zonal_stats_table(slope_reclass, mc, "slope.csv"))
print('')
print(zonal_stats_table(aspect_reclass, mc, "aspect.csv"))


with rasterio.open(r'data/bigElk_dem.tif') as dataset:

    with rasterio.open(f'mean_coeff.tif' , 'w', 
                       driver='GTiff',
                       height=mc.shape[0],
                       width=mc.shape[1],
                       count=1,
                       dtype=mc.dtype,
                       crs=dataset.crs,
                       transform=dataset.transform,
                       nodta=dataset.nodata
                      ) as out_dataset:
        out_dataset.write(mc,1)
        
print('')
print('Unfortunately, my counts are incorrect for this statement. However, some conclusions can still be drawn from the zonal stats output.')
print('In the zonal stats table for slope we can see on slopes 7,8, and 9 we see the largest mean numbers. This menas that the recovery ratio')
print('here was higher than on other slope steepness.')
print('')
print('For the aspect output we can see that aspect 6,7,8 aslo had the highest mean recovery.')
print('This should mean that S,SW, NW directions are best for recovery, posiibly due to better sunlight.')
