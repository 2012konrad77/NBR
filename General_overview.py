#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 7 Feb 2022

@author: kmiotlin


https://gisgeography.com/sentinel-2-bands-combinations/


"""

import os
import matplotlib.pyplot as plt
from osgeo import gdal
import rasterio
import numpy as np

prefire_path = os.path.join(os.path.expanduser("~/"),
                      'OneDrive - Edith Cowan University',
                      'PIFWaQ',
                      'empirical',
                      '0_data',
                      'sentinel',
                      'S2B_MSIL1C_20220126T021339_N0400_R060_T50HMK_20220126T053615.SAFE', 
                      'GRANULE',
                      'L1C_T50HMK_A025540_20220126T021913',
                      'IMG_DATA')


postfire_path = os.path.join(os.path.expanduser("~/"),
                      'OneDrive - Edith Cowan University',
                      'PIFWaQ',
                      'empirical',
                      '0_data',
                      'sentinel',
                      'S2B_MSIL1C_20220205T021339_N0400_R060_T50HMK_20220205T053331.SAFE', 
                      'GRANULE',
                      'L1C_T50HMK_A025683_20220205T022541',
                      'IMG_DATA')


#1.1
os.chdir(postfire_path)
wfile = 'T50HMK_20220205T021339_B12.jp2'
ds = gdal.Open(wfile)
data = ds.GetRasterBand(1).ReadAsArray()
plt.imshow(data)
plt.show()





# Open the prefire input rasters
pre_2 = 'T50HMK_20220126T021339_B0{}.jp2'.format(2)
pre_3 = 'T50HMK_20220126T021339_B0{}.jp2'.format(3)
pre_4 = 'T50HMK_20220126T021339_B0{}.jp2'.format(4)
pre_5 = 'T50HMK_20220126T021339_B0{}.jp2'.format(5)
pre_6 = 'T50HMK_20220126T021339_B0{}.jp2'.format(6)
pre_7 = 'T50HMK_20220126T021339_B0{}.jp2'.format(7)
pre_8 = 'T50HMK_20220126T021339_B0{}.jp2'.format(8) #10 m
pre_8A = 'T50HMK_20220126T021339_B8A.jp2' #20 m 
pre_11 = 'T50HMK_20220126T021339_B{}.jp2'.format(11)
pre_12 = 'T50HMK_20220126T021339_B{}.jp2'.format(12)



# Open the postfire input rasters
post_2 = 'T50HMK_20220205T021339_B0{}.jp2'.format(2)
post_3 = 'T50HMK_20220205T021339_B0{}.jp2'.format(3)
post_4 = 'T50HMK_20220205T021339_B0{}.jp2'.format(4)
post_5 = 'T50HMK_20220205T021339_B0{}.jp2'.format(5)
post_6 = 'T50HMK_20220205T021339_B0{}.jp2'.format(6)
post_7 = 'T50HMK_20220205T021339_B0{}.jp2'.format(7)
post_8 = 'T50HMK_20220205T021339_B0{}.jp2'.format(8) #10m
post_8A = 'T50HMK_20220205T021339_B8A.jp2' #20 m
post_11 = 'T50HMK_20220205T021339_B{}.jp2'.format(11)
post_12 = 'T50HMK_20220205T021339_B{}.jp2'.format(12)







#Rasterio 
os.chdir(prefire_path)
os.listdir()
dataset1 = rasterio.open(pre_8)


os.chdir(postfire_path)
os.listdir()
dataset2 = rasterio.open(post_8)


print('Number of bands:',dataset2.count)
dataset2.width
dataset2.height
dataset2.bounds
dataset2.crs








def get_overview_data(fn, band_index=1, level=-1):
    """Returns an array containing data from an overview.

    fn         - path to raster file
    band_index - band number to get overview for
    level      - overview level, where 1 is the highest resolution;
                 the coarsest can be retrieved with -1
    """
    ds = gdal.Open(fn)
    band = ds.GetRasterBand(band_index)
    if level > 0:
        ov_band = band.GetOverview(level)
    else:
        num_ov = band.GetOverviewCount()
        ov_band = band.GetOverview(num_ov + level)
    return ov_band.ReadAsArray()


#1.2
data = get_overview_data(wfile)
data = np.ma.masked_equal(data, 0)
plt.imshow(data, cmap = 'jet')
plt.show()
#1.3
mean = np.mean(data)
std_range = np.std(data) * 2
plt.imshow(data, cmap = 'jet', vmin = mean - std_range, vmax = mean+std_range)
plt.show()


#code data stretching

def stretch_data(data, num_stddev):
    """Returns the data with a standard deviation stretch applied.

    data       - array containing data to stretch
    num_stddev - number of standard deviations to use
    """
    mean = np.mean(data)
    std_range = np.std(data) * 2
    new_min = max(mean - std_range, np.min(data))
    new_max = min(mean + std_range, np.max(data))
    clipped_data = np.clip(data, new_min, new_max)
    normalised_data = (clipped_data - np.min(clipped_data))/np.ptp(clipped_data)
    return normalised_data

#1.4
data = get_overview_data(wfile)
data = np.ma.masked_equal(data, 0)
plt.imshow(stretch_data(data, 1), cmap = 'jet')
plt.colorbar()
plt.show()


#plot rectange over the area of interest
from matplotlib.transforms import Bbox
left, bottom, width, height = (600, 2350, 300, 200)
rect = plt.Rectangle((left, bottom), width, height,
                     fill=False)

fig, ax = plt.subplots()
ax.add_patch(rect)

bbox = Bbox.from_bounds(left, bottom, width, height)
data = ds.GetRasterBand(1).ReadAsArray()
plt.imshow(data, cmap='jet')

plt.show()




#https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
#rasterio.windows.Window(col_off, row_off, width, height)
window = rasterio.windows.Window(600, 2350, 300, 200)
window10 = rasterio.windows.Window(600*2, 2350*2, 300*2, 200*2)
#prefire
os.chdir(prefire_path)
with rasterio.open(pre_2) as src:
    subset_pre_2 = src.read(1, window=window10)
    
with rasterio.open(pre_3) as src:
    subset_pre_3 = src.read(1, window=window10)

with rasterio.open(pre_4) as src:
    subset_pre_4 = src.read(1, window=window10)

with rasterio.open(pre_5) as src:
    subset_pre_5 = src.read(1, window=window)
        
with rasterio.open(pre_6) as src:
    subset_pre_6 = src.read(1, window=window)
    
with rasterio.open(pre_7) as src:
    subset_pre_7 = src.read(1, window=window)

with rasterio.open(pre_8) as src:
    subset_pre_8 = src.read(1, window=window10)

with rasterio.open(pre_8A) as src:
    subset_pre_8A = src.read(1, window=window)
    
with rasterio.open(pre_11) as src:
    subset_pre_11 = src.read(1, window=window)
    
with rasterio.open(pre_12) as src:
    subset_pre_12 = src.read(1, window=window)
    

#postfire
os.chdir(postfire_path)
with rasterio.open(post_2) as src:
    subset_post_2 = src.read(1, window=window10)
    
with rasterio.open(post_3) as src:
    subset_post_3 = src.read(1, window=window10)

with rasterio.open(post_4) as src:
    subset_post_4 = src.read(1, window=window10)

with rasterio.open(post_5) as src:
    subset_post_5 = src.read(1, window=window)
    
with rasterio.open(post_6) as src:
    subset_post_6 = src.read(1, window=window)
    
with rasterio.open(post_7) as src:
    subset_post_7 = src.read(1, window=window)

with rasterio.open(post_8) as src:
    subset_post_8 = src.read(1, window=window10)

with rasterio.open(post_8A) as src:
    subset_post_8A = src.read(1, window=window)

with rasterio.open(post_11) as src:
    subset_post_11 = src.read(1, window=window)

with rasterio.open(post_12) as src:
    subset_post_12 = src.read(1, window=window)



#Stretch range to visualize the ranges

#2.1
plt.figure(figsize=(10,8.5))
plt.imshow(subset_post_8, vmin = subset_pre_8.mean()-2*(subset_pre_8.std()), vmax = subset_pre_8.mean()+2*(subset_pre_8.std()))
plt.colorbar(shrink=0.5)
plt.title(f'Band NIR Subset\n{window}, raw data')
plt.xticks([])
plt.yticks([])

#auto strech and normalise
#2.2 
plt.figure(figsize=(10,8.5))
plt.imshow(stretch_data(subset_post_8, 2))
plt.colorbar(shrink=0.5)
plt.title(f'Band NIR Subset\n{window}, stretched data')
plt.xticks([])
plt.yticks([])



#Histograms
#see details on: https://matplotlib.org/stable/gallery/pyplots/pyplot_text.html#sphx-glr-gallery-pyplots-pyplot-text-py


bins = 50

plt.figure(figsize=(10,8.5))
plt.title("Histogram of 8A Sentinel-2 data for a selected region")
plt.hist(subset_pre_8A.flatten(), bins = bins, alpha=0.5, label = 'Pre-fire')
plt.hist(subset_post_8A.flatten(), bins = bins, alpha = 0.5, label = 'Post-fire')
plt.legend()
plt.show()


pre_date = '20220126'
post_date = '20220205'



#Test with normalisation
fig, axes = plt.subplots(2,2, figsize=(10,8), sharex=True, sharey=True)

plt.sca(axes[0,0])
plt.imshow(stretch_data(subset_pre_8A,2), cmap='viridis')
plt.colorbar(shrink=0.2)
plt.title('Band 8A: {}'.format(pre_date))

plt.sca(axes[0,1])
plt.imshow(stretch_data(subset_post_8A,2), cmap='viridis')
plt.colorbar(shrink=0.2)
plt.title('Band 8A: {}'.format(post_date))


plt.sca(axes[1,0])
plt.imshow(stretch_data(subset_pre_12,2), cmap='gist_yarg')
plt.colorbar(shrink=0.2)
plt.title('Band 12: {}'.format(pre_date))

plt.sca(axes[1,1])
plt.imshow(stretch_data(subset_post_12,2), cmap='gist_yarg')
plt.colorbar(shrink=0.2)
plt.title('Band 12: {}'.format(post_date))

plt.tight_layout()
plt.xticks([])
plt.yticks([])
fig.suptitle('Sentinel-2 normalised datasets', fontsize=16)
plt.savefig('/Users/kmiotlin/Desktop/NIR_fine_SWIR_Victoria.png', dpi=200)
plt.show()





fig, axes = plt.subplots(1,3, figsize=(10,4), sharex=True, sharey=True)

plt.sca(axes[0])
data = subset_pre_8A.astype(int)
plt.imshow(data, vmin = data.mean()-2*(data.std()), vmax = data.mean()+2*(data.std()))
plt.colorbar(shrink=0.2)
plt.title('Band 8 on {}'.format(pre_date))

plt.sca(axes[1])
data_post = subset_post_8A.astype(int)
plt.imshow(data_post, vmin = data.mean()-2*(data.std()), vmax = data.mean()+2*(data.std()))
plt.colorbar(shrink=0.2)
plt.title('Band 8 on {}'.format(post_date))

plt.sca(axes[2])
plt.imshow(data_post - data, cmap = 'RdBu')
plt.colorbar(shrink=0.2)
plt.title('$\Delta$ {} - {}'.format(post_date, pre_date))


plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.savefig('/Users/kmiotlin/Desktop/NIR.png', dpi=200)
plt.show()




fig, axes = plt.subplots(1,3, figsize=(10,4), sharex=True, sharey=True)

plt.sca(axes[0])
data = subset_pre_12.astype(int)
plt.imshow(data, vmin = data.mean()-2*(data.std()), vmax = data.mean()+2*(data.std()), cmap='Wistia')
plt.colorbar(shrink=0.2)
plt.title('Band 12 on {}'.format(pre_date))

plt.sca(axes[1])
data_post = subset_post_12.astype(int)
plt.imshow(data_post, vmin = data.mean()-2*(data.std()), vmax = data.mean()+2*(data.std()), cmap='Wistia')
plt.colorbar(shrink=0.2)
plt.title('Band 12 on {}'.format(post_date))

plt.sca(axes[2])
plt.imshow(data_post - data, cmap = 'Blues', vmin=0)
plt.colorbar(shrink=0.2)
plt.title('$\Delta$ {} - {}'.format(post_date, pre_date))

plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.savefig('/Users/kmiotlin/Desktop/SWIR.png', dpi=100)
plt.show()



def ndvi(band1, band2):
    """
    This function takes an input the arrays of the bands from the read_band_image
    function and returns the NDVI
    input:  band1   array (n x m)      array of first band image e.g B8
            band2   array (n x m)      array of second band image e.g. B4
    output: ndvi     array (n x m)      normalized burn ratio
    """
    band2 = np.ma.masked_where(band1 + band2 == 0, band2)    
    ndvi = (band1 - band2) / (band1 + band2)
    ndvi = ndvi.filled(-99)
    return ndvi



fig, axes = plt.subplots(1,3, figsize=(14,6), sharex=True, sharey=True)
plt.sca(axes[0])
data = (subset_pre_8 - subset_pre_4) / (subset_pre_8 + subset_pre_4).astype(float)
plt.imshow(data, vmin = data.mean()-2*(data.std()), 
                      vmax = data.mean()+2*(data.std()), cmap='YlGn')
plt.colorbar(shrink=0.2)
plt.title('NDVI {}'.format(pre_date))
plt.sca(axes[1])
data_post = (subset_post_8 - subset_post_4) / (subset_post_8 + subset_post_4).astype(float)
plt.imshow(data_post, vmin = data.mean()-2*(data.std()), 
                      vmax = data.mean()+2*(data.std()), cmap='YlGn')
plt.colorbar(shrink=0.2)
plt.title('NDVI {}'.format(post_date))
plt.sca(axes[2])
diff = (data_post - data)
plt.imshow(diff, vmin=-1, vmax=0, cmap='Oranges_r')
plt.colorbar(shrink=0.2)
plt.title('$\Delta$ ({} - {})'.format(post_date, pre_date))
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.savefig('/Users/kmiotlin/Desktop/NDVI.png', dpi=200)
plt.show()



def gndvi(band1, band2):
    """
    This function takes an input the arrays of the bands from the read_band_image
    function and returns the NDVI
    input:  band1   array (n x m)      array of first band image e.g B8
            band2   array (n x m)      array of second band image e.g. B3
    output: nbr     array (n x m)      normalized burn ratio
    """
    band2 = np.ma.masked_where(band1 + band2 == 0, band2)    
    gndvi = (band1 - band2) / (band1 + band2)
    gndvi = gndvi.filled(-99)
    return gndvi

pre_gndvi = gndvi(subset_pre_8, subset_pre_3)
post_gndvi = gndvi(subset_post_8, subset_post_3)
dgndvi = post_gndvi - pre_gndvi


fig, axes = plt.subplots(1,3, figsize=(14,6), sharex=True, sharey=True)
plt.sca(axes[0])
plt.imshow(pre_gndvi, cmap='YlGn', vmin = 0, vmax = 1)
plt.colorbar(shrink=0.2)
plt.title('GNDVI {}'.format(pre_date))
plt.sca(axes[1])
plt.imshow(post_gndvi, cmap='YlGn', vmin = 0, vmax = 1)
plt.colorbar(shrink=0.2)
plt.title('GNDVI {}'.format(post_date))
plt.sca(axes[2])
plt.imshow(dgndvi, cmap='RdBu', vmin=-.5, vmax=.5,
                             interpolation='none')
plt.colorbar(shrink=0.2)
plt.title('$\Delta$ ({} - {})'.format(post_date, pre_date))
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.show()






def nbr(band1, band2):
    """
    This function takes an input the arrays of the bands from the read_band_image
    function and returns the Normalized Burn ratio (NBR)
    input:  band1   array (n x m)      array of first band image e.g B8A
            band2   array (n x m)      array of second band image e.g. B12
    output: nbr     array (n x m)      normalized burn ratio
    """
    
    band2 = np.ma.masked_where(band1 + band2 == 0, band2)
    nbr = (band1 - band2) / (band1 + band2)
    nbr = nbr.filled(-99)
    return nbr



pre_nbr = nbr(subset_pre_8A, subset_pre_12)
post_nbr = nbr(subset_post_8A, subset_post_12)
dnbr = post_nbr - pre_nbr



def rdnbr(dnbr, pre_nbr):
    '''
    Cansley and McKenzie - Remote Sens.2012, 4, 456-483

    Parameters
    ----------
    dnbr : difference NBR between pre- and post-fire
    pre_nbr : pre-fire NBR

    Returns
    -------
    rdnbr : Relative differenced Normalized Burn Ratio

    '''
    nbr_offset = np.mean(pre_nbr)
    rdnbr = (dnbr - nbr_offset) / np.sqrt(np.abs(pre_nbr/1000))
    return rdnbr

drdnbr = rdnbr(dnbr, pre_nbr)


def csi(band1, band2):
    """
    This function takes an input the arrays of the bands from the read_band_image
    function and returns the Char Soil Index (CSI)
    input:  band1   array (n x m)      array of first band image e.g B8A
            band2   array (n x m)      array of second band image e.g. B12
    output: csi     array (n x m)      char soil index
    """
    csi = band1 / band2
    return csi

pre_csi = csi(subset_pre_8A, subset_pre_12)
post_csi = csi(subset_post_8A, subset_post_12)
dcsi = pre_csi - post_csi

fig, axes = plt.subplots(1,3, figsize=(14,6), sharex=True, sharey=True)
plt.sca(axes[0])
plt.imshow(pre_csi, cmap='bone_r', vmin=0, vmax = 12)
plt.colorbar(shrink=0.2)
plt.title('8A/12 {}'.format(pre_date))
plt.sca(axes[1])
plt.imshow(post_csi, cmap='bone_r', vmin=0, vmax = 12)
plt.colorbar(shrink=0.2)
plt.title('8A/12 {}'.format(post_date))
plt.sca(axes[2])
plt.imshow(dcsi, cmap='bone_r', interpolation='none')
plt.colorbar(shrink=0.2)
plt.title('$\Delta$ ({} - {})'.format(post_date, pre_date))
plt.annotate(r'$CSI =  \frac{8A_{pre}}{12_{pre}} - \frac{8A_{post}}{12_{post}}$',
            xy=(0, 55), xycoords='data',
            xytext=(15, 25), textcoords='offset points', fontsize=16)
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.savefig('/Users/kmiotlin/Desktop/CSI.png', dpi=200)
plt.show()


fig, axes = plt.subplots(figsize=(14,6))
plt.imshow(dcsi, cmap='bone_r', interpolation='none',)
plt.colorbar(shrink=0.2)
plt.title('$\Delta$ ({} - {})'.format(post_date, pre_date))
plt.annotate(r'$CSI =  \frac{8A_{pre}}{12_{pre}} - \frac{8A_{post}}{12_{post}}$',
            xy=(0, 55), xycoords='data',
            xytext=(15, 25), textcoords='offset points', fontsize=16)
plt.tight_layout()
plt.xticks([])
plt.yticks([])
#plt.savefig('/Users/kmiotlin/Desktop/CSI.png', dpi=200)
plt.show()





