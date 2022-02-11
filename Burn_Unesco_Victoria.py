# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:43:07 2018

@author: egli.michailidou
with modifications by k.miotlinski@ecu.edu.au

https://www.un-spider.org/advisory-support/recommended-practices/recommended-practice-burn-severity-mapping


"""

from osgeo import osr
from osgeo import ogr
from osgeo import gdal
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rasterio
import glob
from rasterio.plot import show
from rasterio.mask import mask
from shapely.geometry import mapping
import geopandas as gpd
import math  
from matplotlib_scalebar.scalebar import ScaleBar

import os


# Function definition
#https://scipy-lectures.org/intro/language/functions.html#function-definition 

def read_band_image(band, path):
    """
    This function takes as input the Sentinel-2 band name and the path of the 
    folder that the images are stored, reads the image and returns the data as
    an array
    input:   band           string            Sentinel-2 band name
             path           string            path of the folder
    output:  data           array (n x m)     array of the band image
             spatialRef     string            projection 
             geoTransform   tuple             affine transformation coefficients
             targetprj                        spatial reference
    """
    a = path+'*B'+band+'*.jp2'
    img = gdal.Open(glob.glob(a)[0])
    data = np.array(img.GetRasterBand(1).ReadAsArray())
    spatialRef = img.GetProjection()
    geoTransform = img.GetGeoTransform()
    targetprj = osr.SpatialReference(wkt = img.GetProjection())
    return data, spatialRef, geoTransform, targetprj



def reproject_shp_gdal(infile, outfile, targetprj):
    """
    This function takes as input the input and output file names and the projection
    in which the input file will be reprojected and reprojects the input file using
    gdal
    input:  infile     string      input filename
            outfile    string      output filename
            targetprj              projection (output of function read_band_image)
    """
    ## reprojection with gdal 
    
    driver = ogr.GetDriverByName("ESRI Shapefile") 
    dataSource = driver.Open(infile, 1) # 0 means read-only. 1 means writeable.
    layer = dataSource.GetLayer()
    sourceprj = layer.GetSpatialRef()
    transform = osr.CoordinateTransformation(sourceprj, targetprj)
    
    # Create the output shapefile
    outDriver = ogr.GetDriverByName("Esri Shapefile")
    outDataSource = outDriver.CreateDataSource(outfile)
    outlayer = outDataSource.CreateLayer('', targetprj, ogr.wkbPolygon)
    outlayer.CreateField(ogr.FieldDefn('id', ogr.OFTInteger))
    
    #Iterate over Features
    i = 0
    for feature in layer:
        transformed = feature.GetGeometryRef()
        transformed.Transform(transform) #reproject geometry

        geom = ogr.CreateGeometryFromWkb(transformed.ExportToWkb()) # create geometry from wkb (write geometry of reprojected geometry)
        defn = outlayer.GetLayerDefn() #layer definition
        feat = ogr.Feature(defn)  #create new feature
        feat.SetField('id', i) #set id
        feat.SetGeometry(geom) #set geometry
        outlayer.CreateFeature(feat) 
        i += 1
        feat = None
        
def array2raster(array, geoTransform, projection, filename):
    """ 
    This function tarnsforms a numpy array to a geotiff projected raster
    input:  array                       array (n x m)   input array
            geoTransform                tuple           affine transformation coefficients
            projection                  string          projection
            filename                    string          output filename
    output: dataset                                     gdal raster dataset
            dataset.GetRasterBand(1)                    band object of dataset
    
    """
    pixels_x = array.shape[1]
    pixels_y = array.shape[0]
    
    driver = gdal.GetDriverByName('GTiff')
    dataset = driver.Create(
        filename,
        pixels_x,
        pixels_y,
        1,
        gdal.GDT_Float64, )
    dataset.SetGeoTransform(geoTransform)
    dataset.SetProjection(projection)
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()  # Write to disk.
    return dataset, dataset.GetRasterBand(1)  #If you need to return, remenber to return  also the dataset because the band don`t live without dataset.
 
def clip_raster(filename, shp):
    """
    This function clips a raster based on a shapefile
    input:  filename          string                input raster filename
            shp               dataframe             input shapefile open with geopandas
    output: clipped           array (1 x n x m)     clipped array 
            clipped_meta      dict                  metadata
            cr_ext            tuple                 extent of clipped data
            gt                tuple                 affine transformation coefficients
    """
    inraster = rasterio.open(filename)
    
    extent_geojson = mapping(shp['geometry'][0])
    clipped, crop_affine = mask(inraster, 
                                shapes=[extent_geojson], 
                                nodata = np.nan,
                                crop=True)
    clipped_meta = inraster.meta.copy()
    # Update the metadata to have the new shape (x and y and affine information)
    clipped_meta.update({"driver": "GTiff",
                 "height": clipped.shape[0],
                 "width": clipped.shape[1],
                 "transform": crop_affine})
    cr_ext = rasterio.transform.array_bounds(clipped_meta['height'], 
                                            clipped_meta['width'], 
                                            clipped_meta['transform'])
    
    # transform to gdal
    gt = crop_affine.to_gdal()
    
    return clipped, clipped_meta, cr_ext, gt
    

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

def dnbr(nbr1,nbr2):
    """
    This function takes as input the pre- and post-fire NBR and returns the dNBR
    input:  nbr1     array (n x m)       pre-fire NBR
            nbr2     array (n x m)       post-fire NBR
    output: dnbr     array (n x m)       dNBR
    """
    dnbr = nbr1 - nbr2
    return dnbr


def rdnbr(dnbr,nbr1):
    """
    This function takes as input the pre- and post-fire NBR and returns the dNBR
    input:  nbr1     array (n x m)       pre-fire NBR
            nbr2     array (n x m)       post-fire NBR
    output: rdnbr     array (n x m)      RdNBR
    """
    rdnbr = (dnbr - np.mean(nbr1)) / np.sqrt(np.abs(nbr1))
    return rdnbr


def reclassify(array):
    """
    This function reclassifies an array
    input:  array           array (n x m)    input array
    output: reclass         array (n x m)    reclassified array
    """
    reclass = np.zeros((array.shape[0],array.shape[1]))
    for i in range(0,array.shape[0]):
        for j in range(0,array.shape[1]):
            if math.isnan(array[i,j]):
                reclass[i,j] = np.nan
            elif array[i,j] < 0.1:
                reclass[i,j] = 1
            elif array[i,j] < 0.27:
                 reclass[i,j] = 2
            elif array[i,j] < 0.44:
                 reclass[i,j] = 3
            elif array[i,j] < 0.66:
                 reclass[i,j] = 4
            else:
                reclass[i,j] = 5
                
    return reclass
                     

#Source data
#prefire
prefire_path = os.path.join(os.path.expanduser("~/"),
                      'OneDrive - Edith Cowan University',
                      'PIFWaQ',
                      'empirical',
                      '0_data',
                      'sentinel',
                      'S2B_MSIL1C_20220126T021339_N0400_R060_T50HMK_20220126T053615.SAFE', 
                      'GRANULE',
                      'L1C_T50HMK_A025540_20220126T021913',
                      'IMG_DATA/')

#post-fire
postfire_path = os.path.join(os.path.expanduser("~/"),
                      'OneDrive - Edith Cowan University',
                      'PIFWaQ',
                      'empirical',
                      '0_data',
                      'sentinel',
                      'S2B_MSIL1C_20220205T021339_N0400_R060_T50HMK_20220205T053331.SAFE', 
                      'GRANULE',
                      'L1C_T50HMK_A025683_20220205T022541',
                      'IMG_DATA/')



# Define shapefile for clipping 
infile_shp = "/Users/kmiotlin/Library/CloudStorage/OneDrive-EdithCowanUniversity/PIFWaQ/explorative/Victoria_wildfire2022/gis/Burnt_site_Victoria_Feb2022.shp"
# Define reprojected shapefile
outfile_shp = "/Users/kmiotlin/Desktop/Victoria_wildfire_Feb2022/projected.shp"
# name of the output dNBR raster file
filename = "/Users/kmiotlin/Desktop/Victoria_wildfire_Feb2022/dNBR.tiff"
# name of clipped dNBR raster
filename2 = "/Users/kmiotlin/Desktop/Victoria_wildfire_Feb2022/dNBR_clipped.tiff"
# name of the output dNBR raster file
filename5 = "/Users/kmiotlin/Desktop/Victoria_wildfire_Feb2022/RdNBR.tiff"
# name of clipped dNBR raster
filename6 = "/Users/kmiotlin/Desktop/Victoria_wildfire_Feb2022/RdNBR_clipped.tiff"

# path to save figure
fname1 = "/Users/kmiotlin/Desktop/dnbr_map.png"
fname3 = '/Users/kmiotlin/Desktop/Rdnbr_map.png'

# Sentinel-2 Bands used for NBR calculation 
band1 = '8A'
band2 = '12'

    
# Read the pre-fire band images 
(pre_fire_b8a, crs, geoTransform, targetprj) = read_band_image(band1, prefire_path)
(pre_fire_b12, crs, geoTransform, targetprj) = read_band_image(band2, prefire_path)
    
# Calculation of pre-fire NBR
pre_fire_nbr = nbr(pre_fire_b8a.astype(int),pre_fire_b12.astype(int))

# Read the post-fire band images
(post_fire_b8a, crs, geoTransform, targetprj) = read_band_image(band1, postfire_path)
(post_fire_b12, crs, geoTransform, targetprj) = read_band_image(band2, postfire_path)
    
# Calculation of post-fire NBR
post_fire_nbr = nbr(post_fire_b8a.astype(int),post_fire_b12.astype(int))
    
# Calculation of dNBR
DNBR = dnbr(pre_fire_nbr,post_fire_nbr)
    

# Reprojection of shapefile with gdal to match projection of Sentinel-2 images
reproject_shp_gdal(infile_shp, outfile_shp, targetprj)
    
# Read the reprojected shapefile
fire_boundary = gpd.read_file(outfile_shp)
    
# project dNBR to images projection
dnbr_tif, dnbr_tifBand = array2raster(DNBR, geoTransform, crs, filename)
    
# clip raster dNBR file to shapefile
(clipped_dnbr, clipped_dnbr_meta, cr_extent, gt) = clip_raster(filename, fire_boundary)
clipped_ds , clipped_ds_rasterband = array2raster(clipped_dnbr[0], gt, crs, filename2)


# Calculation of RdNBR
RdNBR = rdnbr(pre_fire_nbr,post_fire_nbr)
    
    
# project dNBR to images projection
rdnbr_tif, rdnbr_tifBand = array2raster(RdNBR, geoTransform, crs, filename5)
    
# clip raster dNBR file to shapefile
(clipped_rdnbr, clipped_rdnbr_meta, cr_extent, gt) = clip_raster(filename, fire_boundary)
clipped_rdnbr , clipped_rdndb_rasterband = array2raster(clipped_rdnbr[0], gt, crs, filename6)





#set colors for plotting and classes
cmap = matplotlib.colors.ListedColormap(['green','yellow','orange','red','purple'])
cmap.set_over('purple')
cmap.set_under('white')
bounds = [-0.5, 0.1, 0.27, 0.440, 0.660, 1.3]        
norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)  

#sns.set_style("whitegrid")

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})
cax = ax.imshow(clipped_ds_rasterband.ReadAsArray(), cmap=cmap, norm = norm)
plt.title('Burn Severity - Victoria - Feb 2022')
cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04, ticks=[-0.2, 0.18, 0.35, 0.53, 1])
cbar.ax.set_yticklabels(['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity'])
scalebar = ScaleBar(20, length_fraction=0.25)
ax.add_artist(scalebar)
plt.savefig(fname1, bbox_inches="tight")
plt.show() 

# calculate burnt area (pixel size 20m*20m)
reclass = reclassify(clipped_ds_rasterband.ReadAsArray())
k = ['Unburned hectares', 'Low severity hectares', 'Moderate-low severity hectares', 'Moderate-high severity hectares', 'High severity']
for i in range(1,6):
    x = reclass[reclass == i]
    l= x.size*0.04
    print("%s: %.2f" % (k[i-1], l))
        





fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})
cax = ax.imshow(clipped_rdndb_rasterband.ReadAsArray(), cmap=cmap, norm = norm)
plt.title('Relative Normalized Burn Ratio (RdNBR) - Victoria - Feb 2022')
cbar = fig.colorbar(cax, ax=ax, fraction=0.035, pad=0.04, ticks=[-0.2, 0.18, 0.35, 0.53, 1])
cbar.ax.set_yticklabels(['Unburned', 'Low Severity', 'Moderate-low Severity', 'Moderate-high Severity', 'High Severity'])
scalebar = ScaleBar(20, length_fraction=0.25)
ax.add_artist(scalebar)
#ax.annotate(r'$RdNBR=\frac{864.7 \pm 21 nm}{2202.4 \pm 175 nm}$',
  #          xy=(0, 40), xycoords='data',
   #         xytext=(25, 25), textcoords='offset points', fontsize=16)
plt.savefig(fname3, bbox_inches="tight")
plt.show() 

# calculate burnt area (pixel size 20m*20m)
reclass = reclassify(clipped_rdndb_rasterband.ReadAsArray())
k = ['Unburned hectares', 'Low severity hectares', 'Moderate-low severity hectares', 'Moderate-high severity hectares', 'High severity']
for i in range(1,6):
    x = reclass[reclass == i]
    l= x.size*0.04
    print("%s: %.2f" % (k[i-1], l))


