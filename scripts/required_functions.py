
# import required libraries
import rasterio as rs
from rasterio.warp import Resampling
import pandas as pd
import numpy as np
import rioxarray
import numpy.matlib
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.mask import mask
from shapely.geometry import shape, mapping
from pathlib import Path

# function to resample geotiffs
def tiff_resampler(input_tif,  # input tiff (string)
                   target_resolution,  # target x and y cell resolution (tuple)
                   resampling_method,  # choose rs resampling method (string)
                   output_name):  # output tiff (string)
    with rs.open(input_tif) as src:

        transform, width, height = rs.warp.aligned_target(
            src.transform,
            src.width,
            src.height,
            target_resolution)

        data, transform = rs.warp.reproject(source=src.read(masked=True),
                                            destination=np.zeros(
                                                (src.count, height, width)),
                                            src_transform=src.transform,
                                            dst_transform=transform,
                                            src_crs=src.crs,
                                            dst_crs=src.crs,
                                            dst_nodata=src.nodata,
                                            resampling=Resampling[resampling_method])
        profile = src.profile
        profile.update(transform=transform, driver='GTiff',
                       height=data.shape[1], width=data.shape[2])

        with rs.open(output_name, 'w', **profile) as dst:
            dst.write(data)

# function to create binary raster based on refugia threshold
def binary_converter(input_tif,  # input tif (string)
                     filepath,  # string + /
                     threshold,  # minimum value (integer)
                     output_name):  # specify output name (string)
    with rs.open(filepath / input_tif) as src:
        data = src.read()  # read the geotiff
        profile = src.profile  # get metadata of geotiff

    output_data = np.where(data >= threshold, 1,
                           np.where(data < threshold, 0, data))

    profile.update(dtype=rs.float32)  # update metadata
    with rs.open(filepath / output_name, 'w', **profile) as dst:  # create and write output file
        dst.write(output_data.astype(profile['dtype']))

# function to convert nc to geotiff
def nc_geotiff_converter(input_nc,  # string
                         filepath,  # string + /
                         variable,  # string
                         time_band,  # options: 0-9 -> 2010-2100 (10y steps)
                         output_name):  # output tiff (string)
    nc_file = rioxarray.open_rasterio(filepath / input_nc, decode_times=False)
    data_array = nc_file[variable][time_band]
    data_array.rio.to_raster(filepath / output_name, driver='GTiff')
    # update meta data with rasterio to avoid later conflicts
    with rs.open(filepath / output_name) as src:  # specify file name
        data = src.read(1)  # assuming single band image
        profile = src.profile
        profile.update(dtype=rs.float32)

    with rs.open(filepath / output_name, 'w', **profile) as dst:
        dst.write(data, 1)


# area calculation per raster cell in WGS84
def land_area_calculation(filepath, input_name, output_name=None):
    """
    Function to calc land area for each raster cell in WGS84 without reprojecting
    Adapted from:
    https://gis.stackexchange.com/questions/317392/determine-area-of-cell-in-raster-qgis
    """
    with rs.open(filepath / input_name) as src:
        input_raster = src.read(1)
        input_raster = np.nan_to_num(input_raster, nan=-3.40282e+38)
        profile = src.profile
        gt = src.transform
        pix_width = gt[0]
        ulY = gt[5]  # upper left y
        rows = src.height
        cols = src.width
        lrY = ulY + gt[4] * rows  # lower right y

    lats = np.linspace(ulY, lrY, rows+1)

    a = 6378137  # semi-major axis for WGS84
    b = 6356752.314245179  # semi-minor axis WGS84
    lats = lats * np.pi/180  # degrees to radians
    e = np.sqrt(1-(b/a)**2)
    sinlats = np.sin(lats)
    zm = 1 - e * sinlats
    zp = 1 + e * sinlats
    q = pix_width/360  # distance between meridians

    # compute areas for each latitude
    areas_to_equator = np.pi * b**2 * \
        ((2*np.arctanh(e*sinlats) / (2*e) + sinlats / (zp*zm))) / 10**6
    areas_between_lats = np.diff(areas_to_equator)
    areas_cells = np.abs(areas_between_lats) * q  # unit is km2 (x100 for ha)
    areagrid = np.transpose(np.matlib.repmat(areas_cells, cols, 1))

    # set all values to nan that <= 0 in input (if < area will be calculated for zero values too)
    areagrid[input_raster <= 0] = np.nan

    if output_name:
        with rs.open(filepath / output_name, 'w', **profile) as dst:
            dst.write(areagrid, 1)
    else:
        return areagrid

# function to sum cells in array that are positive (squeeze removes non-required dims)
def pos_val_summer(arr, squeeze=True):
    if squeeze:
        arr = np.squeeze(arr)

    arr = np.clip(arr, 0, None)  # set values below zero to 0
    return np.nansum(arr)  # sum only non-NaN values

# function to plot land-per-removal, removal, and land
def process_data(land_df, removal_df, cdr_option):
    land_per_removal = pd.merge(land_df, removal_df, on=['Scenario', 'Year'])
    land_per_removal['Land'] = land_per_removal['Land'] * 0.000001  # km2 to Mkm2
    land_per_removal.loc[land_per_removal['Land'] == 0, 'Removal'] = 0  # consistency rule: if land=0, removal=0
    land_per_removal['Mkm2/GtCO2'] = land_per_removal['Land'] / land_per_removal['Removal']
    land_per_removal['SSP'] = land_per_removal['Scenario'].str.split('-').str[0]
    land_per_removal['RCP'] = land_per_removal['Scenario'].str.split('-').str[1]

    return land_per_removal[['SSP', 'RCP', 'Year', 'Land', 'Removal', 'Mkm2/GtCO2']]

# function to concat multiple dfs across models
def load_and_concat(suffix, paths):
    dfs = [pd.read_csv(_path / f'{i}_{suffix}.csv') for i, _path in paths.items()]
    return pd.concat(dfs, ignore_index=True)

#function to calculate cumulative removal
def cum_cdr_calc(cdr_df):
    uniq = cdr_df[['Model', 'Scenario', 'Variable']].drop_duplicates()
    all_yrs = pd.Series(range(2020, 2101))
    all_yrs = pd.DataFrame({'Year': all_yrs})
    all_yrs = all_yrs.assign(key=1).merge(uniq.assign(key=1), on='key').drop('key', axis=1)
    cdr = pd.merge(all_yrs, cdr_df, on=['Model', 'Scenario', 'Year', 'Variable'], how='left')
    cdr = cdr.sort_values(['Model', 'Scenario', 'Year'])
    cdr.set_index(['Model', 'Scenario', 'Year', 'Variable'], inplace=True)
    cdr.interpolate(method='linear', inplace=True)
    cdr.reset_index(inplace=True)
    cdr['Cum'] = cdr.groupby(['Model', 'Scenario', 'Variable'])['Removal'].cumsum()

    return cdr[['Model', 'Scenario', 'Variable', 'Year', 'Removal', 'Cum']].copy()

# function to find the year of a given annual removal
def yr_target_finder(df, cdr_target):
    up = df[df['Removal'] >= cdr_target].groupby(['SSP', 'RCP']).first().reset_index()
    up = up[['SSP', 'RCP', 'Year']].copy()
    down = up.copy()
    down['Year'] = down['Year'] - 10

    down = pd.merge(down, df[['RCP', 'SSP', 'Year', 'Removal']],
                    on=['SSP', 'RCP', 'Year'], how='inner')

    up = pd.merge(up, df[['RCP', 'SSP', 'Year', 'Removal']],
                  on=['SSP', 'RCP', 'Year'], how='inner')

    cdr_range = pd.merge(down, up, on=['SSP', 'RCP'], suffixes=['_low', '_up'])

    def helper_func(row):
        yr_low = row['Year_low']
        yr_up = row['Year_up']
        down = row['Removal_low']
        up = row['Removal_up']
        # for each scenario, calc in which year x-amount of CDR is removed
        yr_target = yr_low + ((yr_up - yr_low) / (up - down)) * (cdr_target - down)
        return yr_target

    cdr_range['yr_target'] = cdr_range.apply(helper_func, axis=1)
    return cdr_range

# function to overlay raster and admin boundary shapefile
def admin_bound_calculator(key, admin_sf, intersect_src):
    sf = admin_sf
    shapes = sf.shapes()
    records = sf.records()

    country_vals = {}
    for record, shp in zip(records, shapes):  # calc raster vals in polygons
        country_name = record['iso3']
        geom = shape(shp.__geo_interface__)
        # mask the raster with the reprojected geometry
        out_image, _ = mask(intersect_src, [mapping(geom)], crop=True)
        out_image = out_image[0]  # extract the first band

        nodata_value = intersect_src.nodata
        if nodata_value is not None:
            out_image = np.where(out_image == nodata_value, np.nan, out_image)

        total_value = np.nansum(out_image)  # calc sum without nan values
        country_vals[country_name] = total_value
    df = pd.DataFrame(list(country_vals.items()), columns=['iso3', 'km2'])
    df['key'] = key
    return df[['key', 'iso3', 'km2']].copy()
