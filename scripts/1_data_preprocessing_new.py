
# import required libraries
import rasterio as rs
from rasterio.warp import Resampling
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import os
import numpy.matlib
from time import time

from required_functions import *

filepath_globiom = '/Users/rubenprutz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps2/'
filepath_uea = '/Users/rubenprutz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km/'
filepath_aim = '/Users/rubenprutz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps/'

# load lookup table containing nc file information
lookup_resample = pd.read_csv(
    filepath_uea + 'lookup_table_uea_resample_20km.csv')

lookup_interpol = pd.read_csv(
    filepath_uea + 'lookup_table_uea_interpol_20km.csv')

lookup_globiom_nc_df = pd.read_csv(filepath_globiom + 'lookup_table_globiom_nc_files.csv')
lookup_globiom_nc_df['year'] = lookup_globiom_nc_df['year'].astype(str)

lookup_aim_nc_df = pd.read_csv(
    filepath_aim + 'lookup_table_aim_nc_files.csv')

# %% adjust names of biodiv files
for index, row in lookup_resample.iterrows():  # use lookup to resample uea files
    input_tif = row['filename']
    output_name = row['output_name']

    with rs.open(filepath_uea + input_tif, 'r') as input_tiff1:
        tiff = input_tiff1.read()
        profile = input_tiff1.profile

    with rs.open(filepath_uea + output_name, 'w', **profile) as dst:
        dst.write(tiff.astype(profile['dtype']))

# linearily interpolate warmig level between rasters
inter_steps = 4  # number of desired interpolation steps

for index, row in lookup_interpol.iterrows():  # use lookup to interpolate uea files
    lower_file = row['lower_file']
    upper_file = row['upper_file']
    step_a = row['step_a']
    step_b = row['step_b']
    step_c = row['step_c']
    step_d = row['step_d']

    with rs.open(filepath_uea + lower_file, 'r') as input_tiff1:
        lower_tiff = input_tiff1.read()
        profile = input_tiff1.profile

    with rs.open(filepath_uea + upper_file, 'r') as input_tiff2:
        upper_tiff = input_tiff2.read()

    tiff_diff = upper_tiff - lower_tiff
    step_1 = tiff_diff * (1/(inter_steps+1)) + lower_tiff
    step_2 = tiff_diff * (2/(inter_steps+1)) + lower_tiff
    step_3 = tiff_diff * (3/(inter_steps+1)) + lower_tiff
    step_4 = tiff_diff * (4/(inter_steps+1)) + lower_tiff

    filenames = [step_a, step_b, step_c, step_d]
    files = [step_1, step_2, step_3, step_4]

    for filename, file in zip(filenames, files):
        with rs.open(filepath_uea + filename, 'w', **profile) as dst:
            dst.write(file.astype(profile['dtype']))

# create binary raster based on refugia threshold (0.75) using binary_refugia_converter
input_files = ['bio1.0_near.tif', 'bio1.1_near.tif', 'bio1.2_near.tif',
               'bio1.3_near.tif', 'bio1.4_near.tif', 'bio1.5_near.tif',
               'bio1.6_near.tif', 'bio1.7_near.tif', 'bio1.8_near.tif',
               'bio1.9_near.tif', 'bio2.0_near.tif', 'bio2.1_near.tif',
               'bio2.2_near.tif', 'bio2.3_near.tif', 'bio2.4_near.tif',
               'bio2.5_near.tif', 'bio2.6_near.tif', 'bio2.7_near.tif',
               'bio2.8_near.tif', 'bio2.9_near.tif', 'bio3.0_near.tif',
               'bio3.1_near.tif', 'bio3.2_near.tif', 'bio3.3_near.tif',
               'bio3.4_near.tif', 'bio3.5_near.tif', 'bio3.6_near.tif',
               'bio3.7_near.tif', 'bio3.8_near.tif', 'bio3.9_near.tif',
               'bio4.0_near.tif', 'bio4.1_near.tif', 'bio4.2_near.tif',
               'bio4.3_near.tif', 'bio4.4_near.tif', 'bio4.5_near.tif']

for input_file in input_files:
    output_file = input_file.replace('near.tif', 'bin.tif')
    binary_refugia_converter(input_file, filepath_uea, 0.75, output_file)

# %% AIM and GLOBIOM land use data processing:

# preprocess GLOBIOM data to order dimensions and to select the data variable
for i in lookup_globiom_nc_df['nc_file'].unique().tolist():

    nc_file_xr = xr.open_dataset(filepath_globiom + i, decode_times=False)
    nc_file_xr = nc_file_xr[['longitude', 'latitude', 'time', 'lc_class',
                             'GLOBIOM land use projections']]
    os.remove(filepath_globiom + i)  # delete original file before saving new
    nc_file_xr.to_netcdf(filepath_globiom + i)

# %% write crs, convert to tif, and create individual tifs per year and variable
target_res = (0.1666666666670000019, 0.1666666666670000019)  # uea resolution
land_infos = np.array(['Afforestation', 'Bioenergy',
                       'cropland_other', 'forest_total'])  # define for later

start = time()

models = ['AIM', 'GLOBIOM']  # AIM and GLOBIOM

for model in models:

    if model == 'GLOBIOM':
        filepath = filepath_globiom
        lookup_table = lookup_globiom_nc_df
    elif model == 'AIM':
        filepath = filepath_aim
        lookup_table = lookup_aim_nc_df

    for index, row in lookup_table.iterrows():  # use lookup to resample uea files
        input_file = row['nc_file']
        band = row['band']
        output_name = row['output_name']

        nc_file = rioxarray.open_rasterio(filepath + input_file,
                                          decode_times=False,
                                          band_as_variable=True)
        data_array_proj = nc_file.rio.write_crs('EPSG:4326')
        data_array_proj = data_array_proj['band_' + str(band)]
        data_array_proj.rio.to_raster(
            filepath + 'temp_large_file.tif', driver='GTiff')

        with rs.open(filepath + 'temp_large_file.tif') as src:
            data = src.read(1)
            profile = src.profile.copy()
            profile.update(count=1)
        with rs.open(filepath + output_name, 'w', **profile) as dst:
            dst.write(data, 1)

        # resample land use data to resolution of biodiv data
        tiff_resampler(filepath + output_name, target_res,
                       'nearest', filepath + output_name)

    # compute total forest per scenario and year

    scenarios = lookup_table['scenario'].unique()
    scenarios = scenarios.astype(str)
    years = lookup_table['year'].unique()
    years = years.astype(str)

    for scenario in scenarios:
        for year in years:

            unmanaged_forest = f'{model}_forest_unmanaged_{scenario}_{year}.tif'
            managed_forest = f'{model}_forest_managed_{scenario}_{year}.tif'
            output_name = f'{model}_forest_total_{scenario}_{year}.tif'

            unmanaged_forest = rioxarray.open_rasterio(filepath + unmanaged_forest,
                                                       masked=True)
            managed_forest = rioxarray.open_rasterio(filepath + managed_forest,
                                                     masked=True)
            total_forest = unmanaged_forest + managed_forest

            total_forest.rio.to_raster(filepath + output_name,
                                       driver='GTiff')

    # compute afforestation for all years vs 2020

    for scenario in scenarios:
        file_2020 = f'{model}_forest_total_{scenario}_2020.tif'

        for year in years:
            forest_file = f'{model}_forest_total_{scenario}_{year}.tif'
            ar_file_yr = f'{model}_Afforestation_{scenario}_{year}.tif'

            forest_2020 = rioxarray.open_rasterio(
                filepath + file_2020, masked=True)
            forest_yr = rioxarray.open_rasterio(
                filepath + forest_file, masked=True)

            forest_change = (forest_yr - forest_2020)  # -ve=loss; +ve=gain

            gain_yr = forest_change.where(
                (forest_change > 0) | forest_change.isnull(), 0)

            gain_yr.rio.to_raster(filepath + ar_file_yr,
                                  driver='GTiff')

    # calculate grid area based on arbitrarily chosen input file
    arbit_input = rioxarray.open_rasterio(
        filepath + f'{model}_Afforestation_SSP1-19_2050.tif', masked=True)

    bin_land = arbit_input.where(arbit_input.isnull(), 1)  # all=1 if not nodata
    bin_land.rio.to_raster(filepath + 'bin_land.tif',
                           driver='GTiff')

    land_area_calculation(filepath, 'bin_land.tif', f'{model}_max_land_area_km2.tif')
    max_land_area = rioxarray.open_rasterio(filepath +
                                            f'{model}_max_land_area_km2.tif', masked=True)

    # calculate land use areas based on total surface and land use fractions
    for land_info in land_infos:
        for scenario in scenarios:
            for year in years:

                try:
                    processing = f'{model}_{land_info}_{scenario}_{year}.tif'
                    land_fract = rioxarray.open_rasterio(
                        filepath + processing, masked=True)

                    land_fract = land_fract.rio.reproject_match(max_land_area)
                    land_area = land_fract * max_land_area

                    land_area.rio.to_raster(filepath + processing,
                                            driver='GTiff')
                except Exception as e:
                    print(f'Error processing: {e}')
                    continue

end = time()
print(f'Runtime {(end - start) /60} min')

# %% test alignment between spatial data and AR6 data concerning land cover

scenario_set = ['SSP1-19', 'SSP2-26', 'SSP3-34']
year_set = [2020, 2050, 2100]

for model in models:
    if model == 'GLOBIOM':
        filepath = filepath_globiom
    elif model == 'AIM':
        filepath = filepath_aim

    for scenario in scenario_set:
        for year in year_set:

            cropland_bioeng = f'{model}_Bioenergy_{scenario}_{year}.tif'
            cropland_other = f'{model}_cropland_other_{scenario}_{year}.tif'
            output_name = f'{model}_Cropland_total_{scenario}_{year}.tif'

            cropland_bioeng = rioxarray.open_rasterio(filepath + cropland_bioeng,
                                                      masked=True)
            cropland_other = rioxarray.open_rasterio(filepath + cropland_other,
                                                     masked=True)
            total_cropland = cropland_bioeng + cropland_other

            total_cropland.rio.to_raster(filepath + output_name,
                                         driver='GTiff')

for model in models:
    if model == 'GLOBIOM':
        filepath = filepath_globiom
    elif model == 'AIM':
        filepath = filepath_aim

    for scenario in scenario_set:
        for year in year_set:

            forest_total = f'{model}_forest_total_{scenario}_{year}.tif'
            cropland_total = f'{model}_Cropland_total_{scenario}_{year}.tif'

            forest = rioxarray.open_rasterio(filepath + forest_total,
                                             masked=True)
            cropland = rioxarray.open_rasterio(filepath + cropland_total,
                                               masked=True)
            forest = pos_val_summer(forest, squeeze=True)
            cropland = pos_val_summer(cropland, squeeze=True)

            forest = forest * 100 / 1000000  # from km2 to Mha
            cropland = cropland * 100 / 1000000  # from km2 to Mha

            print(f'{model} {scenario} {year} Forest: {forest} Mha')
            print(f'{model} {scenario} {year} Cropland: {cropland} Mha')
