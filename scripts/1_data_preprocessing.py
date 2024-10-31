
# import required libraries
import rasterio as rs
from rasterio.warp import Resampling
import numpy as np
import pandas as pd
import rioxarray
import numpy.matlib
from time import time

from required_functions import *

filepath_globiom = '/Users/rubenprutz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps1/'
filepath_uea = '/Users/rubenprutz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km/'
filepath_aim = '/Users/rubenprutz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps/'

# load lookup table containing nc file information
lookup_resample = pd.read_csv(
    filepath_uea + 'lookup_table_uea_resample_20km.csv')

lookup_interpol = pd.read_csv(
    filepath_uea + 'lookup_table_uea_interpol_20km.csv')

lookup_cdr_globiom_df = pd.read_csv(filepath_globiom + 'lookup_table_globiom1_nc_files.csv')
lookup_cdr_globiom_df['year'] = lookup_cdr_globiom_df['year'].astype(str)

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

# %% convert nc file to geotiff using nc_geotiff_converter
# iterate through lookup tables to fill the nc_geotiff_converter function
start = time()

for index, row in lookup_cdr_globiom_df.iterrows():
    input_nc = row['iiasa_file']
    variable = row['variable'].replace(' ', '')
    time_band = row['time_band']
    output_name = row['output_name']

    nc_geotiff_converter(input_nc,
                         filepath_globiom,
                         variable,
                         time_band,
                         output_name)

# # %% resample land use data to resolution of biodiv data using tiff_resampler
target_res = (0.1666666666670000019, 0.1666666666670000019)  # uea resolution

for index, row in lookup_cdr_globiom_df.iterrows():  # use lookup to resample iiasa files
    file_name = row['output_name']

    tiff_resampler(filepath_globiom + file_name,
                   target_res,
                   'sum',  # sum instead of nearest to retain numerical property
                   filepath_globiom + file_name)

end = time()
print(f'Runtime {(end - start) /60} min')

# %% convert globiom cdr-related files to km2
for index, row in lookup_cdr_globiom_df.iterrows():
    processing_file = row['output_name']

    unconverted_file = rioxarray.open_rasterio(
        filepath_globiom + processing_file, masked=True)

    converted_file = unconverted_file * 1000 / 100  # x1000ha and ha to km2

    converted_file.rio.to_raster(filepath_globiom + processing_file,
                                driver='GTiff')

# %% AIM land use data processing:
# write crs, convert to tif, and create individual tifs per year and variable

for index, row in lookup_aim_nc_df.iterrows():  # use lookup to resample uea files
    input_file = row['nc_file']
    band = row['band']
    output_name = row['output_name']

    nc_file = rioxarray.open_rasterio(filepath_aim + input_file,
                                      decode_times=False,
                                      band_as_variable=True)
    data_array_proj = nc_file.rio.write_crs('EPSG:4326')
    data_array_proj.rio.to_raster(
        filepath_aim + 'temp_large_file.tif', driver='GTiff')

    with rs.open(filepath_aim + 'temp_large_file.tif') as src:
        data = src.read(band)
        profile = src.profile.copy()
        profile.update(count=1)
    with rs.open(filepath_aim + output_name, 'w', **profile) as dst:
        dst.write(data, 1)

# compute total forest area per scenario and year

scenarios = lookup_aim_nc_df['scenario'].unique()
scenarios = scenarios.astype(str)
years = lookup_aim_nc_df['year'].unique()
years = years.astype(str)

for scenario in scenarios:
    for year in years:

        unmanaged_forest = f'AIM_forest_unmanaged_{scenario}_{year}.tif'
        managed_forest = f'AIM_forest_managed_{scenario}_{year}.tif'
        output_name = f'AIM_forest_total_{scenario}_{year}.tif'

        unmanaged_forest = rioxarray.open_rasterio(filepath_aim + unmanaged_forest,
                                                   masked=True)
        managed_forest = rioxarray.open_rasterio(filepath_aim + managed_forest,
                                                 masked=True)
        total_forest = unmanaged_forest + managed_forest

        total_forest.rio.to_raster(filepath_aim + output_name,
                                   driver='GTiff')

# compute afforestation for all years vs 2020

for scenario in scenarios:
    file_2020 = f'AIM_forest_total_{scenario}_2020.tif'

    for year in range(2020, 2101, 10):
        forest_file = f'AIM_forest_total_{scenario}_{year}.tif'
        ar_file_yr = f'AIM_Afforestation_{scenario}_{year}.tif'

        forest_2020 = rioxarray.open_rasterio(
            filepath_aim + file_2020, masked=True)
        forest_yr = rioxarray.open_rasterio(
            filepath_aim + forest_file, masked=True)

        forest_change = (forest_yr - forest_2020)  # -ve=loss; +ve=gain

        gain_yr = forest_change.where(
            (forest_change > 0) | forest_change.isnull(), 0)

        gain_yr.rio.to_raster(filepath_aim + ar_file_yr,
                              driver='GTiff')

land_infos = np.array(['Afforestation', 'Bioenergy'])
years = [str(year) for year in range(2020, 2101, 10)]

for land_info in land_infos:
    for scenario in scenarios:
        for year in years:

            processing_in = f'AIM_{land_info}_{scenario}_{year}.tif'
            processing_out = f'_AIM_{land_info}_{scenario}_{year}.tif'

            tiff_resampler(filepath_aim + processing_in,
                           target_res,
                           'nearest',  # nearest as AIM is more coarse than biodiv data
                           filepath_aim + processing_out)

# calculate grid area based on arbitrarily chosen input file
arbit_input = rioxarray.open_rasterio(
    filepath_aim + '_AIM_Afforestation_SSP1-19_2050.tif', masked=True)

bin_land = arbit_input.where(arbit_input.isnull(), 1)  # all=1 if not nodata
bin_land.rio.to_raster(filepath_aim + 'bin_land.tif',
                       driver='GTiff')

land_area_calculation(filepath_aim, 'bin_land.tif', 'AIM_max_land_area_km2.tif')

# calculate land use areas based on total surface and land use fractions

for land_info in land_infos:
    for scenario in scenarios:
        for year in years:

            processing_in = f'_AIM_{land_info}_{scenario}_{year}.tif'
            processing_out = f'AIM_{land_info}_{scenario}_{year}.tif'

            land_fraction = rioxarray.open_rasterio(
                filepath_aim + processing_in, masked=True)

            max_land_area = rioxarray.open_rasterio(
                filepath_aim + 'AIM_max_land_area_km2.tif', masked=True)

            land_area = land_fraction * max_land_area

            land_area.rio.to_raster(filepath_aim + processing_out,
                                    driver='GTiff')
