
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
import seaborn as sns
import rioxarray
import rasterio as rs
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cmasher as cmr
from time import time
from required_functions import *
from rasterio.mask import mask
import shapefile
import shapely.geometry as sgeom
from shapely.geometry import shape, mapping
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
ar6_file = 'AR6_Scenarios_Database_World_v1.1.csv'
ar6_db = pd.read_csv(path_ar6_data / ar6_file)

path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_ipl = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ipl_maps/01_Data')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')

path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
file_all = 'lookup_table_cdr_files_all_models.csv'

lookup_names = pd.read_csv(path_globiom / 'lookup_table_ssp-rcp_names.csv')

# %% choose model to run the script with
model = 'GLOBIOM'  # options: 'GLOBIOM' or 'AIM' or 'IMAGE'

if model == 'GLOBIOM':
    path = path_globiom
    model_setup = 'MESSAGE-GLOBIOM 1.0'
    removal_lvl = 4
elif model == 'AIM':
    path = path_aim
    model_setup = 'AIM/CGE 2.0'
    removal_lvl = 2.5
elif model == 'IMAGE':
    path = path_image
    model_setup = 'IMAGE 3.0.1'
    removal_lvl = 2.5

# land-per-removal curve calculation
# %% STEP1: calculate removal per scenario in for 2020-2100
scenarios = ['SSP1-Baseline', 'SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45',
             'SSP2-Baseline', 'SSP2-19', 'SSP2-26', 'SSP2-34', 'SSP2-45',
             'SSP2-60', 'SSP3-Baseline', 'SSP3-34', 'SSP3-45', 'SSP3-60']

ar6_db = ar6_db.loc[ar6_db['Model'].isin([model_setup]) & ar6_db['Scenario'].isin(scenarios)]
numeric_cols = [str(year) for year in range(2020, 2110, 10)]
cdr = ar6_db[['Scenario', 'Variable'] + numeric_cols].copy()
cdr_array = ['Carbon Sequestration|CCS|Biomass',
             'Carbon Sequestration|Land Use']
cdr = cdr[cdr['Variable'].isin(cdr_array)]
cdr[numeric_cols] = cdr[numeric_cols].abs()
cdr = cdr.melt(id_vars=['Scenario', 'Variable'], var_name='Year', value_name='Removal')
cdr['Removal'] = cdr['Removal'] * 0.001  # Mt to Gt
cdr['Year'] = pd.to_numeric(cdr['Year'])

ar_removal = cdr[cdr['Variable'] == 'Carbon Sequestration|Land Use']
ar_removal['Variable'] = 'AR removal'

# %% STEP2: calculate afforestation land per scenario in for 2020-2100
# filter afforestation related files from lookup table
landfile_lookup = pd.read_csv(path_all / file_all)
ar_lookup = landfile_lookup[landfile_lookup['mitigation_option'] == 'Afforestation']
ar_land = pd.DataFrame(columns=['Scenario', 'Year', 'Land'])

for index, row in ar_lookup.iterrows():  # calculate afforestation land per scenario
    input_nc = row['file_name']
    scenario = row['scenario']
    year = row['year']

    land_use = rioxarray.open_rasterio(path / f'{model}_{input_nc}',
                                       masked=True)
    tot_cdr_area = pos_val_summer(land_use, squeeze=True)

    new_row = pd.DataFrame({'Scenario': [scenario], 'Year': [year],
                            'Land': [tot_cdr_area]})
    ar_land = pd.concat([ar_land, new_row], ignore_index=True)

ar_land['Variable'] = 'Land demand'

# %% calculate BECCS removal and bioenergy land use fractions
beccs_removal = cdr[cdr['Variable'] == 'Carbon Sequestration|CCS|Biomass']
beccs_removal['Variable'] = 'BECCS removal'

# calculate share of bioenergy for BECCS based on biomass with/without CCS
bioeng_ncss = ar6_db.query('Variable == "Primary Energy|Biomass|Modern|w/o CCS"').reset_index(drop=True)
bioeng_tot = ar6_db.query('Variable == "Primary Energy|Biomass"').reset_index(drop=True)

bioeng_wccs = bioeng_tot[['Scenario']].copy()
bioeng_wccs[numeric_cols] = 1 - (bioeng_ncss[numeric_cols] / bioeng_tot[numeric_cols])
bioeng_wccs['Variable'] = 'Removal fraction'
bioeng_wccs = pd.melt(bioeng_wccs,
                      id_vars=['Scenario', 'Variable'],
                      var_name='Year',
                      value_name='Fraction')

# filter bioenergy related files from lookup table
be_lookup = landfile_lookup[landfile_lookup['mitigation_option'] == 'Bioenergy plantation']
be_land = pd.DataFrame(columns=['Scenario', 'Year', 'Land'])

for index, row in be_lookup.iterrows():  # calculate bioenergy land per scenario
    input_nc = row['file_name']
    scenario = row['scenario']
    year = row['year']

    land_use = rioxarray.open_rasterio(path / f'{model}_{input_nc}',
                                       masked=True)
    tot_cdr_area = pos_val_summer(land_use, squeeze=True)

    be_land = pd.concat([be_land, pd.DataFrame({
        'Scenario': [scenario],
        'Year': [year],
        'Land': [tot_cdr_area]
    })], ignore_index=True)

be_land['Variable'] = 'Land demand'
be_land['Year'] = be_land['Year'].apply(pd.to_numeric)
bioeng_wccs['Year'] = bioeng_wccs['Year'].apply(pd.to_numeric)

beccs_land = pd.merge(be_land[['Scenario', 'Year', 'Land', 'Variable']],
                      bioeng_wccs[['Scenario', 'Year', 'Fraction']],
                      on=['Scenario', 'Year'])

beccs_land['Land'] = beccs_land['Land'] * beccs_land['Fraction']

# %% plot land-per-removal, removal and land for AR and BECCS
lpr_ar = process_data_and_plot(ar_land, ar_removal, 'AR')  # plot AR data
lpr_beccs = process_data_and_plot(beccs_land, beccs_removal, 'BECCS')  # plot BECCS data

# %% impact-per-removal analysis (afforestation)

lpr_ar_strict = lpr_ar.loc[lpr_ar['RCP'].isin(['34', '45'])]  # only RCPs available for all SSPs
removal_steps = [2, 2.5, 3, 4]  # specify CDR levels (add more if required)

all_results = []
for removal_step in removal_steps:
    # for each scenario, get first yr >= x GtCO2 and -10 yrs for lower bound
    ar_up_xgt = lpr_ar_strict[lpr_ar_strict['Removal'] >=
                              removal_step].groupby(['SSP', 'RCP']).first().reset_index()
    ar_up_xgt = ar_up_xgt[['SSP', 'RCP', 'Year']].copy()
    ar_low_xgt = ar_up_xgt.copy()
    ar_low_xgt['Year'] = ar_low_xgt['Year'] - 10

    ar_low_xgt = pd.merge(ar_low_xgt, lpr_ar_strict[['RCP', 'SSP', 'Year', 'Removal']],
                          on=['SSP', 'RCP', 'Year'], how='inner')

    ar_up_xgt = pd.merge(ar_up_xgt, lpr_ar_strict[['RCP', 'SSP', 'Year', 'Removal']],
                         on=['SSP', 'RCP', 'Year'], how='inner')

    ar_range = pd.merge(ar_low_xgt, ar_up_xgt, on=['SSP', 'RCP'],
                        suffixes=['_low', '_up'])

    # for each scenario, calc in which year x GtCO2 is removed
    def yr_target_finder(row):
        yr_low = row['Year_low']
        yr_up = row['Year_up']
        cdr_low = row['Removal_low']
        cdr_up = row['Removal_up']
        cdr_target = removal_step

        yr_target = yr_low + ((yr_up - yr_low) / (cdr_up - cdr_low)) * (cdr_target - cdr_low)
        return yr_target

    ar_range['yr_target'] = ar_range.apply(yr_target_finder, axis=1)

    # interpolate land use layers to yr_target
    ar_test = []

    for index, row in ar_range.iterrows():
        ssp = row['SSP']
        rcp = row['RCP']
        yr_low = row['Year_low']
        yr_up = row['Year_up']
        yr_target = row['yr_target']

        lower_tiff = f'{model}_Afforestation_{ssp}-{rcp}_{yr_low}.tif'
        upper_tiff = f'{model}_Afforestation_{ssp}-{rcp}_{yr_up}.tif'
        output_name = f'{model}_Afforestation_{ssp}-{rcp}_{removal_step}GtCO2.tif'

        with rs.open(path / lower_tiff) as src_low:
            with rs.open(path / upper_tiff) as src_up:
                # Read raster data and geospatial information
                lower_tiff = src_low.read(1)
                upper_tiff = src_up.read(1)
                profile_lower = src_low.profile

                lower_tiff = lower_tiff * 0.000001  # km2 to Mkm2
                upper_tiff = upper_tiff * 0.000001  # km2 to Mkm2

                yr_diff = yr_up - yr_low  # diff of known years
                tiff_diff = upper_tiff - lower_tiff  # diff of known tiffs

                # lower tiff plus the fraction of tiff_diff for a given target yr
                tiff_target = lower_tiff + (tiff_diff * ((yr_target - yr_low) / yr_diff))

                profile_updated = profile_lower.copy()
                profile_updated.update(dtype=rs.float32)

                with rs.open(path / output_name, "w", **profile_updated) as dst:
                    dst.write(tiff_target.astype(rs.float32), 1)

# %% impact-per-removal analysis (BECCS)

lpr_beccs_strict = lpr_beccs.loc[lpr_beccs['RCP'].isin(['34', '45'])]  # only RCPs available for all SSPs

all_results = []
for removal_step in removal_steps:
    # for each scenario, get first yr >= x GtCO2 and -10 yrs for lower bound
    beccs_up_xgt = lpr_beccs_strict[lpr_beccs_strict['Removal'] >=
                               removal_step].groupby(['SSP', 'RCP']).first().reset_index()
    beccs_up_xgt = beccs_up_xgt[['SSP', 'RCP', 'Year']].copy()
    beccs_low_xgt = beccs_up_xgt.copy()
    beccs_low_xgt['Year'] = beccs_low_xgt['Year'] - 10

    beccs_low_xgt = pd.merge(beccs_low_xgt, lpr_beccs_strict[['RCP', 'SSP', 'Year', 'Removal']],
                             on=['SSP', 'RCP', 'Year'], how='inner')

    beccs_up_xgt = pd.merge(beccs_up_xgt, lpr_beccs_strict[['RCP', 'SSP', 'Year', 'Removal']],
                            on=['SSP', 'RCP', 'Year'], how='inner')

    beccs_range = pd.merge(beccs_low_xgt, beccs_up_xgt, on=['SSP', 'RCP'],
                           suffixes=['_low', '_up'])

    # for each scenario, calc in which year x-amount of CDR is removed
    def yr_target_finder(row):
        yr_low = row['Year_low']
        yr_up = row['Year_up']
        cdr_low = row['Removal_low']
        cdr_up = row['Removal_up']
        cdr_target = removal_step

        yr_target = yr_low + ((yr_up - yr_low) / (cdr_up - cdr_low)) * (cdr_target - cdr_low)
        return yr_target

    beccs_range['yr_target'] = beccs_range.apply(yr_target_finder, axis=1)

    # interpolate land use layers to yr_target
    beccs_test = []

    for index, row in beccs_range.iterrows():
        ssp = row['SSP']
        rcp = row['RCP']
        yr_low = row['Year_low']
        yr_up = row['Year_up']
        yr_target = row['yr_target']

        lower_tiff = f'{model}_Bioenergy_{ssp}-{rcp}_{yr_low}.tif'
        upper_tiff = f'{model}_Bioenergy_{ssp}-{rcp}_{yr_up}.tif'
        output_name = f'{model}_BECCS_{ssp}-{rcp}_{removal_step}GtCO2.tif'

        # get BECCS fraction of bioenergy for respective year and scenario
        lower_fract = float(beccs_land.loc[(beccs_land['Year'] == yr_low) &
                                           (beccs_land['Scenario'] == f'{ssp}-{rcp}'),
                                           'Fraction'].iloc[0])
        upper_fract = float(beccs_land.loc[(beccs_land['Year'] == yr_up) &
                                           (beccs_land['Scenario'] == f'{ssp}-{rcp}'),
                                           'Fraction'].iloc[0])

        with rs.open(path / lower_tiff) as src_low:
            with rs.open(path / upper_tiff) as src_up:
                # Read raster data and geospatial information
                lower_tiff = src_low.read(1)
                upper_tiff = src_up.read(1)
                profile_lower = src_low.profile

                lower_tiff = lower_tiff * 0.000001  # km2 to Mkm2
                upper_tiff = upper_tiff * 0.000001  # km2 to Mkm2

                # multiply bioenery cells by BECCS fraction assuming even distribution
                lower_beccs = lower_tiff * lower_fract
                upper_beccs = upper_tiff * upper_fract

                yr_diff = yr_up - yr_low  # diff of known years
                tiff_diff = upper_beccs - lower_beccs  # diff of known tiffs

                # lower tiff plus the fraction of tiff_diff for a given target yr
                tiff_target = lower_beccs + (tiff_diff * ((yr_target - yr_low) / yr_diff))

                profile_updated = profile_lower.copy()
                profile_updated.update(dtype=rs.float32)

                with rs.open(path / output_name, "w", **profile_updated) as dst:
                    dst.write(tiff_target.astype(rs.float32), 1)

# %% maps of land impact of CDR across SSP1-3 for a certain warming level

ssps = ['SSP1', 'SSP2', 'SSP3']

for ssp in ssps:
    try:
        ar_34 = f'{model}_Afforestation_{ssp}-34_{removal_lvl}GtCO2.tif'
        be_34 = f'{model}_BECCS_{ssp}-34_{removal_lvl}GtCO2.tif'

        ar_45 = f'{model}_Afforestation_{ssp}-45_{removal_lvl}GtCO2.tif'
        be_45 = f'{model}_BECCS_{ssp}-45_{removal_lvl}GtCO2.tif'

        ar_34 = rioxarray.open_rasterio(path / ar_34, masked=True)
        be_34 = rioxarray.open_rasterio(path / be_34, masked=True)

        ar_45 = rioxarray.open_rasterio(path / ar_45, masked=True)
        be_45 = rioxarray.open_rasterio(path / be_45, masked=True)

        ar = (ar_34 + ar_45) / 2  # average between the two available rcps
        be = (be_34 + be_45) / 2  # average between the two available rcps

        refugia = rioxarray.open_rasterio(path_uea / 'bio1.5_bin.tif', masked=True)  # specify warming level
        bin_land = refugia.where(refugia.isnull(), 1)  # all=1 if not nodata
        bin_land.rio.to_raster(path_uea / 'bin_land.tif', driver='GTiff')
        land_area_calculation(path_uea, 'bin_land.tif', 'bin_land_km2.tif')
        max_land_area = rioxarray.open_rasterio(path_uea / 'bin_land_km2.tif',
                                                masked=True)

        ar = ar.rio.reproject_match(refugia)  # match
        ar_in_bio = ar * refugia * 1000000  # calc overlay (Mkm2 to km2)

        be = be.rio.reproject_match(refugia)  # match
        be_in_bio = be * refugia * 1000000  # calc overlay (Mkm2 to km2)

        max_land_area = max_land_area.rio.reproject_match(ar_in_bio)  # match
        ar_in_bio_rel = ar_in_bio / max_land_area * 100  # calc overlay share per cell

        max_land_area = max_land_area.rio.reproject_match(be_in_bio)  # match
        be_in_bio_rel = be_in_bio / max_land_area * 100  # calc overlay share per cell

        # save overlays for later
        ar_in_bio.rio.to_raster(path / f'{ssp}_ar{removal_lvl}_bio_absolute.tif', driver='GTiff')
        be_in_bio.rio.to_raster(path / f'{ssp}_be{removal_lvl}_bio_absolute.tif', driver='GTiff')
        ar_in_bio_rel.rio.to_raster(path / f'{ssp}_ar{removal_lvl}_bio_relative.tif', driver='GTiff')
        be_in_bio_rel.rio.to_raster(path / f'{ssp}_be{removal_lvl}_bio_relative.tif', driver='GTiff')

        perc_thres = 1  # threshold is less than x% per cell for visualization
        ar_overlay = np.where(ar_in_bio_rel >= perc_thres, ar_in_bio_rel, np.where(ar_in_bio_rel < perc_thres, np.nan, ar_in_bio_rel))
        ar_overlay = ar_in_bio_rel.copy(data=ar_overlay)
        be_overlay = np.where(be_in_bio_rel >= perc_thres, be_in_bio_rel, np.where(be_in_bio_rel < perc_thres, np.nan, be_in_bio_rel))
        be_overlay = be_in_bio_rel.copy(data=be_overlay)

        ar_overlay.rio.to_raster(path / 'ar_overlay.tif', driver='GTiff')
        be_overlay.rio.to_raster(path / 'be_overlay.tif', driver='GTiff')
        refugia.rio.to_raster(path_uea / 'refugia_back.tif', driver='GTiff')

        ar = rs.open(path / 'ar_overlay.tif')
        be = rs.open(path / 'be_overlay.tif')
        refug = rs.open(path_uea / 'refugia_back.tif')

        data_ar = ar.read(1)
        data_be = be.read(1)
        data_refug = refug.read(1)

        # get the metadata
        transform = ar.transform
        extent_ar = [transform[2], transform[2] + transform[0] * ar.width,
                     transform[5] + transform[4] * ar.height, transform[5]]

        transform = be.transform
        extent_be = [transform[2], transform[2] + transform[0] * be.width,
                     transform[5] + transform[4] * be.height, transform[5]]

        transform = refug.transform
        extent_refug = [transform[2], transform[2] + transform[0] * refug.width,
                        transform[5] + transform[4] * refug.height, transform[5]]

        bounds_ar = [1, 20, 50, 80]
        norm_ar = mpl.colors.BoundaryNorm(bounds_ar, mpl.cm.Greens.N, extend='max')
        cmap_ar = cmr.get_sub_cmap('Greens', 0.2, 1)  # specify colormap subrange

        bounds_be = [1, 5, 10, 15]
        norm_be = mpl.colors.BoundaryNorm(bounds_be, mpl.cm.Reds.N, extend='max')
        cmap_be = cmr.get_sub_cmap('Reds', 0.2, 1)  # specify colormap subrange

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())

        img_re = ax.imshow(data_refug, extent=extent_refug, transform=ccrs.PlateCarree(),
                           origin='upper', cmap='Greys', alpha=0.1)

        img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                           origin='upper', cmap=cmap_ar, norm=norm_ar, alpha=1)

        img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                           origin='upper', cmap=cmap_be, norm=norm_be, alpha=0.7)

        ax.coastlines(linewidth=0.2)

        cbar_ar = plt.colorbar(img_ar, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
        cbar_be = plt.colorbar(img_be, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
        cbar_ar.ax.set_position([0.395, 0, 0.1, 0.5])
        cbar_be.ax.set_position([0.529, 0, 0.1, 0.5])
        cbar_ar.ax.tick_params(labelsize=8)
        cbar_be.ax.tick_params(labelsize=8)
        cbar_ar.set_label('Afforestation [%]', labelpad=1, fontsize=8)
        cbar_be.set_label('BECCS [%]', labelpad=1, fontsize=8)
        plt.title(f'{ssp}', fontsize=8)
        plt.show()
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        continue

# %% calculate country burden for refugia
# read the administrative boundary shapefile data
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

# use admin_bound_calculator for all SSPs for AR
for ssp in ssps:
    try:
        intersect_src = rs.open(path / f'{ssp}_ar{removal_lvl}_bio_absolute.tif')
        globals()[f'df_{ssp}'] = admin_bound_calculator(ssp, admin_sf, intersect_src)
        globals()[f'df_{ssp}']['option'] = 'AR'
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        globals()[f'df_{ssp}'] = pd.DataFrame()
        continue
df_ar = pd.concat([df_SSP1, df_SSP2, df_SSP3], axis=0)

# use admin_bound_calculator for all SSPs for BECCS
for ssp in ssps:
    try:
        intersect_src = rs.open(path / f'{ssp}_be{removal_lvl}_bio_absolute.tif')
        globals()[f'df_{ssp}'] = admin_bound_calculator(ssp, admin_sf, intersect_src)
        globals()[f'df_{ssp}']['option'] = 'Bioenergy'
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        globals()[f'df_{ssp}'] = pd.DataFrame()
        continue
df_be = pd.concat([df_SSP1, df_SSP2, df_SSP3], axis=0)

# concat output dfs for AR and BECCS
df_options = pd.concat([df_ar, df_be], axis=0)

# calculate refugia area at 1.5°C
land_area_calculation(path_uea, 'bio1.5_bin.tif', 'bio1.5_bin_km2.tif')

# use admin_bound_calculator to calc refugia area at 1.5°C per country
intersect_src = rs.open(path_uea / 'bio1.5_bin_km2.tif')
df_bio15 = admin_bound_calculator('all_ssps', admin_sf, intersect_src)
df_bio15 = df_bio15.rename(columns={'km2': 'bio_km2'})

# combine refugia total and refugia affected in one df and calc burdens
wab_out = pd.merge(df_options, df_bio15[['iso3', 'bio_km2']],
                   on='iso3', how='inner')
wab_out['affected_bio_share'] = wab_out['km2'] / wab_out['bio_km2'] * 100

# combine effect from AR and BECCS
wab_cum = wab_out.groupby(['key', 'iso3', 'bio_km2'], as_index=False)['km2'].sum()
wab_cum['cum_affected_bio_share'] = wab_cum['km2'] / wab_cum['bio_km2'] * 100
wab_cum['cum_affected_bio_share'] = wab_cum['cum_affected_bio_share'].round(1)
wab_cum.dropna(inplace=True)

wab_cum_ssp1 = wab_cum.query('key == "SSP1"').reset_index(drop=True)
wab_cum_ssp2 = wab_cum.query('key == "SSP2"').reset_index(drop=True)
wab_cum_ssp3 = wab_cum.query('key == "SSP3"').reset_index(drop=True)

# %% plot share of national refugia affected by AR and BECCS
wab_dict = {'SSP1': wab_cum_ssp1, 'SSP2': wab_cum_ssp2, 'SSP3': wab_cum_ssp3}
cdr_sum = 2*removal_lvl  # removal_level from both CDR options
cdr_sum = int(cdr_sum)

bounds = [0, 5, 10, 15, 20, 25, 30]
norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.PuRd.N, extend='max')
cmap = mpl.cm.PuRd

for ssp in wab_dict.keys():
    df = wab_dict[ssp].copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea()})

    shape_records = list(Reader(sf_path / 'world-administrative-boundaries.shp').records())

    # plot each country with data
    for record in shape_records:
        country_iso = record.attributes['iso3']
        if country_iso in df['iso3'].values:
            value = df[df['iso3'] == country_iso]['cum_affected_bio_share'].values[0]
            color = cmap(norm(value))
            geom = record.geometry
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=color,
                              edgecolor='black', linewidth=0.2)

    ax.coastlines(linewidth=0.2)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, orientation='horizontal',
                        boundaries=bounds, ticks=bounds,
                        spacing='proportional', extend='max')
    cbar.ax.set_position([0.346, -0.175, 0.334, 0.5])
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(f'Share of national refugia covered by AR and\n BECCS for removals of {cdr_sum} GtCO$_2$ [%]',
                   fontsize=11)
    plt.title(f'{ssp}', fontsize=11)
    plt.show()

# %% resample land use to match IPL resolution
upscale_factor = 5  # equivalent to IPL resolution
land_infos = np.array(['Afforestation', 'BECCS'])
scenarios = np.array(['SSP1-34', 'SSP2-34', 'SSP3-34', 'SSP1-45', 'SSP2-45', 'SSP3-45'])

for land_info in land_infos:
    for scenario in scenarios:

        file_in = f'{model}_{land_info}_{scenario}_{removal_lvl}GtCO2.tif'
        file_out = f'{model}_{land_info}_{scenario}_{removal_lvl}GtCO2_highres.tif'

        try:
            with rs.open(path / file_in) as dataset:

                data = dataset.read(
                    out_shape=(
                        dataset.count,
                        int(dataset.height * upscale_factor),
                        int(dataset.width * upscale_factor)),
                    resampling=Resampling.nearest)

                transform = dataset.transform * dataset.transform.scale(
                    (dataset.width / data.shape[-1]),
                    (dataset.height / data.shape[-2]))

                profile = dataset.meta.copy()
                profile.update(transform=transform, driver='GTiff',
                               height=data.shape[1], width=data.shape[2])

                with rs.open(path / file_out, 'w', **profile) as dataset:
                    dataset.write(data)

                old_size = rioxarray.open_rasterio(path / file_out, masked=True)
                new_size = old_size / 25  # divide by 25 due to changed resolution (5x5)
                new_size.rio.to_raster(path / file_out, driver='GTiff')

        except Exception as e:
            print(f'Error processing {scenario}')

# %% plot impact on IPLs
ipl = rioxarray.open_rasterio(path_ipl / 'IPL_2017_2arcmin.tif', masked=True)
bin_land = ipl.where(ipl.isnull(), 1)  # all=1 if not nodata
bin_land.rio.to_raster(path_ipl / 'bin_land.tif', driver='GTiff')
land_area_calculation(path_ipl, 'bin_land.tif', 'bin_land_km2.tif')
max_land_area = rioxarray.open_rasterio(path_ipl / 'bin_land_km2.tif',
                                        masked=True)

for ssp in ssps:
    try:
        ar_34 = f'{model}_Afforestation_{ssp}-34_{removal_lvl}GtCO2_highres.tif'
        be_34 = f'{model}_BECCS_{ssp}-34_{removal_lvl}GtCO2_highres.tif'

        ar_45 = f'{model}_Afforestation_{ssp}-45_{removal_lvl}GtCO2_highres.tif'
        be_45 = f'{model}_BECCS_{ssp}-45_{removal_lvl}GtCO2_highres.tif'

        ar_34 = rioxarray.open_rasterio(path / ar_34, masked=True)
        be_34 = rioxarray.open_rasterio(path / be_34, masked=True)

        ar_45 = rioxarray.open_rasterio(path / ar_45, masked=True)
        be_45 = rioxarray.open_rasterio(path / be_45, masked=True)

        ar = (ar_34 + ar_45) / 2  # average between the two available rcps
        be = (be_34 + be_45) / 2  # average between the two available rcps

        ar = ar.rio.reproject_match(ipl)  # match
        ar_in_ipl = ar * ipl * 1000000  # calc overlay (Mkm2 to km2)

        be = be.rio.reproject_match(ipl)  # match
        be_in_ipl = be * ipl * 1000000  # calc overlay (Mkm2 to km2)

        max_land_area = max_land_area.rio.reproject_match(ar_in_ipl)  # match
        ar_in_ipl_rel = ar_in_ipl / max_land_area * 100  # calc overlay share per cell

        max_land_area = max_land_area.rio.reproject_match(be_in_ipl)  # match
        be_in_ipl_rel = be_in_ipl / max_land_area * 100  # calc overlay share per cell

        # save overlays for later
        ar_in_ipl.rio.to_raster(path / f'{ssp}_ar{removal_lvl}_ipl_absolute.tif', driver='GTiff')
        be_in_ipl.rio.to_raster(path / f'{ssp}_be{removal_lvl}_ipl_absolute.tif', driver='GTiff')
        ar_in_ipl_rel.rio.to_raster(path / f'{ssp}_ar{removal_lvl}_ipl_relative.tif', driver='GTiff')
        be_in_ipl_rel.rio.to_raster(path / f'{ssp}_be{removal_lvl}_ipl_relative.tif', driver='GTiff')

        perc_thres = 1  # threshold is less than x% per cell for visualization
        ar_overlay = np.where(ar_in_ipl_rel >= perc_thres, ar_in_ipl_rel, np.where(ar_in_ipl_rel < perc_thres, np.nan, ar_in_ipl_rel))
        ar_overlay = ar_in_ipl_rel.copy(data=ar_overlay)
        be_overlay = np.where(be_in_ipl_rel >= perc_thres, be_in_ipl_rel, np.where(be_in_ipl_rel < perc_thres, np.nan, be_in_ipl_rel))
        be_overlay = be_in_ipl_rel.copy(data=be_overlay)

        ar_overlay.rio.to_raster(path / 'ar_overlay.tif', driver='GTiff')
        be_overlay.rio.to_raster(path / 'be_overlay.tif', driver='GTiff')
        ipl.rio.to_raster(path_ipl / 'ipl_back.tif', driver='GTiff')

        ar = rs.open(path / 'ar_overlay.tif')
        be = rs.open(path / 'be_overlay.tif')
        ipl_file = rs.open(path_ipl / 'ipl_back.tif')

        data_ar = ar.read(1)
        data_be = be.read(1)
        data_ipl = ipl_file.read(1)

        # get the metadata
        transform = ar.transform
        extent_ar = [transform[2], transform[2] + transform[0] * ar.width,
                     transform[5] + transform[4] * ar.height, transform[5]]

        transform = be.transform
        extent_be = [transform[2], transform[2] + transform[0] * be.width,
                     transform[5] + transform[4] * be.height, transform[5]]

        transform = ipl_file.transform
        extent_ipl = [transform[2], transform[2] + transform[0] * ipl_file.width,
                      transform[5] + transform[4] * ipl_file.height, transform[5]]

        bounds_ar = [1, 20, 50, 80]
        norm_ar = mpl.colors.BoundaryNorm(bounds_ar, mpl.cm.Greens.N, extend='max')
        cmap_ar = cmr.get_sub_cmap('Greens', 0.2, 1)  # specify colormap subrange

        bounds_be = [1, 5, 10, 15]
        norm_be = mpl.colors.BoundaryNorm(bounds_be, mpl.cm.Reds.N, extend='max')
        cmap_be = cmr.get_sub_cmap('Reds', 0.2, 1)  # specify colormap subrange

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())

        img_be = ax.imshow(data_ipl, extent=extent_ipl, transform=ccrs.PlateCarree(),
                           origin='upper', cmap='Greys', alpha=0.1)

        img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                           origin='upper', cmap=cmap_ar, norm=norm_ar, alpha=1)

        img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                           origin='upper', cmap=cmap_be, norm=norm_be, alpha=0.7)

        ax.coastlines(linewidth=0.2)

        cbar_ar = plt.colorbar(img_ar, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
        cbar_be = plt.colorbar(img_be, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
        cbar_ar.ax.set_position([0.395, 0, 0.1, 0.5])
        cbar_be.ax.set_position([0.529, 0, 0.1, 0.5])
        cbar_ar.ax.tick_params(labelsize=8)
        cbar_be.ax.tick_params(labelsize=8)
        cbar_ar.set_label('Afforestation [%]', labelpad=1, fontsize=8)
        cbar_be.set_label('BECCS [%]', labelpad=1, fontsize=8)
        plt.title(f'{ssp}', fontsize=8)
        plt.show()
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        continue

# %% calculate country burden for IPLs
# use admin_bound_calculator for all SSPs for AR
for ssp in ssps:
    try:
        intersect_src = rs.open(path / f'{ssp}_ar{removal_lvl}_ipl_absolute.tif')
        globals()[f'df_{ssp}'] = admin_bound_calculator(ssp, admin_sf, intersect_src)
        globals()[f'df_{ssp}']['option'] = 'AR'
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        globals()[f'df_{ssp}'] = pd.DataFrame()
        continue
df_ar = pd.concat([df_SSP1, df_SSP2, df_SSP3], axis=0)

# use admin_bound_calculator for all SSPs for BECCS
for ssp in ssps:
    try:
        intersect_src = rs.open(path / f'{ssp}_be{removal_lvl}_ipl_absolute.tif')
        globals()[f'df_{ssp}'] = admin_bound_calculator(ssp, admin_sf, intersect_src)
        globals()[f'df_{ssp}']['option'] = 'Bioenergy'
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        globals()[f'df_{ssp}'] = pd.DataFrame()
        continue
df_be = pd.concat([df_SSP1, df_SSP2, df_SSP3], axis=0)

# concat output dfs for AR and BE
df_options = pd.concat([df_ar, df_be], axis=0)

# calculate IPLs area
land_area_calculation(path_ipl, 'IPL_2017_2arcmin.tif', 'IPL_2017_2arcmin_km2.tif')

# use admin_bound_calculator to calc IPLs area per country
intersect_src = rs.open(path_ipl / 'IPL_2017_2arcmin_km2.tif')
df_ipl = admin_bound_calculator('all_ssps', admin_sf, intersect_src)
df_ipl = df_ipl.rename(columns={'km2': 'ipl_km2'})

# combine IPL total and IPL affected in one df and calc burdens
wab_out = pd.merge(df_options, df_ipl[['iso3', 'ipl_km2']],
                   on='iso3', how='inner')
wab_out['affected_ipl_share'] = wab_out['km2'] / wab_out['ipl_km2'] * 100

# combine effect from AR and BECCS
wab_cum = wab_out.groupby(['key', 'iso3', 'ipl_km2'], as_index=False)['km2'].sum()
wab_cum['cum_affected_ipl_share'] = wab_cum['km2'] / wab_cum['ipl_km2'] * 100
wab_cum['cum_affected_ipl_share'] = wab_cum['cum_affected_ipl_share'].round(1)
wab_cum.dropna(inplace=True)

wab_cum_ssp1 = wab_cum.query('key == "SSP1"').reset_index(drop=True)
wab_cum_ssp2 = wab_cum.query('key == "SSP2"').reset_index(drop=True)
wab_cum_ssp3 = wab_cum.query('key == "SSP3"').reset_index(drop=True)

# %% plot share of national IPLs affected by AR and BECCS
wab_dict = {'SSP1': wab_cum_ssp1, 'SSP2': wab_cum_ssp2, 'SSP3': wab_cum_ssp3}

bounds = [0, 5, 10, 15, 20, 25, 30]
norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.RdPu.N, extend='max')
cmap = mpl.cm.RdPu

for ssp in wab_dict.keys():
    df = wab_dict[ssp].copy()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.LambertAzimuthalEqualArea()})

    shape_records = list(Reader(sf_path / 'world-administrative-boundaries.shp').records())

    # plot each country with data
    for record in shape_records:
        country_iso = record.attributes['iso3']
        if country_iso in df['iso3'].values:
            value = df[df['iso3'] == country_iso]['cum_affected_ipl_share'].values[0]
            color = cmap(norm(value))
            geom = record.geometry
            ax.add_geometries([geom], ccrs.PlateCarree(), facecolor=color,
                              edgecolor='black', linewidth=0.2)

    ax.coastlines(linewidth=0.2)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, orientation='horizontal',
                        boundaries=bounds, ticks=bounds,
                        spacing='proportional', extend='max')
    cbar.ax.set_position([0.346, -0.175, 0.334, 0.5])
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(f'Share of national IPLs covered by AR and\n BECCS for removals of {cdr_sum} GtCO$_2$ [%]',
                   fontsize=11)
    plt.title(f'{ssp}', fontsize=11)
    plt.show()

# %% plot ar and beccs for target removal across SSPs
for ssp in ssps:
    try:
        ar_34 = f'{model}_Afforestation_{ssp}-34_{removal_lvl}GtCO2.tif'
        be_34 = f'{model}_BECCS_{ssp}-34_{removal_lvl}GtCO2.tif'

        ar_45 = f'{model}_Afforestation_{ssp}-45_{removal_lvl}GtCO2.tif'
        be_45 = f'{model}_BECCS_{ssp}-45_{removal_lvl}GtCO2.tif'

        ar_34 = rioxarray.open_rasterio(path / ar_34, masked=True)
        be_34 = rioxarray.open_rasterio(path / be_34, masked=True)

        ar_45 = rioxarray.open_rasterio(path / ar_45, masked=True)
        be_45 = rioxarray.open_rasterio(path / be_45, masked=True)

        ar = (ar_34 + ar_45) / 2  # average between the two available rcps
        be = (be_34 + be_45) / 2  # average between the two available rcps
        ar_km2 = ar * 1000000  # Mkm2 to km2
        be_km2 = be * 1000000  # Mkm2 to km2

        ar_bin = ar_km2.where(ar_km2.isnull(), 1)  # all=1 if not nodata
        ar_bin.rio.to_raster(path / 'ar_bin_land.tif', driver='GTiff')
        land_area_calculation(path, 'ar_bin_land.tif', 'ar_bin_land_km2.tif')
        ar_max_land_area = rioxarray.open_rasterio(path / 'ar_bin_land_km2.tif',
                                                masked=True)

        be_bin = be_km2.where(be_km2.isnull(), 1)  # all=1 if not nodata
        be_bin.rio.to_raster(path / 'be_bin_land.tif', driver='GTiff')
        land_area_calculation(path, 'be_bin_land.tif', 'be_bin_land_km2.tif')
        be_max_land_area = rioxarray.open_rasterio(path / 'be_bin_land_km2.tif',
                                                masked=True)

        ar_max_land_area = ar_max_land_area.rio.reproject_match(ar_km2)  # match
        ar_km2 = ar_km2 / ar_max_land_area * 100  # calc share per cell

        be_max_land_area = be_max_land_area.rio.reproject_match(be_km2)  # match
        be_km2 = be_km2 / be_max_land_area * 100  # calc share per cell

        perc_thres = 1  # threshold is less than x% per cell for visualization
        ar_overlay = np.where(ar_km2 >= perc_thres, ar_km2, np.where(ar_km2 < perc_thres, np.nan, ar_km2))
        ar_overlay = ar_km2.copy(data=ar_overlay)
        be_overlay = np.where(be_km2 >= perc_thres, be_km2, np.where(be_km2 < perc_thres, np.nan, be_km2))
        be_overlay = be_km2.copy(data=be_overlay)

        ar_overlay.rio.to_raster(path / 'ar_overlay.tif', driver='GTiff')
        be_overlay.rio.to_raster(path / 'be_overlay.tif', driver='GTiff')

        ar = rs.open(path / 'ar_overlay.tif')
        be = rs.open(path / 'be_overlay.tif')

        data_ar = ar.read(1)
        data_be = be.read(1)

        # get the metadata
        transform = ar.transform
        extent_ar = [transform[2], transform[2] + transform[0] * ar.width,
                     transform[5] + transform[4] * ar.height, transform[5]]

        transform = be.transform
        extent_be = [transform[2], transform[2] + transform[0] * be.width,
                     transform[5] + transform[4] * be.height, transform[5]]

        bounds_ar = [1, 20, 50, 80]
        norm_ar = mpl.colors.BoundaryNorm(bounds_ar, mpl.cm.Greens.N, extend='max')
        cmap_ar = cmr.get_sub_cmap('Greens', 0.2, 1)  # specify colormap subrange

        bounds_be = [1, 5, 10, 15]
        norm_be = mpl.colors.BoundaryNorm(bounds_be, mpl.cm.Reds.N, extend='max')
        cmap_be = cmr.get_sub_cmap('Reds', 0.2, 1)  # specify colormap subrange

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())
        img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                           origin='upper', cmap=cmap_ar, norm=norm_ar, alpha=1)
        ax.coastlines(linewidth=0.2)
        cbar_ar = plt.colorbar(img_ar, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
        cbar_ar.ax.set_position([0.465, 0, 0.1, 0.35])
        cbar_ar.ax.tick_params(labelsize=11)
        cbar_ar.set_label('Afforestation share per grid cell [%]', labelpad=1, fontsize=11)
        plt.title(f'{ssp}', fontsize=11)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())
        img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                           origin='upper', cmap=cmap_be, norm=norm_be)
        ax.coastlines(linewidth=0.2)
        cbar_be = plt.colorbar(img_be, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
        cbar_be.ax.set_position([0.465, 0, 0.1, 0.35])
        cbar_be.ax.tick_params(labelsize=11)
        cbar_be.set_label('BECCS share per grid cell [%]', labelpad=1, fontsize=11)
        plt.title(f'{ssp}', fontsize=11)
        plt.show()
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        continue

# %% plot sensitivities for SSP1
removal_steps = [2, 3, 4]
for ssp in ssps:
    for removal_step in removal_steps:
        try:
            ar_34 = f'{model}_Afforestation_{ssp}-34_{removal_step}GtCO2.tif'
            be_34 = f'{model}_BECCS_{ssp}-34_{removal_step}GtCO2.tif'

            ar_45 = f'{model}_Afforestation_{ssp}-45_{removal_step}GtCO2.tif'
            be_45 = f'{model}_BECCS_{ssp}-45_{removal_step}GtCO2.tif'

            ar_34 = rioxarray.open_rasterio(path / ar_34, masked=True)
            be_34 = rioxarray.open_rasterio(path / be_34, masked=True)

            ar_45 = rioxarray.open_rasterio(path / ar_45, masked=True)
            be_45 = rioxarray.open_rasterio(path / be_45, masked=True)

            ar = (ar_34 + ar_45) / 2  # average between the two available rcps
            be = (be_34 + be_45) / 2  # average between the two available rcps
            ar_km2 = ar * 1000000  # Mkm2 to km2
            be_km2 = be * 1000000  # Mkm2 to km2

            ar_bin = ar_km2.where(ar_km2.isnull(), 1)  # all=1 if not nodata
            ar_bin.rio.to_raster(path / 'ar_bin_land.tif', driver='GTiff')
            land_area_calculation(path, 'ar_bin_land.tif', 'ar_bin_land_km2.tif')
            ar_max_land_area = rioxarray.open_rasterio(path / 'ar_bin_land_km2.tif',
                                                    masked=True)

            be_bin = be_km2.where(be_km2.isnull(), 1)  # all=1 if not nodata
            be_bin.rio.to_raster(path / 'be_bin_land.tif', driver='GTiff')
            land_area_calculation(path, 'be_bin_land.tif', 'be_bin_land_km2.tif')
            be_max_land_area = rioxarray.open_rasterio(path / 'be_bin_land_km2.tif',
                                                    masked=True)

            ar_max_land_area = ar_max_land_area.rio.reproject_match(ar_km2)  # match
            ar_km2 = ar_km2 / ar_max_land_area * 100  # calc share per cell

            be_max_land_area = be_max_land_area.rio.reproject_match(be_km2)  # match
            be_km2 = be_km2 / be_max_land_area * 100  # calc share per cell

            perc_thres = 1  # threshold is less than x% per cell for visualization
            ar_overlay = np.where(ar_km2 >= perc_thres, ar_km2, np.where(ar_km2 < perc_thres, np.nan, ar_km2))
            ar_overlay = ar_km2.copy(data=ar_overlay)
            be_overlay = np.where(be_km2 >= perc_thres, be_km2, np.where(be_km2 < perc_thres, np.nan, be_km2))
            be_overlay = be_km2.copy(data=be_overlay)

            ar_overlay.rio.to_raster(path / 'ar_overlay.tif', driver='GTiff')
            be_overlay.rio.to_raster(path / 'be_overlay.tif', driver='GTiff')

            ar = rs.open(path / 'ar_overlay.tif')
            be = rs.open(path / 'be_overlay.tif')

            data_ar = ar.read(1)
            data_be = be.read(1)

            # get the metadata
            transform = ar.transform
            extent_ar = [transform[2], transform[2] + transform[0] * ar.width,
                         transform[5] + transform[4] * ar.height, transform[5]]

            transform = be.transform
            extent_be = [transform[2], transform[2] + transform[0] * be.width,
                         transform[5] + transform[4] * be.height, transform[5]]

            bounds_ar = [1, 20, 50, 80]
            norm_ar = mpl.colors.BoundaryNorm(bounds_ar, mpl.cm.Greens.N, extend='max')
            cmap_ar = cmr.get_sub_cmap('Greens', 0.2, 1)  # specify colormap subrange

            bounds_be = [1, 5, 10, 15]
            norm_be = mpl.colors.BoundaryNorm(bounds_be, mpl.cm.Reds.N, extend='max')
            cmap_be = cmr.get_sub_cmap('Reds', 0.2, 1)  # specify colormap subrange

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())
            img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                               origin='upper', cmap=cmap_ar, norm=norm_ar, alpha=1)
            ax.coastlines(linewidth=0.2)
            cbar_ar = plt.colorbar(img_ar, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
            cbar_ar.ax.set_position([0.465, 0, 0.1, 0.35])
            cbar_ar.ax.tick_params(labelsize=11)
            cbar_ar.set_label('Afforestation share per grid cell [%]', labelpad=1, fontsize=11)
            plt.title(f'{removal_step} GtCO$_2$ in {ssp}', fontsize=11)

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.LambertAzimuthalEqualArea())  # choose projection | LambertAzimuthalEqualArea())
            img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                               origin='upper', cmap=cmap_be, norm=norm_be)
            ax.coastlines(linewidth=0.2)
            cbar_be = plt.colorbar(img_be, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
            cbar_be.ax.set_position([0.465, 0, 0.1, 0.35])
            cbar_be.ax.tick_params(labelsize=11)
            cbar_be.set_label('BECCS share per grid cell [%]', labelpad=1, fontsize=11)
            plt.title(f'{removal_step} GtCO$_2$ in {ssp}', fontsize=11)
            plt.show()
        except Exception as e:
            print(f'Error processing {ssp}: {e}')
            continue
