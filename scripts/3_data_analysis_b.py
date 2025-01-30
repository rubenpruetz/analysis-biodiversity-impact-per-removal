
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import rioxarray
import rasterio as rs
from time import time
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cmasher as cmr
from required_functions import *
import shapefile
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_ipl = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ipl_maps/01_Data')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_cz = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/koppen_geiger_maps/1991_2020')
path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
energy_crop_share = pd.read_csv(path_all / 'share_energy_crops_estimates.csv')
lookup_mi_cdr_df = pd.read_csv(path_all / 'lookup_table_cdr_files_all_models.csv')
lookup_mi_cdr_df['year'] = lookup_mi_cdr_df['year'].astype(str)

# %% choose model to run the script with
model = 'AIM'  # options: 'GLOBIOM' or 'AIM' or 'IMAGE'

if model == 'GLOBIOM':
    path = path_globiom
    model_setup = 'MESSAGE-GLOBIOM 1.0'
    removal_lvl = 2
elif model == 'AIM':
    path = path_aim
    model_setup = 'AIM/CGE 2.0'
    removal_lvl = 2
elif model == 'IMAGE':
    path = path_image
    model_setup = 'IMAGE 3.0.1'
    removal_lvl = 2

# %% get temperatures for SSP-RCP combinations
all_years = [str(year) for year in range(2020, 2101)]

models = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'IMAGE 3.0.1']
scenarios = ['SSP1-Baseline', 'SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45',
             'SSP2-Baseline', 'SSP2-19', 'SSP2-26', 'SSP2-34', 'SSP2-45',
             'SSP2-60', 'SSP3-Baseline', 'SSP3-34', 'SSP3-45', 'SSP3-60']
variable = ['AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile']

ar6_data = ar6_db.loc[ar6_db['Variable'].isin(variable)]
ar6_data = ar6_data.loc[ar6_data['Model'].isin(models)]
ar6_data = ar6_data.loc[ar6_data['Scenario'].isin(scenarios)]
ar6_data = ar6_data.round(1)  # round temperatures

# allow no temperature decline by calculating peak warming up until each year
for year in range(2021, 2101):
    cols_til_year = ar6_data.loc[:, '2020':str(year)]
    ar6_data[f'{year}_max'] = cols_til_year.max(axis=1)

cols = ['Model', 'Scenario', '2020'] + [f'{year}_max' for year in range(2021, 2101)]
ar6_data_stab = ar6_data[cols]
ar6_data_stab = ar6_data_stab.rename(columns={f'{year}_max': str(year) for year in all_years})

ar6_data = ar6_data[['Model', 'Scenario'] + all_years].copy()

# %% calculate AR and BECCS removals for SSP-RCP combinations
numeric_cols20 = [str(year) for year in range(2020, 2110, 10)]

if model == 'GLOBIOM':
    ar_variable = 'Carbon Sequestration|Land Use|Afforestation'
elif model == 'AIM':
    ar_variable = 'Carbon Sequestration|Land Use|Afforestation'
elif model == 'IMAGE':  # for IMAGE afforestation is not available
    ar_variable = 'Carbon Sequestration|Land Use'

ar6_db = ar6_db.loc[ar6_db['Model'].isin([model_setup]) & ar6_db['Scenario'].isin(scenarios)]
cdr = ar6_db[['Scenario', 'Variable'] + numeric_cols20].copy()
cdr_array = ['Carbon Sequestration|CCS|Biomass',
             ar_variable]
cdr = cdr[cdr['Variable'].isin(cdr_array)]
cdr[numeric_cols20] = cdr[numeric_cols20].clip(lower=0)  # set negative values to zero
cdr = cdr.melt(id_vars=['Scenario', 'Variable'], var_name='Year', value_name='Removal')
cdr['Removal'] = cdr['Removal'] * 0.001  # Mt to Gt
cdr['Year'] = pd.to_numeric(cdr['Year'])

ar_removal = cdr[cdr['Variable'] == ar_variable]
ar_removal['Variable'] = 'Afforestation'

beccs_removal = cdr[cdr['Variable'] == 'Carbon Sequestration|CCS|Biomass']
beccs_removal['Variable'] = 'BECCS'

# calculate BECCS removal through energy crops only (no residues)
ec_share = energy_crop_share.loc[energy_crop_share['Model'].isin([model])]
beccs_removal = pd.merge(beccs_removal,
                         ec_share[['Scenario', 'Year', 'Share_energy_crops']],
                         on=['Scenario', 'Year'])
beccs_removal['Removal'] = beccs_removal['Removal'] * beccs_removal['Share_energy_crops']
beccs_removal = beccs_removal[['Scenario', 'Year', 'Variable', 'Removal']].copy()

cdr = pd.concat([ar_removal, beccs_removal], axis=0)
cdr.rename(columns={'Variable': 'mitigation_option'}, inplace=True)

# %% choose between biodiv recovery or no recovery after peak warming

temperature_decline = 'not_allowed'  # options: 'allowed' or 'not_allowed'

if temperature_decline == 'allowed':
    warm_file = ar6_data.copy()
elif temperature_decline == 'not_allowed':
    warm_file = ar6_data_stab.copy()

bio_select = warm_file.set_index(['Model', 'Scenario'])
bio_select = 'bio' + \
    bio_select.select_dtypes(include=np.number).astype(str) + '_bin.tif'
bio_select.reset_index(inplace=True)

# rename models for the subsequent step
bio_select.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                              'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                              'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

# %% calculate CDR land impact over time
years = ['2020', '2040', '2060', '2080', '2100']
lookup_sub_yrs = lookup_mi_cdr_df.copy()
lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

# load climate zone files
cz1 = rioxarray.open_rasterio(path_cz / 'clim_zon_class1.tif', masked=True)
cz2 = rioxarray.open_rasterio(path_cz / 'clim_zon_class2.tif', masked=True)
cz3 = rioxarray.open_rasterio(path_cz / 'clim_zon_class3.tif', masked=True)
cz4 = rioxarray.open_rasterio(path_cz / 'clim_zon_class4.tif', masked=True)
cz5 = rioxarray.open_rasterio(path_cz / 'clim_zon_class5.tif', masked=True)

start = time()  # runtime monitoring

def overlay_calculator(input_tif,  # land use model input file (string)
                       filepath,  # filepath input file + / (string)
                       file_year,  # year of input file (string)
                       file_scenario,  # input file SSP-RCP scenario (string)
                       mitigation_option,  # 'Afforestation' or 'Bioenergy'
                       lu_model):  # GLOBIOM or AIM or IMAGE

    # load files for CDR and refugia
    land_use = rioxarray.open_rasterio(filepath / f'{lu_model}_{input_tif}',
                                       masked=True)

    bio_file = ''.join(bio_select[(bio_select['Model'] == lu_model) &
                                  (bio_select['Scenario'] == file_scenario)][file_year])

    refugia = rioxarray.open_rasterio(path_uea / bio_file, masked=True)
    refugia = refugia.rio.reproject_match(land_use)  # align files

    # calculate land impact on refugia (global)
    cdr_in_bio = land_use * refugia

    # calculate refugia extent
    bio_area = land_area_calculation(path_uea, bio_file)

    # calculate aggregated area "losses" and refugia
    cdr_in_bio_agg = pos_val_summer(cdr_in_bio, squeeze=True)
    bio_area_agg = pos_val_summer(bio_area, squeeze=True)

    # calculate warming and land impact on refugia (regional)
    cdr_in_bio_regs = []
    bio_area_regs = []

    cz_values = [cz1, cz2, cz3, cz4, cz5]

    for cz in cz_values:
        cz_m = cz.rio.reproject_match(land_use)  # ensure consistent match
        cdr_in_bio_cz = (land_use * refugia) * cz_m

        refugia_cz = refugia * cz_m
        refugia_cz.rio.to_raster(path_uea / 'refugia_cz_temp.tif', driver='GTiff')
        refugia_cz = land_area_calculation(path_uea, 'refugia_cz_temp.tif')

        cdr_in_bio_cz = pos_val_summer(cdr_in_bio_cz, squeeze=True)
        refugia_cz = pos_val_summer(refugia_cz, squeeze=True)

        # calculate regional area "losses" and refugia
        cdr_in_bio_regs.append(cdr_in_bio_cz)
        bio_area_regs.append(refugia_cz)

    return cdr_in_bio_agg, bio_area_agg, cdr_in_bio_regs, bio_area_regs

# use overlay_calculator
def process_row(row):
    input_tif = row['file_name']
    file_year = row['year']
    file_scenario = row['scenario']
    mitigation_option = row['mitigation_option']

    try:
        # run overlay_calculator for all scenarios to retrieve areas as outputs
        cdr_in_bio_agg, bio_area_agg, cdr_in_bio_regs, \
            bio_area_regs = overlay_calculator(input_tif,
                                               path,
                                               file_year,
                                               file_scenario,
                                               mitigation_option,
                                               model)

        # create a dictionary with the calculated values
        result_dict = {
            'scenario': file_scenario,
            'mitigation_option': mitigation_option,
            'year': file_year,
            'bio_area': bio_area_agg,
            'cdr_in_bio': cdr_in_bio_agg,
            'bio_area_reg': bio_area_regs,
            'cdr_in_bio_reg': cdr_in_bio_regs}

        return result_dict

    except:
        print(f'Unsuccessful for file: {input_tif}')
        return {
            'scenario': file_scenario,
            'mitigation_option': mitigation_option,
            'year': file_year,
            'bio_area': float('nan'),
            'cdr_in_bio': float('nan'),
            'bio_area_reg': [float('nan')] * 5,
            'cdr_in_bio_reg': [float('nan')] * 5}


area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                         axis=1).values)
area_df = area_df.reset_index(drop=True)

cz_columns = pd.DataFrame(area_df['bio_area_reg'].to_list(),
                          columns=['bio_area_cz1',
                                   'bio_area_cz2',
                                   'bio_area_cz3',
                                   'bio_area_cz4',
                                   'bio_area_cz5'])
area_df = pd.concat([area_df.drop(columns='bio_area_reg'), cz_columns], axis=1)

cz_columns = pd.DataFrame(area_df['cdr_in_bio_reg'].to_list(),
                          columns=['cdr_in_bio_cz1',
                                   'cdr_in_bio_cz2',
                                   'cdr_in_bio_cz3',
                                   'cdr_in_bio_cz4',
                                   'cdr_in_bio_cz5'])
area_df = pd.concat([area_df.drop(columns='cdr_in_bio_reg'), cz_columns], axis=1)

end = time()
print(f'Runtime {(end - start) / 60} min')

area_df['alloc_perc'] = area_df['cdr_in_bio'] / area_df['bio_area'] * 100
for i in range(1, 6):  # calculate land loss percentages for all climate zones
    area_df[f'alloc_perc_cz{i}'] = area_df[f'cdr_in_bio_cz{i}'] / area_df[f'bio_area_cz{i}'] * 100

area_df['SSP'] = area_df['scenario'].str.split('-').str[0]
area_df['RCP'] = area_df['scenario'].str.split('-').str[1]
area_df.rename(columns={'year': 'Year'}, inplace=True)
area_df['Model'] = f'{model}'

# merge with removal data
area_df = pd.merge(area_df, cdr, how='inner', on=['Scenario', 'Year', 'mitigation_option'])

area_df.to_csv(path / f'{model}_area_df_clim_zone_temp_decline_{temperature_decline}.csv', index=False)

# %% plot land allocation within refugia across scenarios

paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
area_df = load_and_concat('area_df_clim_zone', paths)

cdr_option = 'Afforestation'
area_df = area_df.query('mitigation_option == @cdr_option')

rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
           '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}
all_rcps = sorted(area_df['RCP'].unique())

fig, axes = plt.subplots(4, 6, figsize=(12, 8), sharex=True, sharey=True)
sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 0])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), hue_order=all_rcps, legend=True, ax=axes[1, 0])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 0])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 1])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 1])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz1',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 1])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 2])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 2])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz2',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 2])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 3])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 3])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz3',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 3])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 4])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 4])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz4',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 4])

sns.lineplot(data=area_df.query('Model == "AIM"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[0, 5])
sns.lineplot(data=area_df.query('Model == "GLOBIOM"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[1, 5])
sns.lineplot(data=area_df.query('Model == "IMAGE"'), x='Year', y='alloc_perc_cz5',
             palette=rcp_pal, hue='RCP', errorbar=('pi', 100), legend=False, ax=axes[3, 5])

axes[1, 0].legend(bbox_to_anchor=(-0.05, 2.6), loc='upper left', ncols=12,
                  columnspacing=0.8, handletextpad=0.1)

axes[0, 0].set_title('Global')
axes[0, 1].set_title('Tropical')
axes[0, 2].set_title('Arid')
axes[0, 3].set_title('Temperate')
axes[0, 4].set_title('Cold')
axes[0, 5].set_title('Polar')

axes[3, 0].set_xlabel('')
axes[3, 1].set_xlabel('')
axes[3, 2].set_xlabel('')
axes[3, 3].set_xlabel('')
axes[3, 4].set_xlabel('')
axes[3, 5].set_xlabel('')

axes[0, 0].set_ylabel('AIM')
axes[1, 0].set_ylabel('GLOBIOM')
axes[3, 0].set_ylabel('IMAGE')

fig.supylabel(f'Remaining refugia allocated for {cdr_option} [%] (SSP1-SSP3 range)', 
              x=0.05, va='center')

for ax in axes.flat:
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2060, 2100])

plt.subplots_adjust(hspace=0.15)
plt.subplots_adjust(wspace=0.4)
sns.despine()
plt.show()

# %% maps of refugia land impact of CDR across SSP1-3 for a certain warming level

rcp_lvl = '34'  # select RCP level (without dot)
ssps = ['SSP1', 'SSP2', 'SSP3']

for ssp in ssps:
    try:
        ar = f'{model}_Afforestation_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'
        be = f'{model}_BECCS_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'

        ar = rioxarray.open_rasterio(path / ar, masked=True)
        be = rioxarray.open_rasterio(path / be, masked=True)

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

        bounds_ar = [1, 5, 10, 20]
        norm_ar = mpl.colors.BoundaryNorm(bounds_ar, mpl.cm.Greens.N, extend='max')
        cmap_ar = cmr.get_sub_cmap('Greens', 0.2, 1)  # specify colormap subrange

        bounds_be = [1, 5, 10, 20]
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

bounds = [0, 1, 5, 10, 15, 20, 25]
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

for land_info in land_infos:
    for ssp in ssps:

        file_in = f'{model}_{land_info}_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'
        file_out = f'{model}_{land_info}_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2_highres.tif'

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
            print(f'Error processing {ssp}')

# %% plot impact on IPLs
ipl = rioxarray.open_rasterio(path_ipl / 'IPL_2017_2arcmin.tif', masked=True)
bin_land = ipl.where(ipl.isnull(), 1)  # all=1 if not nodata
bin_land.rio.to_raster(path_ipl / 'bin_land.tif', driver='GTiff')
land_area_calculation(path_ipl, 'bin_land.tif', 'bin_land_km2.tif')
max_land_area = rioxarray.open_rasterio(path_ipl / 'bin_land_km2.tif',
                                        masked=True)

for ssp in ssps:
    try:
        ar = f'{model}_Afforestation_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2_highres.tif'
        be = f'{model}_BECCS_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2_highres.tif'

        ar = rioxarray.open_rasterio(path / ar, masked=True)
        be = rioxarray.open_rasterio(path / be, masked=True)

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

bounds = [0, 1, 5, 10, 15, 20, 25]
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
        ar = f'{model}_Afforestation_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'
        be = f'{model}_BECCS_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'

        ar = rioxarray.open_rasterio(path / ar, masked=True)
        be = rioxarray.open_rasterio(path / be, masked=True)

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

        bounds_ar = [1, 5, 10, 20]
        norm_ar = mpl.colors.BoundaryNorm(bounds_ar, mpl.cm.Greens.N, extend='max')
        cmap_ar = cmr.get_sub_cmap('Greens', 0.2, 1)  # specify colormap subrange

        bounds_be = [1, 5, 10, 20]
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
            ar = f'{model}_Afforestation_{ssp}-{rcp_lvl}_{removal_step}GtCO2.tif'
            be = f'{model}_BECCS_{ssp}-{rcp_lvl}_{removal_step}GtCO2.tif'

            ar = rioxarray.open_rasterio(path / ar, masked=True)
            be = rioxarray.open_rasterio(path / be, masked=True)

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