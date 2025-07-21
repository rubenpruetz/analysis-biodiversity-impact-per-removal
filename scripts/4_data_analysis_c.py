
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import rioxarray
import rasterio as rs
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from required_functions import *
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_hotspots = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ar6_hotspots')

# %% estimate land CDR conflict with SDG 15.5 based on different criteria
hotspots = rioxarray.open_rasterio(path_hotspots / 'ar6_hotspots_10arcmin.tif')
res_bio = rioxarray.open_rasterio(path_uea / 'bio1.8_bin.tif', masked=True)  # change file if required

# estimate hotspot areas that a resilient to selected warming
hotspot_repro = hotspots.rio.reproject_match(res_bio)
hs_resil = hotspot_repro * res_bio

# estimate reduction in land allocation when excluding areas
models = ['AIM', 'GLOBIOM', 'IMAGE']
cdr_options = ['Afforestation', 'BECCS']
years = [2030, 2050, 2100]

exclu_df = pd.DataFrame(columns=['Model', 'CDR_option', 'Year', 'CDR_land',
                                 'CDR_in_hs', 'CDR_in_hs_res', 'CDR_in_bio'])
for model in models:
    for cdr_option in cdr_options:
        for year in years:

            if model == 'GLOBIOM':
                path = path_globiom
            elif model == 'AIM':
                path = path_aim
            elif model == 'IMAGE':
                path = path_image

            try:
                cdr_land = f'{model}_{cdr_option}_SSP2-26_{year}.tif'  # change scenario if required

                cdr = rioxarray.open_rasterio(path / cdr_land, masked=True)
                tot_cdr_area = pos_val_summer(cdr, squeeze=True)

                # CDR in biodiversity hotspots
                cdr_repro = cdr.rio.reproject_match(hotspots)
                cdr_in_hs = cdr_repro * hotspots
                cdr_in_hs = pos_val_summer(cdr_in_hs, squeeze=True)

                # CDR in biodiversity hotspots resilient to warming
                cdr_repro = cdr.rio.reproject_match(hs_resil)
                cdr_in_hs_res = cdr_repro * hs_resil
                cdr_in_hs_res = pos_val_summer(cdr_in_hs_res, squeeze=True)

                # CDR in warming resilient refugia
                cdr_repro = cdr.rio.reproject_match(res_bio)
                cdr_in_bio = cdr_repro * res_bio
                cdr_in_bio = pos_val_summer(cdr_in_bio, squeeze=True)

                new_row = pd.DataFrame({'Model': [model],
                                        'CDR_option': [cdr_option],
                                        'Year': [year],
                                        'CDR_land': [tot_cdr_area],
                                        'CDR_in_hs': [cdr_in_hs],
                                        'CDR_in_hs_res': [cdr_in_hs_res],
                                        'CDR_in_bio': [cdr_in_bio]})

                exclu_df = pd.concat([exclu_df, new_row], ignore_index=True)
            except Exception as e:
                print(f'Error processing {model}: {e}')
                continue

# sum afforestation and BECCS values to get overall land intensive CDR values
exclu_df_sum = exclu_df.groupby(['Model', 'Year'])[['CDR_land',
                                                    'CDR_in_hs',
                                                    'CDR_in_hs_res',
                                                    'CDR_in_bio']].agg('sum')
exclu_df_sum.reset_index(inplace=True)
exclu_df_sum['CDR_option'] = 'Forestation & BECCS'
exclu_df = pd.concat([exclu_df, exclu_df_sum])

# calculate share of overlap of CDR land with biodiversity criteria
exclu_df['Reduct_hs'] = exclu_df['CDR_in_hs'] / exclu_df['CDR_land'] * 100
exclu_df['Reduct_hs_res'] = exclu_df['CDR_in_hs_res'] / exclu_df['CDR_land'] * 100
exclu_df['Reduct_bio'] = exclu_df['CDR_in_bio'] / exclu_df['CDR_land'] * 100

exclu_df = pd.melt(exclu_df, id_vars=['Model', 'CDR_option', 'Year'],
                   value_vars=['Reduct_hs', 'Reduct_hs_res', 'Reduct_bio'],
                   var_name='Reduct_criteria',
                   value_name='Value')

exclu_df.replace({'CDR_option': {'Afforestation': 'Forestation'}}, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)

model_colors = {'AIM': 'darkslategrey', 'GLOBIOM': 'blueviolet',
                'IMAGE': 'royalblue'}
cdr_colors = {'Forestation': 'crimson', 'BECCS': 'darkorange',
              'Forestation & BECCS': 'lightsteelblue'}

sns.barplot(data=exclu_df.query('Reduct_criteria == "Reduct_hs_res"'), x='Year',
            y='Value', hue='CDR_option', legend=True, alpha=0.6, palette=cdr_colors,
            gap=0, estimator='median', errorbar=('pi', 100), ax=axes[0])

for model, color in model_colors.items():
    sns.stripplot(data=exclu_df.query(f'Model == "{model}" & Reduct_criteria == "Reduct_hs_res"'),
                  x='Year', y='Value', hue='CDR_option', dodge=True, jitter=0,
                  s=5, marker='o', edgecolor=color, linewidth=5, legend=False, ax=axes[0])

sns.barplot(data=exclu_df.query('Reduct_criteria == "Reduct_hs"'), x='Year',
            y='Value', hue='CDR_option', legend=False, alpha=0.6, palette=cdr_colors,
            gap=0, estimator='median', errorbar=('pi', 100), ax=axes[1])

for model, color in model_colors.items():
    sns.stripplot(data=exclu_df.query(f'Model == "{model}" & Reduct_criteria == "Reduct_hs"'),
                  x='Year', y='Value', hue='CDR_option', dodge=True, jitter=0,
                  s=5, marker='o', edgecolor=color, linewidth=5, legend=False, ax=axes[1])

sns.barplot(data=exclu_df.query('Reduct_criteria == "Reduct_bio"'), x='Year',
            y='Value', hue='CDR_option', legend=False, alpha=0.6, palette=cdr_colors,
            gap=0, estimator='median', errorbar=('pi', 100), ax=axes[2])

for model, color in model_colors.items():
    sns.stripplot(data=exclu_df.query(f'Model == "{model}" & Reduct_criteria == "Reduct_bio"'),
                  x='Year', y='Value', hue='CDR_option', dodge=True, jitter=0,
                  s=5, marker='o', edgecolor=color, linewidth=5, legend=False, ax=axes[2])

model_patches = [mpatches.Patch(color=color, label=model) for model, color in model_colors.items()]
legend1 = axes[0].legend(handles=model_patches, bbox_to_anchor=(1.85, 1.1),
                         loc='upper left', ncols=5, columnspacing=0.8,
                         handletextpad=0.5, frameon=False, fontsize=12)

axes[0].legend(bbox_to_anchor=(-0.05, 1.1), loc='upper left', ncols=5,
               columnspacing=0.8, handletextpad=0.5, frameon=False, fontsize=12)
axes[0].add_artist(legend1)

axes[0].set_xlabel('Exclusion of land within 1.8 °C resilient \nbiodiversity hotspots', fontsize=11)
axes[1].set_xlabel('Exclusion of land within current \nbiodiversity hotspots', fontsize=11)
axes[2].set_xlabel('Exclusion of land within 1.8 °C resilient \nbiodiversity refugia', fontsize=11)
axes[0].set_ylabel(f'Share of CDR land not available for allocation in SSP2-26 [%] \n(median and min-max range across models)',
                   fontsize=12)

for ax in axes.flat:
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

plt.subplots_adjust(wspace=0.1)
sns.despine()
plt.show()

# %% plot hotspot areas of concern in terms of model agreement
# make files from different models binary (cell threshold >= 10% of max area)
for model in models:
    for cdr_option in cdr_options:
        if model == 'GLOBIOM':
            path = path_globiom
        elif model == 'AIM':
            path = path_aim
        elif model == 'IMAGE':
            path = path_image

        land_in = f'{model}_{cdr_option}_SSP2-26_2100.tif'  # change scenario if required
        land_temp = f'{model}_{cdr_option}_SSP2-26_2100_temp.tif'
        land_out = f'{model}_{cdr_option}_SSP2-26_2100_bin.tif'

        land_in = rioxarray.open_rasterio(path / land_in, masked=True)
        bin_land = land_in.where(land_in.isnull(), 1)  # all=1 if not nodata
        bin_land.rio.to_raster(path / 'bin_land.tif', driver='GTiff')

        land_area_calculation(path, 'bin_land.tif', f'{model}_max_land_area_km2.tif')
        land_max = rioxarray.open_rasterio(path / f'{model}_max_land_area_km2.tif', masked=True)

        land_allo_share = land_in / land_max  # estimate cell shares allocated
        land_allo_share.rio.to_raster(path / land_temp , driver='GTiff')
        binary_converter(land_temp, path, 0.10, land_out)  # adjust threshold if needed

for cdr_option in cdr_options:
    aim_land = rioxarray.open_rasterio(path_aim / f'AIM_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)
    globiom_land = rioxarray.open_rasterio(path_globiom / f'GLOBIOM_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)
    image_land = rioxarray.open_rasterio(path_image / f'IMAGE_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)

    # calculate model agreement in refugia
    aim_land = aim_land.rio.reproject_match(res_bio)
    globiom_land = globiom_land.rio.reproject_match(res_bio)
    image_land = image_land.rio.reproject_match(res_bio)
    agree_in_res_bio = (aim_land + globiom_land + image_land) * res_bio
    agree_in_res_bio.rio.to_raster(path_all / f'mi_{cdr_option}_SSP2-26_2100_index_in_res_bio.tif', driver='GTiff')
    binary_converter(f'mi_{cdr_option}_SSP2-26_2100_index_in_res_bio.tif',
                     path_all, 2,
                     f'mi_{cdr_option}_SSP2-26_2100_index_in_res_bio.tif')

    # calculate model agreement in resilient hotspot
    aim_land = aim_land.rio.reproject_match(hs_resil)
    globiom_land = globiom_land.rio.reproject_match(hs_resil)
    image_land = image_land.rio.reproject_match(hs_resil)
    agree_in_res_bio = (aim_land + globiom_land + image_land) * hs_resil
    agree_in_res_bio.rio.to_raster(path_all / f'mi_{cdr_option}_SSP2-26_2100_index_in_res_hs.tif', driver='GTiff')
    binary_converter(f'mi_{cdr_option}_SSP2-26_2100_index_in_res_hs.tif',
                     path_all, 2,
                     f'mi_{cdr_option}_SSP2-26_2100_index_in_res_hs.tif')

hs_resil.rio.to_raster(path_hotspots / 'hs_resilient.tif', driver='GTiff')

# %%

ar = rs.open(path_all / 'mi_Afforestation_SSP2-26_2100_index_in_res_bio.tif')
be = rs.open(path_all / 'mi_BECCS_SSP2-26_2100_index_in_res_bio.tif')
ar_hs = rs.open(path_all / 'mi_Afforestation_SSP2-26_2100_index_in_res_hs.tif')
be_hs = rs.open(path_all / 'mi_BECCS_SSP2-26_2100_index_in_res_hs.tif')
refug = rs.open(path_uea / 'bio1.8_bin.tif')
hs_resil = rs.open(path_hotspots / 'hs_resilient.tif')

data_ar = ar.read(1)
data_be = be.read(1)
data_ar_hs = ar_hs.read(1)
data_be_hs = be_hs.read(1)
data_refug = refug.read(1)
data_hs_resil = hs_resil.read(1)

# get the metadata
transform = ar.transform
extent_ar = [transform[2], transform[2] + transform[0] * ar.width,
             transform[5] + transform[4] * ar.height, transform[5]]

transform = be.transform
extent_be = [transform[2], transform[2] + transform[0] * be.width,
             transform[5] + transform[4] * be.height, transform[5]]

transform = ar.transform
extent_ar_hs = [transform[2], transform[2] + transform[0] * ar_hs.width,
             transform[5] + transform[4] * ar_hs.height, transform[5]]

transform = be.transform
extent_be_hs = [transform[2], transform[2] + transform[0] * be_hs.width,
             transform[5] + transform[4] * be_hs.height, transform[5]]

transform = refug.transform
extent_refug = [transform[2], transform[2] + transform[0] * refug.width,
                transform[5] + transform[4] * refug.height, transform[5]]

transform = hs_resil.transform
extent_hs = [transform[2], transform[2] + transform[0] * hs_resil.width,
             transform[5] + transform[4] * hs_resil.height, transform[5]]

color_g1 = ListedColormap([(0, 0, 0, 0), 'gainsboro'])
norm_g1 = BoundaryNorm([0, 1], color_g1.N)

color_g2 = ListedColormap([(0, 0, 0, 0), 'grey'])
norm_g2 = BoundaryNorm([0, 1], color_g2.N)

color_bio = ListedColormap([(0, 0, 0, 0), 'gold'])
norm_bio = BoundaryNorm([0, 1], color_bio.N)

color_hs = ListedColormap([(0, 0, 0, 0), 'crimson'])
norm_hs = BoundaryNorm([0, 1], color_hs.N)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

img_re = ax.imshow(data_refug, extent=extent_refug, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_g1, norm=norm_g1)

img_hs = ax.imshow(data_hs_resil, extent=extent_hs, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_g2, norm=norm_g2)

img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_bio, norm=norm_bio)

img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_bio, norm=norm_bio)

img_ar_hs = ax.imshow(data_ar_hs, extent=extent_ar_hs, transform=ccrs.PlateCarree(),
                      origin='upper', cmap=color_hs, norm=norm_hs)

img_be_hs = ax.imshow(data_be_hs, extent=extent_be_hs, transform=ccrs.PlateCarree(),
                      origin='upper', cmap=color_hs, norm=norm_hs)

ax.coastlines(linewidth=0.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.2)

legend_patches = [
    mpatches.Patch(color='gold', label='CDR in refugia'),
    mpatches.Patch(color='crimson', label='CDR in refugia & hotspot'),
    mpatches.Patch(color='gainsboro', label='Refugia'),
    mpatches.Patch(color='grey', label='Hotspot')]

legend = ax.legend(bbox_to_anchor=(-0.01, 0.07), handles=legend_patches, ncols=1,
          loc='lower left', fontsize=9.5, columnspacing=0.8, handletextpad=0.5,
          borderpad=1.5, frameon=True)

legend.get_frame().set_alpha(1)
legend.get_frame().set_edgecolor('none')

plt.show()

