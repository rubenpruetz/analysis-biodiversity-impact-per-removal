
# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
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
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_magpie = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/magpie_maps')
path_hotspots = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ar6_hotspots')
path_ref_pot = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/reforest_potential')
path_beccs_pot = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/Braun_et_al_2024_PB_BECCS/Results/1_source_data_figures/Fig2')

# %% estimate land CDR conflict with SDG 15.5 based on different criteria
hotspots = rioxarray.open_rasterio(path_hotspots / 'ar6_hotspots_10arcmin.tif',
                                   masked=True)
res_bio = rioxarray.open_rasterio(path_uea / 'bio1.8_bin.tif', masked=True)  # change file if required

# estimate hotspot areas that a resilient to selected warming
hotspot_repro = hotspots.rio.reproject_match(res_bio)
hs_resil = hotspot_repro * res_bio

# estimate reduction in land allocation when excluding areas
models = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE', 'MAgPIE']
cdr_options = ['Afforestation', 'BECCS']
years = [2030, 2050, 2100]

exclu_df = pd.DataFrame(columns=['Model', 'CDR_option', 'Year', 'CDR_land',
                                 'CDR_in_hs', 'CDR_in_hs_res', 'CDR_in_bio'])
for model in models:
    for cdr_option in cdr_options:
        for year in years:

            if model == 'AIM':
                path = path_aim
            elif model == 'GCAM':
                path = path_gcam
            elif model == 'GLOBIOM':
                path = path_globiom
            elif model == 'IMAGE':
                path = path_image
            elif model == 'MAgPIE':
                path = path_magpie

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
exclu_df_sum['CDR_option'] = 'Both'
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

# plot reduction in land allocated for CDR
fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)

model_colors = {'AIM': 'darkslategrey', 'GCAM': '#997700', 'GLOBIOM': 'blueviolet',
                'IMAGE': 'royalblue', 'MAgPIE': '#994455'}
cdr_colors = {'Forestation': 'crimson', 'BECCS': 'darkorange',
              'Both': 'lightsteelblue'}

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

model_patches = [Line2D([0], [0], marker='o', color='w', label=label,
                        markerfacecolor=color, markeredgecolor='none', markersize=10)
                 for label, color in model_colors.items()]

legend1 = axes[0].legend(handles=model_patches, bbox_to_anchor=(1.3, 1.1),
                         loc='upper left', ncols=5, columnspacing=0.6,
                         handletextpad=0.5, frameon=False, fontsize=12)

axes[0].legend(bbox_to_anchor=(-0.05, 1.1), loc='upper left', ncols=5,
               columnspacing=0.6, handletextpad=0.5, frameon=False, fontsize=12)
axes[0].add_artist(legend1)

axes[0].set_xlabel('No CDR within 1.8 °C resilient \nbiodiversity hotspots', fontsize=11)
axes[1].set_xlabel('No CDR within current \nbiodiversity hotspots', fontsize=11)
axes[2].set_xlabel('No CDR within 1.8 °C resilient \nclimate refugia', fontsize=11)
axes[0].set_ylabel('Reduction in land allocated for CDR in SSP2-26 [%] \n(median and individual model estimate)',
                   fontsize=12)

for ax in axes.flat:
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)

plt.subplots_adjust(wspace=0.1)
sns.despine()
plt.show()

# %% plot hotspot areas of concern in terms of model agreement
# make files from different models binary (cell share >= x% of max area)
for model in models:
    for cdr_option in cdr_options:
        if model == 'AIM':
            path = path_aim
        elif model == 'GCAM':
            path = path_gcam
        elif model == 'GLOBIOM':
            path = path_globiom
        elif model == 'IMAGE':
            path = path_image
        elif model == 'MAgPIE':
            path = path_magpie

        land_in = f'{model}_{cdr_option}_SSP2-26_2100.tif'  # change scenario if required
        land_temp = f'{model}_{cdr_option}_SSP2-26_2100_temp.tif'
        land_out = f'{model}_{cdr_option}_SSP2-26_2100_bin.tif'

        land_in = rioxarray.open_rasterio(path / land_in, masked=True)

        # calculate grid area based on arbitrarily chosen input file
        if model not in ['IMAGE', 'MAgPIE', 'AIM']:  # for these models use predefined file
            bin_land = land_in.where(land_in.isnull(), 1)  # all=1 if not nodata
            bin_land.rio.to_raster(path / 'bin_land.tif', driver='GTiff')
            land_area_calculation(path, 'bin_land.tif', f'{model}_max_land_area_km2.tif')

        land_max = rioxarray.open_rasterio(path / f'{model}_max_land_area_km2.tif', masked=True)
        land_allo_share = land_in / land_max  # estimate cell shares allocated
        land_allo_share.rio.to_raster(path / land_temp , driver='GTiff')
        binary_converter(land_temp, path, 0.1, land_out)  # adjust cell threshold if needed

# load area-based criteria for beneficia/harmful effects on biodiversity
ref_suit = rioxarray.open_rasterio(path_ref_pot / 'ref_suit.tif', masked=True)
ref_not_suit = rioxarray.open_rasterio(path_ref_pot / 'ref_not_suit.tif', masked=True)
beccs_suit = rioxarray.open_rasterio(path_beccs_pot / 'beccs_suit.tif', masked=True)
beccs_not_suit = rioxarray.open_rasterio(path_beccs_pot / 'beccs_not_suit.tif', masked=True)

# calculate model agreement in refugia and check if likely positive or negative
for cdr_option in cdr_options:
    aim_lc = rioxarray.open_rasterio(path_aim / f'AIM_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)
    gcam_lc = rioxarray.open_rasterio(path_gcam / f'GCAM_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)
    globiom_lc = rioxarray.open_rasterio(path_globiom / f'GLOBIOM_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)
    image_lc = rioxarray.open_rasterio(path_image / f'IMAGE_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)
    magpie_lc = rioxarray.open_rasterio(path_magpie / f'MAgPIE_{cdr_option}_SSP2-26_2100_bin.tif', masked=True)

    if cdr_option == 'Afforestation':
        suit = ref_suit
        not_suit = ref_not_suit
    elif cdr_option == 'BECCS':
        suit = beccs_suit
        not_suit = beccs_not_suit

    aim_lc = aim_lc.rio.reproject_match(res_bio)
    gcam_lc = gcam_lc.rio.reproject_match(res_bio)
    globiom_lc = globiom_lc.rio.reproject_match(res_bio)
    image_lc = image_lc.rio.reproject_match(res_bio)
    magpie_lc = magpie_lc.rio.reproject_match(res_bio)
    suit = suit.rio.reproject_match(res_bio)
    not_suit = not_suit.rio.reproject_match(res_bio)

    agree_in_bio_pos = (aim_lc + gcam_lc + globiom_lc + image_lc + magpie_lc) * res_bio * suit
    agree_in_bio_neg = (aim_lc + gcam_lc + globiom_lc + image_lc + magpie_lc) * res_bio * not_suit

    # at least x-of-five models need to agree
    agree_in_bio_pos.rio.to_raster(path_all / f'mi_{cdr_option}_SSP2-26_2100_suit.tif', driver='GTiff')
    binary_converter(f'mi_{cdr_option}_SSP2-26_2100_suit.tif', path_all, 2,  # adjust model threshold if needed
                     f'mi_{cdr_option}_SSP2-26_2100_suit.tif')
    agree_in_bio_neg.rio.to_raster(path_all / f'mi_{cdr_option}_SSP2-26_2100_not_suit.tif', driver='GTiff')
    binary_converter(f'mi_{cdr_option}_SSP2-26_2100_not_suit.tif', path_all, 2,   # adjust model threshold if needed
                     f'mi_{cdr_option}_SSP2-26_2100_not_suit.tif')

# %%

ar_suit = rs.open(path_all / 'mi_Afforestation_SSP2-26_2100_suit.tif')
be_suit = rs.open(path_all / 'mi_BECCS_SSP2-26_2100_suit.tif')
ar_nsuit = rs.open(path_all / 'mi_Afforestation_SSP2-26_2100_not_suit.tif')
be_nsuit = rs.open(path_all / 'mi_BECCS_SSP2-26_2100_not_suit.tif')
refug = rs.open(path_uea / 'bio1.8_bin.tif')
hs_resil = rs.open(path_hotspots / 'hs_resilient.tif')

data_ar = ar_suit.read(1)
data_be = be_suit.read(1)
data_ar_n = ar_nsuit.read(1)
data_be_n = be_nsuit.read(1)
data_refug = refug.read(1)
data_hs_resil = hs_resil.read(1)

# get the metadata
transform = ar_suit.transform
extent_ar = [transform[2], transform[2] + transform[0] * ar_suit.width,
             transform[5] + transform[4] * ar_suit.height, transform[5]]

transform = be_suit.transform
extent_be = [transform[2], transform[2] + transform[0] * be_suit.width,
             transform[5] + transform[4] * be_suit.height, transform[5]]

transform = ar_nsuit.transform
extent_ar_n = [transform[2], transform[2] + transform[0] * ar_nsuit.width,
               transform[5] + transform[4] * ar_nsuit.height, transform[5]]

transform = be_nsuit.transform
extent_be_n = [transform[2], transform[2] + transform[0] * be_nsuit.width,
               transform[5] + transform[4] * be_nsuit.height, transform[5]]

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

PotBen = ListedColormap([(0, 0, 0, 0), 'orange'])
norm_PotBen = BoundaryNorm([0, 1], PotBen.N)

LikHarm = ListedColormap([(0, 0, 0, 0), 'crimson'])
norm_LikHarm = BoundaryNorm([0, 1], LikHarm.N)

# plot agreement for forestation
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

img_re = ax.imshow(data_refug, extent=extent_refug, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_g1, norm=norm_g1)

img_hs = ax.imshow(data_hs_resil, extent=extent_hs, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_g2, norm=norm_g2)

img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=PotBen, norm=norm_PotBen)

img_ar_n = ax.imshow(data_ar_n, extent=extent_ar_n, transform=ccrs.PlateCarree(),
                      origin='upper', cmap=LikHarm, norm=norm_LikHarm)

ax.coastlines(linewidth=0.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.2)

legend_patches = [
    mpatches.Patch(color='orange', label='Potentially beneficial'),
    mpatches.Patch(color='crimson', label='Likely harmful'),
    mpatches.Patch(color='gainsboro', label='Refugia at 1.8 °C'),
    mpatches.Patch(color='grey', label='Hotspot resilient to 1.8 °C')]

legend = ax.legend(bbox_to_anchor=(-0.01, 0.07), handles=legend_patches, ncols=1,
                                   loc='lower left', fontsize=9.5, columnspacing=0.8,
                                   handletextpad=0.5, borderpad=1.5, frameon=True)

legend.get_frame().set_alpha(1)
legend.get_frame().set_edgecolor('none')

ax.text(-177, -25, 'Forestation', transform=ccrs.PlateCarree(), fontsize=11,
        fontweight='bold', zorder=10)

ax.text(-30, -58, 'SSP2-26 2100\nMinimum cell share: 10%\nModel agreement: 2-of-5',
        transform=ccrs.PlateCarree(), fontsize=10, zorder=10)

plt.show()

# plot agreement for BECCS
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())

img_re = ax.imshow(data_refug, extent=extent_refug, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_g1, norm=norm_g1)

img_hs = ax.imshow(data_hs_resil, extent=extent_hs, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=color_g2, norm=norm_g2)

img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                   origin='upper', cmap=PotBen, norm=norm_PotBen)

img_be_n = ax.imshow(data_be_n, extent=extent_be_n, transform=ccrs.PlateCarree(),
                     origin='upper', cmap=LikHarm, norm=norm_LikHarm)

ax.coastlines(linewidth=0.2)
ax.add_feature(cfeature.BORDERS, linewidth=0.2)

legend_patches = [
    mpatches.Patch(color='orange', label='Potentially beneficial'),
    mpatches.Patch(color='crimson', label='Likely harmful'),
    mpatches.Patch(color='gainsboro', label='Refugia at 1.8 °C'),
    mpatches.Patch(color='grey', label='Hotspot resilient to 1.8 °C')]

legend = ax.legend(bbox_to_anchor=(-0.01, 0.07), handles=legend_patches, ncols=1,
                                   loc='lower left', fontsize=9.5, columnspacing=0.8,
                                   handletextpad=0.5, borderpad=1.5, frameon=True)

legend.get_frame().set_alpha(1)
legend.get_frame().set_edgecolor('none')

ax.text(-177, -25, 'BECCS', transform=ccrs.PlateCarree(), fontsize=11,
        fontweight='bold', zorder=10)

ax.text(-30, -58, 'SSP2-26 2100\nMinimum cell share: 10%\nModel agreement: 2-of-5',
        transform=ccrs.PlateCarree(), fontsize=10, zorder=10)

plt.show()
