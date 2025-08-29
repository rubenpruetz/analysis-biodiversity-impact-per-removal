
# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches
import matplotlib as mpl
import itertools
import rioxarray
import rasterio as rs
import shapefile
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
from pathlib import Path
from required_functions import *

plt.rcParams.update({'figure.dpi': 600})

path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')

path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_gcam = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/gcam_maps')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_magpie = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/magpie_maps')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_hotspots = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ar6_hotspots')
path_ref_pot = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/reforest_potential')
path_beccs_pot = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/Braun_et_al_2024_PB_BECCS/Results/1_source_data_figures/Fig2')
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')
cfs_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/cfs')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')

# define lists
years = ['2010', '2020', '2030', '2040', '2050', '2060', '2070', '2080',
         '2090', '2100']
models = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'IMAGE 3.0.1', 'GCAM 4.2',
          'REMIND-MAgPIE 1.5']
scenarios = ['SSP1-19', 'SSP1-26', 'SSP1-45', 'SSP2-19', 'SSP2-26', 'SSP2-45',
             'SSP3-45']
variables = ['Land Cover|Built-up Area', 'Land Cover|Cropland',
             'Land Cover|Cropland|Energy Crops', 'Land Cover|Forest',
             'Land Cover|Pasture']
variables_adjust = ['Land Cover|Built-up Area', 'Other cropland',
                    'Land Cover|Forest', 'Energy cropland (for BECCS)',
                    'Energy cropland (not for BECCS)', 'Land Cover|Pasture']

tcre_df = pd.read_csv(path_ar6_data / 'tcre_estimates.csv')
p50_est = float(tcre_df[(tcre_df['Source'] == 'Own trans') &
                        (tcre_df['Estimate'] == 'point')]['Value'].iloc[0])

# %% plot supplementary figure on BECCS in Annex-I and Non-Annex I refugia

paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
area_recov = load_and_concat('area_df_annex_group_temp_decline_allowed', paths)
area_recov['BioRecov'] = 'Allowed'
area_norecov = load_and_concat('area_df_annex_group_temp_decline_not_allowed', paths)
area_norecov['BioRecov'] = 'Not allowed'

area_df = pd.concat([area_recov, area_norecov]).reset_index(drop=True)

rcps = ['19', '26', '45']  # specify RCPs that shall be plotted
area_df = area_df.loc[area_df['RCP'].isin(rcps)]

rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
           '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}

fig, axes = plt.subplots(3, 2, figsize=(4.2, 7), sharex=True, sharey=False)
sns.lineplot(data=area_df.query('Model == "AIM" & mitigation_option == "BECCS"'),
             x='Year', y='alloc_perc_ag1', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 0])
sns.lineplot(data=area_df.query('Model == "GLOBIOM" & mitigation_option == "BECCS"'),
             x='Year', y='alloc_perc_ag1', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=True, ax=axes[1, 0])
sns.lineplot(data=area_df.query('Model == "IMAGE" & mitigation_option == "BECCS"'),
             x='Year', y='alloc_perc_ag1', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 0])

sns.lineplot(data=area_df.query('Model == "AIM" & mitigation_option == "BECCS"'),
             x='Year', y='alloc_perc_ag2', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 1])
sns.lineplot(data=area_df.query('Model == "GLOBIOM" & mitigation_option == "BECCS"'),
             x='Year', y='alloc_perc_ag2', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 1])
sns.lineplot(data=area_df.query('Model == "IMAGE" & mitigation_option == "BECCS"'),
             x='Year', y='alloc_perc_ag2', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 1])

handles, labels = axes[1, 0].get_legend_handles_labels()
rename_dict = {'19': '1.5 °C', '26': '2 °C', '45': 'Current Policies'}

new_labels = [rename_dict.get(label, label) for label in labels]

axes[1, 0].legend(handles, new_labels, bbox_to_anchor=(-0.02, 2.78), loc='upper left',
                  ncols=3, columnspacing=1, handletextpad=0.4, handlelength=0.9,
                  fontsize=11)

axes[0, 0].set_title('Annex I\n(BECCS)')
axes[0, 1].set_title('Non-Annex I\n(BECCS)')

axes[2, 0].set_xlabel('')
axes[2, 1].set_xlabel('')

axes[0, 0].set_ylabel('AIM', fontsize=12)
axes[1, 0].set_ylabel('GLOBIOM', fontsize=12)
axes[2, 0].set_ylabel('IMAGE', fontsize=12)

axes[0, 1].set_ylabel('')
axes[1, 1].set_ylabel('')
axes[2, 1].set_ylabel('')

for ax in axes[:, 0]:
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim([0, 5.5])
for ax in axes[:, 1]:
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.set_ylim([0, 5.5])
    ax.set_yticklabels([])

fig.supylabel(f'Share of remaining refugia allocated for CDR [%]',
              x=-0.01, va='center', ha='center', fontsize=13)

for ax in axes.flat:
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2100])
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)
    ax.tick_params(axis='x', labelsize=11.7)
    ax.tick_params(axis='y', labelsize=11.2)

plt.subplots_adjust(hspace=0.25)
plt.subplots_adjust(wspace=0.37)
sns.despine()
plt.show()

# %% plot supplementary figure on avoided warming-related refugia loss due to CDR
models_3 = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'IMAGE 3.0.1']
scenarios = ['SSP1-Baseline', 'SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45',
             'SSP2-Baseline', 'SSP2-19', 'SSP2-26', 'SSP2-34', 'SSP2-45',
             'SSP2-60', 'SSP3-Baseline', 'SSP3-34', 'SSP3-45', 'SSP3-60']
variable = ['AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile']

ar6_data = ar6_db.loc[ar6_db['Variable'].isin(variable)]
ar6_data = ar6_data.loc[ar6_data['Model'].isin(models_3)]
ar6_data = ar6_data.loc[ar6_data['Scenario'].isin(scenarios)]

# rename models for the subsequent step
ar6_data.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                            'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                            'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
ar_removal = load_and_concat('ar_removal', paths)
beccs_removal = load_and_concat('beccs_removal', paths)

# calculate cumulative removals and avoided warming based on TCRE
ar_cum = cum_cdr_calc(ar_removal)
ar_cum['CoolAR'] = ar_cum['Cum'] * p50_est * 1000  # x1000 for Gt to Mt

beccs_cum = cum_cdr_calc(beccs_removal)
beccs_cum['CoolBECCS'] = beccs_cum['Cum'] * p50_est * 1000

# estimate additional warming if AR and BECCS are excluded
ar6_data_m = pd.melt(ar6_data, id_vars=['Model', 'Scenario'], value_vars=years,
                     var_name='Year', value_name='Warming')
ar6_data_m['Year'] = ar6_data_m['Year'].astype(int)

add_warm = pd.merge(ar6_data_m, ar_cum[['Model', 'Scenario', 'Year', 'CoolAR']],
                    on=['Model', 'Scenario', 'Year'], how='left')
add_warm = pd.merge(add_warm, beccs_cum[['Model', 'Scenario', 'Year', 'CoolBECCS']],
                    on=['Model', 'Scenario', 'Year'], how='left')
add_warm.fillna(0, inplace=True)  # account for scenarios with only one CDR option
add_warm['WarmNoAR'] = add_warm['Warming'] + add_warm['CoolAR']
add_warm['WarmNoBECCS'] = add_warm['Warming'] + add_warm['CoolBECCS']
add_warm['WarmNoCDR'] = add_warm['Warming'] + add_warm['CoolAR'] + add_warm['CoolBECCS']
round_cols = ['Warming', 'WarmNoAR', 'WarmNoBECCS', 'WarmNoCDR']
add_warm[round_cols] = add_warm[round_cols].round(2)

# estimate warming curve for warming and WarmNoCDR if there were no decline
add_warm = add_warm.sort_values(['Model', 'Scenario', 'Year']).copy()
grouped = add_warm.groupby(['Model', 'Scenario'])
add_warm['Warming_stab'] = grouped['Warming'].cummax()
add_warm['WarmNoCDR_stab'] = grouped['WarmNoCDR'].cummax()

# create lookup table for global climate refugia size per warming level
warm_list = [round(x, 2) for x in np.arange(1.0, 4.51, 0.01)]
refugia_size = []

for warm in warm_list:
    bio_area = land_area_calculation(path_uea, f'bio{warm}_bin.tif')
    bio_area_agg = pos_val_summer(bio_area, squeeze=True)
    refugia_size.append((warm, bio_area_agg))

refug_df = pd.DataFrame(refugia_size, columns=['Warming', 'RemRef'])

# estimate avoided refugia loss due to CDR for both recovery assumptions
avlo_recov = add_warm.copy()
avlo_recov = pd. merge(avlo_recov, refug_df, left_on='Warming', right_on='Warming')
avlo_recov = pd. merge(avlo_recov, refug_df, left_on='WarmNoCDR',
                       right_on='Warming',  suffixes=('', 'NoCDR'))
avlo_recov['AvLoNoCDR'] = (1 - (avlo_recov['RemRefNoCDR'] / avlo_recov['RemRef'])) * 100

avlo_norecov = add_warm.copy()
avlo_norecov = pd. merge(avlo_norecov, refug_df, left_on='Warming_stab', right_on='Warming')
avlo_norecov = pd. merge(avlo_norecov, refug_df, left_on='WarmNoCDR_stab',
                         right_on='Warming', suffixes=('', 'NoCDR'))
avlo_norecov['AvLoNoCDR'] = (1 - (avlo_norecov['RemRefNoCDR'] / avlo_norecov['RemRef'])) * 100

avlo_recov.drop(columns=['WarmingNoCDR', 'Warming'], inplace=True)
avlo_norecov.drop(columns=['Warming_x', 'Warming_y', 'Warming'], inplace=True)

avlo_recov['BioRecov'] = 'Allowed'
avlo_norecov['BioRecov'] = 'Not allowed'

avlo_df = pd.concat([avlo_recov, avlo_norecov]).reset_index(drop=True)

# plot avoided warming loss of remaining refugia due to CDR
avlo_df['SSP'] = avlo_df['Scenario'].str.split('-').str[0]
avlo_df['RCP'] = avlo_df['Scenario'].str.split('-').str[1]

# drop AIM RCP1.9 as forestation removal variable is missing
avlo_df.drop(avlo_df[(avlo_df['Model'] == 'AIM') &
                     (avlo_df['RCP'] == '19')].index, inplace=True)

avlo_df = avlo_df.loc[avlo_df['RCP'].isin(rcps)]  # specify RCPs to plot

# plot avoided loss as negative
avlo_df['AvLoNoCDR_invers'] = avlo_df['AvLoNoCDR'] * -1

fig, axes = plt.subplots(3, 1, figsize=(1.8, 7), sharex=True, sharey=True)
sns.lineplot(data=avlo_df.query('Model == "AIM"'), x='Year', y='AvLoNoCDR_invers',
             hue='RCP', palette=rcp_pal, errorbar=('pi', 100),
             estimator='median', legend=False, ax=axes[0])
sns.lineplot(data=avlo_df.query('Model == "GLOBIOM"'), x='Year', y='AvLoNoCDR_invers',
             hue='RCP', palette=rcp_pal, errorbar=('pi', 100),
             estimator='median', legend=False, ax=axes[1])
sns.lineplot(data=avlo_df.query('Model == "IMAGE"'), x='Year', y='AvLoNoCDR_invers',
             hue='RCP', palette=rcp_pal, errorbar=('pi', 100),
             estimator='median', legend=False, ax=axes[2])

axes[0].set_title('Global\n(Both)', fontsize=13.5)
axes[2].set_xlabel('')
axes[0].set_ylabel('AIM', fontsize=12)
axes[1].set_ylabel('GLOBIOM', fontsize=12)
axes[2].set_ylabel('IMAGE', fontsize=12)
fig.supylabel(f'Share of remaining refugia lost when excluding CDR [%]',
              x=-0.35, va='center', ha='center', fontsize=13)

for ax in axes.flat:
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2100])
    ax.set_ylim(-25, 0)
    ax.tick_params(axis='x', labelsize=11.7)
    ax.tick_params(axis='y', labelsize=11.2)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

plt.subplots_adjust(hspace=0.23)
sns.despine()
plt.show()

# %% plot supplementary figure on CDR in refugia for combined removal levels (AR+BECCS)
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

rcp_lvl = '26'  # select RCP level (without dot)
ssps = ['SSP1', 'SSP2', 'SSP3']
models_3 = ['AIM', 'GLOBIOM', 'IMAGE']
removal_lvls = [6, 10]

for model in models_3:
    if model == 'GLOBIOM':
        path = path_globiom
    elif model == 'AIM':
        path = path_aim
    elif model == 'IMAGE':
        path = path_image

    for removal_lvl in removal_lvls:
        for ssp in ssps:

            try:
                ar = f'{model}_Afforestation_{ssp}-{rcp_lvl}_sum{removal_lvl}GtCO2.tif'
                be = f'{model}_BECCS_{ssp}-{rcp_lvl}_sum{removal_lvl}GtCO2.tif'

                ar = rioxarray.open_rasterio(path / ar, masked=True)
                be = rioxarray.open_rasterio(path / be, masked=True)

                refugia = rioxarray.open_rasterio(path_uea / 'bio1.8_bin.tif', masked=True)  # specify warming level
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
                ar_in_bio.rio.to_raster(path / f'{ssp}_ar_sum{removal_lvl}_bio_absolute.tif', driver='GTiff')
                be_in_bio.rio.to_raster(path / f'{ssp}_be_sum{removal_lvl}_bio_absolute.tif', driver='GTiff')

            except Exception as e:
                print(f'Error processing {ssp}: {e}')
                continue

            # calculate country burden for refugia
            dfs = []

            try:
                intersect_src = rs.open(path / f'{ssp}_ar_sum{removal_lvl}_bio_absolute.tif')
                df_ar = admin_bound_calculator(ssp, admin_sf, intersect_src)
                df_ar['option'] = 'AR'
                dfs.append(df_ar)
            except Exception as e:
                print(f'Error processing AR {ssp}: {e}')
                continue

            try:
                intersect_src = rs.open(path / f'{ssp}_be_sum{removal_lvl}_bio_absolute.tif')
                df_be = admin_bound_calculator(ssp, admin_sf, intersect_src)
                df_be['option'] = 'BECCS'
                dfs.append(df_be)
            except Exception as e:
                print(f'Error processing BECCS {ssp}: {e}')
                continue

            df_options = pd.concat(dfs, axis=0)

            land_area_calculation(path_uea, 'bio1.8_bin.tif', 'bio1.8_bin_km2.tif')
            intersect_src = rs.open(path_uea / 'bio1.8_bin_km2.tif')
            df_bio15 = admin_bound_calculator('all_ssps', admin_sf, intersect_src)
            df_bio15 = df_bio15.rename(columns={'km2': 'bio_km2'})

            wab_out = pd.merge(df_options, df_bio15[['iso3', 'bio_km2']], on='iso3', how='inner')
            wab_out['affected_bio_share'] = wab_out['km2'] / wab_out['bio_km2'] * 100

            wab_cum = wab_out.groupby(['iso3', 'bio_km2'], as_index=False)['km2'].sum()
            wab_cum['cum_affected_bio_share'] = wab_cum['km2'] / wab_cum['bio_km2'] * 100
            wab_cum['cum_affected_bio_share'] = wab_cum['cum_affected_bio_share'].round(1)
            wab_cum.dropna(inplace=True)

            # plot national allocation levels in resilient refugia
            bounds = [0, 1, 5, 10, 15, 20, 25]
            norm = mpl.colors.BoundaryNorm(bounds, mpl.cm.PuRd.N, extend='max')
            cmap = mpl.cm.PuRd

            fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson()})
            shape_records = list(Reader(sf_path / 'world-administrative-boundaries.shp').records())

            for record in shape_records:
                iso = record.attributes['iso3']
                if iso in wab_cum['iso3'].values:
                    val = wab_cum.loc[wab_cum['iso3'] == iso, 'cum_affected_bio_share'].values[0]
                    color = cmap(norm(val))
                    ax.add_geometries([record.geometry], ccrs.PlateCarree(), facecolor=color,
                                      edgecolor='black', linewidth=0.2)

            ax.coastlines(linewidth=0.2)
            ax.set_extent([-167, 167, -58, 90])

            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                                ax=ax, orientation='horizontal',
                                boundaries=bounds, ticks=bounds,
                                spacing='proportional', extend='max')
            cbar.ax.set_position([0.346, -0.175, 0.334, 0.5])
            cbar.ax.tick_params(labelsize=14)

            cdr_sum = int(removal_lvl)
            cbar.set_label(
                f'Share of national refugia covered by Forestation \nand BECCS for removals of {cdr_sum} GtCO$_2$ [%]',
                fontsize=15)
            plt.title(f'{model} {ssp}-{rcp_lvl}', fontsize=12.5, x=0.04, y=0.2, ha='left')
            plt.show()

# %% plot supplementary figure on removal per CDR option, scenario and model
paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
ar_removal = load_and_concat('ar_removal', paths)
ar_removal['SSP'] = ar_removal['Scenario'].str.split('-').str[0]
ar_removal['RCP'] = ar_removal['Scenario'].str.split('-').str[1]
ar_removal = ar_removal.loc[ar_removal['Scenario'].isin(scenarios)]

beccs_removal = load_and_concat('beccs_removal', paths)
beccs_removal['SSP'] = beccs_removal['Scenario'].str.split('-').str[0]
beccs_removal['RCP'] = beccs_removal['Scenario'].str.split('-').str[1]
beccs_removal = beccs_removal.loc[beccs_removal['Scenario'].isin(scenarios)]

rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
           '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}

# plot removal via forestation
fig, axes = plt.subplots(3, 3, figsize=(6, 9), sharex=True, sharey=True)
sns.lineplot(data=ar_removal.query('Model == "AIM" & SSP == "SSP1"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 0])
sns.lineplot(data=ar_removal.query('Model == "GLOBIOM" & SSP == "SSP1"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=True, ax=axes[1, 0])
sns.lineplot(data=ar_removal.query('Model == "IMAGE" & SSP == "SSP1"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 0])

sns.lineplot(data=ar_removal.query('Model == "AIM" & SSP == "SSP2"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 1])
sns.lineplot(data=ar_removal.query('Model == "GLOBIOM" & SSP == "SSP2"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 1])
sns.lineplot(data=ar_removal.query('Model == "IMAGE" & SSP == "SSP2"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 1])

sns.lineplot(data=ar_removal.query('Model == "AIM" & SSP == "SSP3"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 2])
sns.lineplot(data=ar_removal.query('Model == "GLOBIOM" & SSP == "SSP3"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 2])
sns.lineplot(data=ar_removal.query('Model == "IMAGE" & SSP == "SSP3"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 2])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2010, 2060, 2100])
    ax.set_ylim(0, 5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[2, 0].set_ylabel('IMAGE', fontsize=11)

axes[0, 0].set_title('SSP1')
axes[0, 1].set_title('SSP2')
axes[0, 2].set_title('SSP3')

handles, labels = axes[1, 0].get_legend_handles_labels()
rename_dict = {'19': '1.5 °C', '26': '2 °C', '45': 'Current Policies'}
new_labels = [rename_dict.get(label, label) for label in labels]
axes[1, 0].legend(handles, new_labels, bbox_to_anchor=(-0.05, 2.4), ncols=3,
                  loc='upper left', columnspacing=1, handletextpad=0.4)

fig.supylabel(f'CO$_2$ removal via forestation [GtCO$_2$]',
              x=0.04, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.4)
sns.despine()
plt.show()

# plot removal via BECCS
fig, axes = plt.subplots(3, 3, figsize=(6, 9), sharex=True, sharey=True)
sns.lineplot(data=beccs_removal.query('Model == "AIM" & SSP == "SSP1"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 0])
sns.lineplot(data=beccs_removal.query('Model == "GLOBIOM" & SSP == "SSP1"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=True, ax=axes[1, 0])
sns.lineplot(data=beccs_removal.query('Model == "IMAGE" & SSP == "SSP1"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 0])

sns.lineplot(data=beccs_removal.query('Model == "AIM" & SSP == "SSP2"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 1])
sns.lineplot(data=beccs_removal.query('Model == "GLOBIOM" & SSP == "SSP2"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 1])
sns.lineplot(data=beccs_removal.query('Model == "IMAGE" & SSP == "SSP2"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 1])

sns.lineplot(data=beccs_removal.query('Model == "AIM" & SSP == "SSP3"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 2])
sns.lineplot(data=beccs_removal.query('Model == "GLOBIOM" & SSP == "SSP3"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 2])
sns.lineplot(data=beccs_removal.query('Model == "IMAGE" & SSP == "SSP3"'),
             x='Year', y='Removal', hue='RCP', palette=rcp_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 2])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2010, 2060, 2100])
    ax.set_ylim(0, 12)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[2, 0].set_ylabel('IMAGE', fontsize=11)

axes[0, 0].set_title('SSP1')
axes[0, 1].set_title('SSP2')
axes[0, 2].set_title('SSP3')

handles, labels = axes[1, 0].get_legend_handles_labels()
rename_dict = {'19': '1.5 °C', '26': '2 °C', '45': 'Current Policies'}
new_labels = [rename_dict.get(label, label) for label in labels]
axes[1, 0].legend(handles, new_labels, bbox_to_anchor=(-0.05, 2.4), ncols=3,
                  loc='upper left', columnspacing=1, handletextpad=0.4)

fig.supylabel(f'CO$_2$ removal via crop-based BECCS [GtCO$_2$]',
              x=0.022, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.4)
sns.despine()
plt.show()

# %% plot supplementary figure on land per CDR option, scenario and model
model_fam = ['AIM', 'GCAM', 'GLOBIOM', 'IMAGE', 'MAgPIE']
cdr_options = ['Afforestation', 'BECCS']

area_dfs = []

for model in model_fam:
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

    for cdr_option in cdr_options:
        for scenario in scenarios:
            for year in years:
                try:
                    file = f'{model}_{cdr_option}_{scenario}_{year}.tif'
                    land = rioxarray.open_rasterio(path / file, masked=True)
                    land = pos_val_summer(land, squeeze=True)
                    land = land / 1000000  # km2 to Mkm2

                    area_dfs.append({'Model': model, 'Cdr_option': cdr_option,
                        'Scenario': scenario, 'Year': year, 'Land': land})

                except Exception as e:
                    print(f'Failed to process {file}: {e}')

area_df = pd.DataFrame(area_dfs)
area_df['Year'] = area_df['Year'].astype(int)
area_df['SSP'] = area_df['Scenario'].str.split('-').str[0]
area_df['RCP'] = area_df['Scenario'].str.split('-').str[1]

ar_land = area_df.query('Cdr_option == "Afforestation"').reset_index(drop=True)
beccs_land = area_df.query('Cdr_option == "BECCS"').reset_index(drop=True)

ar_land = ar_land.loc[ar_land['RCP'].isin(rcps)]
beccs_land = beccs_land.loc[beccs_land['RCP'].isin(rcps)]

rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
           '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}

# plot removal via forestation
fig, axes = plt.subplots(5, 3, figsize=(8, 9), sharex=True, sharey=True)
sns.lineplot(data=ar_land.query('Model == "AIM" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[0, 0])
sns.lineplot(data=ar_land.query('Model == "GCAM" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[1, 0])
sns.lineplot(data=ar_land.query('Model == "GLOBIOM" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[2, 0])
sns.lineplot(data=ar_land.query('Model == "IMAGE" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[3, 0])
sns.lineplot(data=ar_land.query('Model == "MAgPIE" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[4, 0])

sns.lineplot(data=ar_land.query('Model == "AIM" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=True, ax=axes[0, 1])
sns.lineplot(data=ar_land.query('Model == "GCAM" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal,legend=False, ax=axes[1, 1])
sns.lineplot(data=ar_land.query('Model == "GLOBIOM" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[2, 1])
sns.lineplot(data=ar_land.query('Model == "IMAGE" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[3, 1])
sns.lineplot(data=ar_land.query('Model == "MAgPIE" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[4, 1])

sns.lineplot(data=ar_land.query('Model == "AIM" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[0, 2])
sns.lineplot(data=ar_land.query('Model == "GCAM" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[1, 2])
sns.lineplot(data=ar_land.query('Model == "GLOBIOM" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[2, 2])
sns.lineplot(data=ar_land.query('Model == "IMAGE" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[3, 2])
sns.lineplot(data=ar_land.query('Model == "MAgPIE" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[4, 2])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2010, 2100)
    ax.set_xticks([2010, 2055, 2100])
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GCAM', fontsize=11)
axes[2, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[3, 0].set_ylabel('IMAGE', fontsize=11)
axes[4, 0].set_ylabel('REMIND-MAgPIE', fontsize=11)

axes[0, 0].set_title('SSP1')
axes[0, 1].set_title('SSP2')
axes[0, 2].set_title('SSP3')

handles, labels = axes[0, 1].get_legend_handles_labels()
rename_dict = {'19': '1.5 °C', '26': '2 °C', '45': 'Current Policies'}
new_labels = [rename_dict.get(label, label) for label in labels]
axes[0, 1].legend(handles, new_labels, bbox_to_anchor=(-01.3, 1.55), ncols=3,
                  loc='upper left', columnspacing=1, handletextpad=0.4)

fig.supylabel(f'Land area for forestation [Mkm$^2$]', x=0.03, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.3)
sns.despine()
plt.show()

# plot removal via BECCS
fig, axes = plt.subplots(5, 3, figsize=(8, 9), sharex=True, sharey=True)
sns.lineplot(data=beccs_land.query('Model == "AIM" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[0, 0])
sns.lineplot(data=beccs_land.query('Model == "GCAM" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[1, 0])
sns.lineplot(data=beccs_land.query('Model == "GLOBIOM" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[2, 0])
sns.lineplot(data=beccs_land.query('Model == "IMAGE" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[3, 0])
sns.lineplot(data=beccs_land.query('Model == "MAgPIE" & SSP == "SSP1"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[4, 0])

sns.lineplot(data=beccs_land.query('Model == "AIM" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=True, ax=axes[0, 1])
sns.lineplot(data=beccs_land.query('Model == "GCAM" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[1, 1])
sns.lineplot(data=beccs_land.query('Model == "GLOBIOM" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[2, 1])
sns.lineplot(data=beccs_land.query('Model == "IMAGE" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[3, 1])
sns.lineplot(data=beccs_land.query('Model == "MAgPIE" & SSP == "SSP2"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[4, 1])

sns.lineplot(data=beccs_land.query('Model == "AIM" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[0, 2])
sns.lineplot(data=beccs_land.query('Model == "GCAM" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[1, 2])
sns.lineplot(data=beccs_land.query('Model == "GLOBIOM" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[2, 2])
sns.lineplot(data=beccs_land.query('Model == "IMAGE" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[3, 2])
sns.lineplot(data=beccs_land.query('Model == "MAgPIE" & SSP == "SSP3"'),
             x='Year', y='Land', hue='RCP', palette=rcp_pal, legend=False, ax=axes[4, 2])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2010, 2100)
    ax.set_xticks([2010, 2055, 2100])
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GCAM', fontsize=11)
axes[2, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[3, 0].set_ylabel('IMAGE', fontsize=11)
axes[4, 0].set_ylabel('REMIND-MAgPIE', fontsize=11)

axes[0, 0].set_title('SSP1')
axes[0, 1].set_title('SSP2')
axes[0, 2].set_title('SSP3')

handles, labels = axes[0, 1].get_legend_handles_labels()
rename_dict = {'19': '1.5 °C', '26': '2 °C', '45': 'Current Policies'}
new_labels = [rename_dict.get(label, label) for label in labels]
axes[0, 1].legend(handles, new_labels, bbox_to_anchor=(-01.3, 1.55), ncols=3,
                  loc='upper left', columnspacing=1, handletextpad=0.4)

fig.supylabel(f'Land area for BECCS [Mkm$^2$]', x=0.03, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.3)
sns.despine()
plt.show()

# %% get AR6 land cover data for SSP-RCP combinations
ar6_db = ar6_db.loc[ar6_db['Model'].isin(models)]
ar6_db = ar6_db.loc[ar6_db['Scenario'].isin(scenarios)]
lc_data = ar6_db.loc[ar6_db['Variable'].isin(variables)]
lc_data = lc_data[['Model', 'Scenario', 'Variable'] + years].copy()

lc_data = pd.melt(lc_data, id_vars=['Model', 'Scenario', 'Variable'],
                  value_vars=years, var_name='Year', value_name='Value')
lc_ar6 = lc_data.copy()

# distinguish between energy cropland and other cropland
cropland = lc_data.loc[lc_data['Variable'].isin(['Land Cover|Cropland'])]
cropland_energy = lc_data.loc[lc_data['Variable'].isin(['Land Cover|Cropland|Energy Crops'])]

cropland_other = pd.merge(cropland, cropland_energy,
                          on=['Model', 'Scenario', 'Year'],
                          suffixes=['_all', '_energy'])

cropland_other['Value'] = cropland_other['Value_all'] - cropland_other['Value_energy']
cropland_other['Variable'] = 'Other cropland'
cropland_other = cropland_other[['Model', 'Scenario', 'Variable', 'Year',
                                 'Value']].copy()
lc_data = pd.concat([lc_data, cropland_other], axis=0)

# distinguish between energy cropland with and without CCS
bioeng_ncss = ar6_db.query('Variable == "Primary Energy|Biomass|Modern|w/o CCS"').reset_index(drop=True)
bioeng_ncss[years] = bioeng_ncss[years].round(2)
bioeng_tot = ar6_db.query('Variable == "Primary Energy|Biomass"').reset_index(drop=True)
bioeng_tot[years] = bioeng_tot[years].round(2)
bioeng_wccs = bioeng_tot[['Model', 'Scenario']].copy()
bioeng_wccs[years] = 1 - (bioeng_ncss[years] / bioeng_tot[years])
bioeng_wccs = pd.melt(bioeng_wccs, id_vars=['Model', 'Scenario'],
                      value_vars=years, var_name='Year',
                      value_name='CCS_share')

cropland_eng_css = cropland_energy.copy()
cropland_eng_css = pd.merge(cropland_eng_css, bioeng_wccs,
                            on=['Model', 'Scenario', 'Year'])
cropland_eng_css['Value'] = cropland_eng_css['Value'] * bioeng_wccs['CCS_share']
cropland_eng_css['Variable'] = 'Energy cropland (for BECCS)'

cropland_eng_nocss = cropland_energy.copy()
cropland_eng_nocss = pd.merge(cropland_eng_nocss, bioeng_wccs,
                              on=['Model', 'Scenario', 'Year'])
cropland_eng_nocss['Value'] = cropland_eng_nocss['Value'] * (1 - bioeng_wccs['CCS_share'])
cropland_eng_nocss['Variable'] = 'Energy cropland (not for BECCS)'

lc_data = pd.concat([lc_data, cropland_eng_css, cropland_eng_nocss], axis=0)

# calculate change from base year (2010)
lc_data = lc_data.loc[lc_data['Variable'].isin(variables_adjust)]
lc_2010 = lc_data.query('Year == "2010"').reset_index(drop=True)
lc_data = pd.merge(lc_data, lc_2010, on=['Model', 'Scenario', 'Variable'],
                   suffixes=['', '_2010'])

lc_data['Change'] = lc_data['Value'] - lc_data['Value_2010']
lc_data['Change'] = lc_data['Change'] / 100  # from Mha to Mkm2

# plot supplementary figure on land use changes based on AR6 Scenarios Database
lc_data['SSP'] = lc_data['Scenario'].str.split('-').str[0]
lc_data['RCP'] = lc_data['Scenario'].str.split('-').str[1]
lc_data['Year'] = lc_data['Year'].astype(int)

lc_data.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                           'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                           'IMAGE 3.0.1': 'IMAGE',
                           'GCAM 4.2': 'GCAM',
                           'REMIND-MAgPIE 1.5': 'MAgPIE'}}, inplace=True)

lc_data.replace({'Variable': {'Land Cover|Built-up Area': 'Built-up',
                              'Land Cover|Forest': 'Forest',
                              'Land Cover|Pasture': 'Pasture'}}, inplace=True)

var_pal = {'Built-up': 'dimgrey',
           'Energy cropland (for BECCS)': 'orangered',
           'Energy cropland (not for BECCS)': 'blue',
           'Forest': 'forestgreen',
           'Other cropland': 'brown',
           'Pasture': 'gold'}
all_vars = sorted(lc_data['Variable'].unique())

fig, axes = plt.subplots(5, 3, figsize=(8, 9), sharex=True, sharey=True)
sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=True, ax=axes[0, 0])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 0])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 0])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 0])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 0])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 1])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 1])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 1])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 1])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 1])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 2])
sns.lineplot(data=lc_data.query('Model == "GCAM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 2])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 2])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[3, 2])
sns.lineplot(data=lc_data.query('Model == "MAgPIE" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[4, 2])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2010, 2100)
    ax.set_xticks([2010, 2055, 2100])
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GCAM', fontsize=11)
axes[2, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[3, 0].set_ylabel('IMAGE', fontsize=11)
axes[4, 0].set_ylabel('REMIND-MAgPIE', fontsize=11)

axes[0, 0].set_title('1.5 °C')
axes[0, 1].set_title('2 °C')
axes[0, 2].set_title('Current Policies')

axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.75), loc='upper left', ncols=3,
                  columnspacing=1, handletextpad=0.4, fontsize=11)

fig.supylabel(f'Land cover change from 2010 [Mkm$^2$] (SSP1-SSP3 range as shading)',
              x=0.03, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.3)
sns.despine()
plt.show()

# %% calculate potentially disappeared fraction of species (PDF) based on CFs in Scherer et al.
cf_var = 'CF_occ_avg_glo'  # choose 'CF_occ_avg_glo' or 'CF_occ_mar_glo'
cf_df = pd.read_csv(cfs_path / 'CF_global.csv')
cf_plant = cf_df.query('kingdom == "Plantae" & weighting == "land_use"').reset_index(drop=True)
cf_plant = cf_plant[['kingdom', 'habitat', cf_var]].copy()

cf_amal = cf_df.query('kingdom == "Animalia" & weighting == "land_use"'). reset_index(drop=True)

# average the CF across species groups in the animalia kingdom
cf_amal = cf_amal.groupby(['kingdom', 'habitat'], as_index=False)[cf_var].mean()

# combine cfs for animalia and plantae
cf_combi = pd.concat([cf_plant, cf_amal], ignore_index=True)

# rename habitat intensisies (conservative approach) to have uniform names
cf_combi['habitat'] = cf_combi['habitat'].str.replace('MinimalLight', 'Minimal')
cf_combi['habitat'] = cf_combi['habitat'].str.replace('LightIntense', 'Light')

# average the CFs across animalia and plantae kingdom
cf_combi = cf_combi.groupby(['habitat'], as_index=False)[cf_var].mean()

# create global scenario lc in m2 based on ar6 data
lc_m2 = lc_ar6.copy()
lc_m2['Value'] = lc_m2['Value'] * 10000000000  # Mha to m2
lc_m2.replace({'Variable': {'Land Cover|Cropland': 'Cropland_Light',
                            'Land Cover|Forest': 'Managed_forest_Light',
                            'Land Cover|Pasture': 'Pasture_Light',
                            'Land Cover|Cropland|Energy Crops': 'Plantation_Light',
                            'Land Cover|Built-up Area': 'Urban_Light'}}, inplace=True)

cf_df = pd.merge(lc_m2, cf_combi, left_on='Variable', right_on='habitat',
                 how='inner')
cf_df['PDF·Year'] = cf_df['Value'] * cf_df[cf_var]
cf_df = cf_df.groupby(['Model', 'Scenario', 'Year'], as_index=False)['PDF·Year'].sum()
cf_df['PDF·Year'] = cf_df['PDF·Year'].round(3)

cd_yrs = ['2020', '2050']
cf_df = cf_df.loc[cf_df['Year'].isin(cd_yrs)]

cf_df.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                         'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                         'IMAGE 3.0.1': 'IMAGE',
                         'GCAM 4.2': 'GCAM',
                         'REMIND-MAgPIE 1.5': 'MAgPIE'}}, inplace=True)

pdf_table = cf_df.pivot(index=['Scenario', 'Year'], columns='Model',
                        values='PDF·Year').reset_index()

# %% explore reduction in CDR land for various sensitivities
# load area-based criteria for beneficia/harmful effects on biodiversity
ref_not_suit = rioxarray.open_rasterio(path_ref_pot / 'ref_not_suit.tif', masked=True)
beccs_not_suit = rioxarray.open_rasterio(path_beccs_pot / 'beccs_not_suit.tif', masked=True)

warmings = [1.8]  # adjust warming level if needed
scenarios = ['SSP1-26', 'SSP2-26']
exclusions = ['all_bio_areas', 'unsuit_bio_areas']

for scenario in scenarios:
    for warm in warmings:
        for exclu in exclusions:
            hotspots = rioxarray.open_rasterio(path_hotspots / 'ar6_hotspots_10arcmin.tif', masked=True)
            res_bio = rioxarray.open_rasterio(path_uea / f'bio{warm}_bin.tif', masked=True)

            # estimate hotspot areas that a resilient to selected warming
            hotspot_repro = hotspots.rio.reproject_match(res_bio)
            hs_resil = hotspot_repro * res_bio

            # estimate reduction in land allocation when excluding areas
            years = [2030, 2050, 2100]

            exclu_df = pd.DataFrame(columns=['Model', 'CDR_option', 'Year', 'CDR_land',
                                             'CDR_in_hs', 'CDR_in_hs_res', 'CDR_in_bio'])
            for model in model_fam:
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

                        if cdr_option == 'Afforestation':
                            not_suit = ref_not_suit
                        elif cdr_option == 'BECCS':
                            not_suit = beccs_not_suit

                        try:
                            cdr_land = f'{model}_{cdr_option}_{scenario}_{year}.tif'  # change scenario if required

                            cdr = rioxarray.open_rasterio(path / cdr_land, masked=True)
                            tot_cdr_area = pos_val_summer(cdr, squeeze=True)

                            if exclu == 'all_bio_areas':
                                # CDR in biodiversity hotspots
                                cdr_repro = cdr.rio.reproject_match(hotspots)
                                cdr_in_hs = cdr_repro * hotspots

                                # CDR in biodiversity hotspots resilient to warming
                                cdr_repro = cdr.rio.reproject_match(hs_resil)
                                cdr_in_hs_res = cdr_repro * hs_resil

                                # CDR in warming resilient refugia
                                cdr_repro = cdr.rio.reproject_match(res_bio)
                                cdr_in_bio = cdr_repro * res_bio

                            elif exclu == 'unsuit_bio_areas':
                                # CDR in biodiversity hotspots
                                cdr_repro = cdr.rio.reproject_match(hotspots)
                                nsuit_repro = not_suit.rio.reproject_match(hotspots)
                                cdr_in_hs = cdr_repro * hotspots * nsuit_repro

                                # CDR in biodiversity hotspots resilient to warming
                                cdr_repro = cdr.rio.reproject_match(hs_resil)
                                nsuit_repro = not_suit.rio.reproject_match(hs_resil)
                                cdr_in_hs_res = cdr_repro * hs_resil * nsuit_repro

                                # CDR in warming resilient refugia
                                cdr_repro = cdr.rio.reproject_match(res_bio)
                                nsuit_repro = not_suit.rio.reproject_match(res_bio)
                                cdr_in_bio = cdr_repro * res_bio * nsuit_repro

                            # sum CDr pixel values in bio areas
                            cdr_in_hs = pos_val_summer(cdr_in_hs, squeeze=True)
                            cdr_in_hs_res = pos_val_summer(cdr_in_hs_res, squeeze=True)
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

            # ensure all unique combinations are in df
            all_combi = pd.DataFrame(list(itertools.product(model_fam, cdr_options, years)),
                                     columns=['Model', 'CDR_option', 'Year'])
            exclu_df = pd.merge(all_combi, exclu_df, on=['Model', 'CDR_option', 'Year'],
                                how='left')

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
            exclu_df.replace({'Model': {'MAgPIE': 'REMIND-MAgPIE'}}, inplace=True)

            # plot reduction in land allocated for CDR
            fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=True, sharey=True)

            model_colors = {'AIM': 'darkslategrey', 'GCAM': '#997700', 'GLOBIOM': 'blueviolet',
                            'IMAGE': 'royalblue', 'REMIND-MAgPIE': '#994455'}
            cdr_colors = {'Forestation': 'crimson', 'BECCS': 'darkorange',
                          'Both': 'lightsteelblue'}

            sns.barplot(data=exclu_df.query('Reduct_criteria == "Reduct_hs"'), x='Year',
                        y='Value', hue='CDR_option', legend=True, alpha=0.6, palette=cdr_colors,
                        gap=0, estimator='median', errorbar=('pi', 100), ax=axes[0])

            for model, color in model_colors.items():
                sns.stripplot(data=exclu_df.query(f'Model == "{model}" & Reduct_criteria == "Reduct_hs"'),
                              x='Year', y='Value', hue='CDR_option', dodge=True, jitter=0,
                              s=5, marker='o', edgecolor=color, linewidth=5, legend=False, ax=axes[0])

            sns.barplot(data=exclu_df.query('Reduct_criteria == "Reduct_bio"'), x='Year',
                        y='Value', hue='CDR_option', legend=False, alpha=0.6, palette=cdr_colors,
                        gap=0, estimator='median', errorbar=('pi', 100), ax=axes[1])

            for model, color in model_colors.items():
                sns.stripplot(data=exclu_df.query(f'Model == "{model}" & Reduct_criteria == "Reduct_bio"'),
                              x='Year', y='Value', hue='CDR_option', dodge=True, jitter=0,
                              s=5, marker='o', edgecolor=color, linewidth=5, legend=False, ax=axes[1])

            sns.barplot(data=exclu_df.query('Reduct_criteria == "Reduct_hs_res"'), x='Year',
                        y='Value', hue='CDR_option', legend=False, alpha=0.6, palette=cdr_colors,
                        gap=0, estimator='median', errorbar=('pi', 100), ax=axes[2])

            for model, color in model_colors.items():
                sns.stripplot(data=exclu_df.query(f'Model == "{model}" & Reduct_criteria == "Reduct_hs_res"'),
                              x='Year', y='Value', hue='CDR_option', dodge=True, jitter=0,
                              s=5, marker='o', edgecolor=color, linewidth=5, legend=False, ax=axes[2])

            model_patches = [Line2D([0], [0], marker='o', color='w', label=label,
                                    markerfacecolor=color, markeredgecolor='none', markersize=10)
                             for label, color in model_colors.items()]

            legend1 = axes[0].legend(handles=model_patches, bbox_to_anchor=(1.25, 1.1),
                                     loc='upper left', ncols=5, columnspacing=0.4,
                                     handletextpad=0.1, frameon=False, fontsize=12)

            axes[0].legend(bbox_to_anchor=(-0.05, 1.1), loc='upper left', ncols=5,
                           columnspacing=0.6, handletextpad=0.5, frameon=False, fontsize=12)
            axes[0].add_artist(legend1)

            axes[0].set_xlabel(f'Criteria A: No CDR within \ncurrent biodiversity hotspots', fontsize=11)
            axes[1].set_xlabel(f'Criteria B: No CDR within\n{warm} °C resilient climate refugia', fontsize=11)
            axes[2].set_xlabel(f'Criteria AB: No CDR within\n{warm} °C resilient biodiversity hotspots', fontsize=11)
            axes[0].set_yticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            axes[0].set_ylabel(f'Reduction in CDR land available for allocation in {scenario} [%]\n(median and individual model estimate)',
                               fontsize=12)

            for ax in axes.flat:
                ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
                ax.set_ylim(0, 100)

            if exclu == 'unsuit_bio_areas':
                axes[0].annotate('Allocation of potentially benefiting\nbiodiversity areas allowed in this figure',
                                 xy=(0.03, 0.87), xycoords='axes fraction',
                                 ha='left', fontsize=11, color='black')

            plt.subplots_adjust(wspace=0.1)
            sns.despine()
            plt.show()

# %% plot supplementary figure on model agreement
cell_thresholds = [0.1, 0.2]  # minimum thresholds for cell area shares
model_agreement_thresholds = [2, 3]  # minimum thresholds for model agreement

for model in model_fam:
    for scenario in scenarios:
        for cdr_option in cdr_options:
            for thres_c in cell_thresholds:
                for thres_m in model_agreement_thresholds:
                    for warm in warmings:

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

                        land_in = f'{model}_{cdr_option}_{scenario}_2100.tif'
                        land_temp = f'{model}_{cdr_option}_{scenario}_2100_temp.tif'
                        land_out = f'{model}_{cdr_option}_{scenario}_2100_bin_c{thres_c}_m{thres_m}_w{warm}.tif'

                        land_in = rioxarray.open_rasterio(path / land_in, masked=True)

                        # calculate grid area based on arbitrarily chosen input file
                        if model not in ['IMAGE', 'MAgPIE', 'AIM']:  # for these models use predefined file
                            bin_land = land_in.where(land_in.isnull(), 1)  # all=1 if not nodata
                            bin_land.rio.to_raster(path / 'bin_land.tif', driver='GTiff')
                            land_area_calculation(path, 'bin_land.tif', f'{model}_max_land_area_km2.tif')

                        land_max = rioxarray.open_rasterio(path / f'{model}_max_land_area_km2.tif', masked=True)
                        land_allo_share = land_in / land_max  # estimate cell shares allocated
                        land_allo_share.rio.to_raster(path / land_temp , driver='GTiff')
                        binary_converter(land_temp, path, thres_c, land_out)

for scenario in scenarios:
    for cdr_option in cdr_options:
        for thres_c in cell_thresholds:
            for thres_m in model_agreement_thresholds:
                for warm in warmings:
                    # load area-based criteria for beneficia/harmful effects on biodiversity
                    ref_suit = rioxarray.open_rasterio(path_ref_pot / 'ref_suit.tif', masked=True)
                    ref_not_suit = rioxarray.open_rasterio(path_ref_pot / 'ref_not_suit.tif', masked=True)
                    beccs_suit = rioxarray.open_rasterio(path_beccs_pot / 'beccs_suit.tif', masked=True)
                    beccs_not_suit = rioxarray.open_rasterio(path_beccs_pot / 'beccs_not_suit.tif', masked=True)

                    # calculate model agreement in refugia and check if likely positive or negative
                    aim_lc = rioxarray.open_rasterio(path_aim / f'AIM_{cdr_option}_{scenario}_2100_bin_c{thres_c}_m{thres_m}_w{warm}.tif', masked=True)
                    gcam_lc = rioxarray.open_rasterio(path_gcam / f'GCAM_{cdr_option}_{scenario}_2100_bin_c{thres_c}_m{thres_m}_w{warm}.tif', masked=True)
                    globiom_lc = rioxarray.open_rasterio(path_globiom / f'GLOBIOM_{cdr_option}_{scenario}_2100_bin_c{thres_c}_m{thres_m}_w{warm}.tif', masked=True)
                    image_lc = rioxarray.open_rasterio(path_image / f'IMAGE_{cdr_option}_{scenario}_2100_bin_c{thres_c}_m{thres_m}_w{warm}.tif', masked=True)
                    magpie_lc = rioxarray.open_rasterio(path_magpie / f'MAgPIE_{cdr_option}_{scenario}_2100_bin_c{thres_c}_m{thres_m}_w{warm}.tif', masked=True)

                    if cdr_option == 'Afforestation':
                        suit = ref_suit
                        not_suit = ref_not_suit
                    elif cdr_option == 'BECCS':
                        suit = beccs_suit
                        not_suit = beccs_not_suit

                    res_bio = rioxarray.open_rasterio(path_uea / f'bio{warm}_bin.tif')
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
                    agree_in_bio_pos.rio.to_raster(path_all / f'mi_{cdr_option}_{scenario}_2100_suit_c{thres_c}_m{thres_m}_w{warm}.tif', driver='GTiff')
                    binary_converter(f'mi_{cdr_option}_{scenario}_2100_suit_c{thres_c}_m{thres_m}_w{warm}.tif', path_all, thres_m,
                                     f'mi_{cdr_option}_{scenario}_2100_suit_c{thres_c}_m{thres_m}_w{warm}.tif')
                    agree_in_bio_neg.rio.to_raster(path_all / f'mi_{cdr_option}_{scenario}_2100_not_suit_c{thres_c}_m{thres_m}_w{warm}.tif', driver='GTiff')
                    binary_converter(f'mi_{cdr_option}_{scenario}_2100_not_suit_c{thres_c}_m{thres_m}_w{warm}.tif', path_all, thres_m,
                                     f'mi_{cdr_option}_{scenario}_2100_not_suit_c{thres_c}_m{thres_m}_w{warm}.tif')

for scenario in scenarios:
    for thres_c in cell_thresholds:
        for thres_m in model_agreement_thresholds:
            for warm in warmings:

                ar_suit = rs.open(path_all / f'mi_Afforestation_{scenario}_2100_suit_c{thres_c}_m{thres_m}_w{warm}.tif')
                be_suit = rs.open(path_all / f'mi_BECCS_{scenario}_2100_suit_c{thres_c}_m{thres_m}_w{warm}.tif')
                ar_nsuit = rs.open(path_all / f'mi_Afforestation_{scenario}_2100_not_suit_c{thres_c}_m{thres_m}_w{warm}.tif')
                be_nsuit = rs.open(path_all / f'mi_BECCS_{scenario}_2100_not_suit_c{thres_c}_m{thres_m}_w{warm}.tif')

                thres_cp = round(thres_c * 100)  # from 0-1 to 0-100
                refug = rs.open(path_uea / f'bio{warm}_bin.tif')
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
                    mpatches.Patch(color='gainsboro', label=f'Refugia at {warm} °C'),
                    mpatches.Patch(color='grey', label=f'Hotspot resilient to {warm} °C')]

                legend = ax.legend(bbox_to_anchor=(-0.01, 0.11), handles=legend_patches, ncols=1,
                                   loc='lower left', fontsize=11.5, handlelength=0.7,
                                   handletextpad=0.5, borderpad=0.1, frameon=True)

                legend.get_frame().set_alpha(1)
                legend.get_frame().set_edgecolor('none')

                ax.text(-177, -19, 'Forestation', transform=ccrs.PlateCarree(),
                        fontsize=11, fontweight='bold', zorder=10)

                ax.text(-30, -58, f'{scenario} 2100\nMinimum cell share: {thres_cp}%\nModel agreement: {thres_m}-of-5',
                        transform=ccrs.PlateCarree(), fontsize=11.5, zorder=10)

                ax.set_extent([-180, 167, -90, 90])

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
                    mpatches.Patch(color='gainsboro', label=f'Refugia at {warm} °C'),
                    mpatches.Patch(color='grey', label=f'Hotspot resilient to {warm} °C')]

                legend = ax.legend(bbox_to_anchor=(-0.01, 0.11), handles=legend_patches, ncols=1,
                                   loc='lower left', fontsize=11.5, handlelength=0.7,
                                   handletextpad=0.5, borderpad=0.1, frameon=True)

                legend.get_frame().set_alpha(1)
                legend.get_frame().set_edgecolor('none')

                ax.text(-177, -19, 'BECCS', transform=ccrs.PlateCarree(),
                        fontsize=11, fontweight='bold', zorder=10)

                ax.text(-30, -58, f'{scenario} 2100\nMinimum cell share: {thres_cp}%\nModel agreement: {thres_m}-of-5',
                        transform=ccrs.PlateCarree(), fontsize=11.5, zorder=10)

                ax.set_extent([-180, 167, -90, 90])

                plt.show()
