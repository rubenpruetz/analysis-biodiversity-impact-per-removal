
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
import rioxarray
import rasterio as rs
from time import time
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cartopy.feature as cfeature
import cmasher as cmr
from required_functions import *
import shapefile
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_ag = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/unfccc_annex')
path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
path_hotspots = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ar6_hotspots')
sf_path = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/wab')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_mi_cdr_df = pd.read_csv(path_all / 'lookup_table_ar_beccs_files_all_models.csv')
lookup_mi_cdr_df['year'] = lookup_mi_cdr_df['year'].astype(str)

tcre_df = pd.read_csv(path_ar6_data / 'tcre_estimates.csv')
p50_est = float(tcre_df[(tcre_df['Source'] == 'Own trans') &
                        (tcre_df['Estimate'] == 'point')]['Value'].iloc[0])

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

# rename models for the subsequent step
ar6_data.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                            'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                            'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

ar6_data_r = ar6_data.round(2)  # round for 1 or 2 digits

# allow no temperature decline by calculating peak warming up until each year
for year in range(2021, 2101):
    cols_til_year = ar6_data_r.loc[:, '2020':str(year)]
    ar6_data_r[f'{year}_max'] = cols_til_year.max(axis=1)

cols = ['Model', 'Scenario', '2020'] + [f'{year}_max' for year in range(2021, 2101)]
ar6_data_stab = ar6_data_r[cols]
ar6_data_stab = ar6_data_stab.rename(columns={f'{year}_max': str(year) for year in all_years})

ar6_data_r = ar6_data_r[['Model', 'Scenario'] + all_years].copy()

# %% choose model to run the script with and recovery assumption after peak warming
models = ['AIM', 'GLOBIOM', 'IMAGE']
temp_declines = ['not_allowed', 'allowed']

for model in models:
    for temp_decline in temp_declines:
        if model == 'GLOBIOM':
            path = path_globiom
            model_setup = 'MESSAGE-GLOBIOM 1.0'
        elif model == 'AIM':
            path = path_aim
            model_setup = 'AIM/CGE 2.0'
        elif model == 'IMAGE':
            path = path_image
            model_setup = 'IMAGE 3.0.1'

        if temp_decline == 'allowed':
            warm_file = ar6_data_r.copy()
        elif temp_decline == 'not_allowed':
            warm_file = ar6_data_stab.copy()

        bio_select = warm_file.set_index(['Model', 'Scenario'])
        bio_select = 'bio' + \
            bio_select.select_dtypes(include=np.number).astype(str) + '_bin.tif'
        bio_select.reset_index(inplace=True)

        # calculate CDR land impact over time
        years = ['2020', '2030', '2040', '2050', '2060', '2070', '2080', '2090', '2100']
        lookup_sub_yrs = lookup_mi_cdr_df.copy()
        lookup_sub_yrs = lookup_sub_yrs.loc[lookup_sub_yrs['year'].isin(years)]

        # load annex group files
        ag1 = rioxarray.open_rasterio(path_ag / 'annex_i.tif', masked=True)
        ag2 = rioxarray.open_rasterio(path_ag / 'non_annex_i.tif', masked=True)

        start = time()  # runtime monitoring

        def overlay_calculator(input_tif,  # land use model input file (string)
                               filepath,  # filepath input file + / (string)
                               file_year,  # year of input file (string)
                               file_scenario,  # input file SSP-RCP scenario (string)
                               mitigation_option,  # 'Afforestation' or 'BECCS'
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

            ag_values = [ag1, ag2]

            for ag in ag_values:
                ag_m = ag.rio.reproject_match(land_use)  # ensure consistent match
                cdr_in_bio_ag = (land_use * refugia) * ag_m

                refugia_ag = refugia * ag_m
                refugia_ag.rio.to_raster(path_uea / 'refugia_ag_temp.tif', driver='GTiff')
                refugia_ag = land_area_calculation(path_uea, 'refugia_ag_temp.tif')

                cdr_in_bio_ag = pos_val_summer(cdr_in_bio_ag, squeeze=True)
                refugia_ag = pos_val_summer(refugia_ag, squeeze=True)

                # calculate regional area "losses" and refugia
                cdr_in_bio_regs.append(cdr_in_bio_ag)
                bio_area_regs.append(refugia_ag)

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

            except Exception as e:
                print(f'Unsuccessful for file {input_tif}: {e}')
                return {
                    'scenario': file_scenario,
                    'mitigation_option': mitigation_option,
                    'year': file_year,
                    'bio_area': float('nan'),
                    'cdr_in_bio': float('nan'),
                    'bio_area_reg': [float('nan')] * 2,
                    'cdr_in_bio_reg': [float('nan')] * 2}


        area_df = pd.DataFrame.from_records(lookup_sub_yrs.apply(process_row,
                                                                 axis=1).values)
        area_df = area_df.reset_index(drop=True)

        ag_columns = pd.DataFrame(area_df['bio_area_reg'].to_list(),
                                  columns=['bio_area_ag1',
                                           'bio_area_ag2'])
        area_df = pd.concat([area_df.drop(columns='bio_area_reg'), ag_columns], axis=1)

        ag_columns = pd.DataFrame(area_df['cdr_in_bio_reg'].to_list(),
                                  columns=['cdr_in_bio_ag1',
                                           'cdr_in_bio_ag2'])
        area_df = pd.concat([area_df.drop(columns='cdr_in_bio_reg'), ag_columns], axis=1)

        end = time()
        print(f'Runtime {(end - start) / 60} min')

        area_df['alloc_perc'] = area_df['cdr_in_bio'] / area_df['bio_area'] * 100
        for i in range(1, 3):  # calculate land loss percentages for both annex group files
            area_df[f'alloc_perc_ag{i}'] = area_df[f'cdr_in_bio_ag{i}'] / area_df[f'bio_area_ag{i}'] * 100

        area_df['SSP'] = area_df['scenario'].str.split('-').str[0]
        area_df['RCP'] = area_df['scenario'].str.split('-').str[1]
        area_df.rename(columns={'scenario': 'Scenario'}, inplace=True)
        area_df.rename(columns={'year': 'Year'}, inplace=True)
        area_df['Year'] = area_df['Year'].astype(int)
        area_df['Model'] = f'{model}'
        area_df.to_csv(path / f'{model}_area_df_annex_group_temp_decline_{temp_decline}.csv', index=False)

# %% plot land allocation within refugia across scenarios

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
all_rcps = sorted(area_df['RCP'].unique())

fig, axes = plt.subplots(3, 4, figsize=(8, 6), sharex=True, sharey=False)
sns.lineplot(data=area_df.query('Model == "AIM" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 0])
sns.lineplot(data=area_df.query('Model == "GLOBIOM" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc',  palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', hue_order=all_rcps, legend=True, ax=axes[1, 0])
sns.lineplot(data=area_df.query('Model == "IMAGE" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc', palette=rcp_pal, hue='RCP',
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

sns.lineplot(data=area_df.query('Model == "AIM" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc_ag1', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 2])
sns.lineplot(data=area_df.query('Model == "GLOBIOM" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc_ag1', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 2])
sns.lineplot(data=area_df.query('Model == "IMAGE" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc_ag1', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 2])

sns.lineplot(data=area_df.query('Model == "AIM" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc_ag2', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 3])
sns.lineplot(data=area_df.query('Model == "GLOBIOM" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc_ag2', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 3])
sns.lineplot(data=area_df.query('Model == "IMAGE" & mitigation_option == "Afforestation"'),
             x='Year', y='alloc_perc_ag2', palette=rcp_pal, hue='RCP',
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 3])

handles, labels = axes[1, 0].get_legend_handles_labels()
rename_dict = {'19': '1.5 °C', '26': '2 °C', '45': 'Current Policies'}

new_labels = [rename_dict.get(label, label) for label in labels]

axes[1, 0].legend(handles, new_labels, bbox_to_anchor=(-0.05, 2.9), loc='upper left',
                  ncols=3, columnspacing=1, handletextpad=0.4)

axes[0, 0].set_title('Global\n(Forestation)')
axes[0, 1].set_title('Global\n(BECCS)')
axes[0, 2].set_title('Annex I\n(Forestation)')
axes[0, 3].set_title('Non-Annex I\n(Forestation)')

axes[2, 0].set_xlabel('')
axes[2, 1].set_xlabel('')
axes[2, 2].set_xlabel('')
axes[2, 3].set_xlabel('')

axes[0, 0].set_ylabel('AIM')
axes[1, 0].set_ylabel('GLOBIOM')
axes[2, 0].set_ylabel('IMAGE')

axes[0, 1].set_ylabel('')
axes[1, 1].set_ylabel('')
axes[2, 1].set_ylabel('')
axes[0, 2].set_ylabel('')
axes[1, 2].set_ylabel('')
axes[2, 2].set_ylabel('')
axes[0, 3].set_ylabel('')
axes[1, 3].set_ylabel('')
axes[2, 3].set_ylabel('')

for ax in axes[:, 0]:
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylim([0, 12])
for ax in axes[:, 1]:
    ax.set_yticks([0, 3, 6, 9, 12])
    ax.set_ylim([0, 12])
    ax.set_yticklabels([])
for ax in axes[:, 2]:
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylim([0, 16])
for ax in axes[:, 3]:
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_ylim([0, 16])
    ax.set_yticklabels([])

fig.supylabel(f'Share of remaining refugia allocated for CDR [%]',
              x=0.05, va='center', ha='center')

for ax in axes.flat:
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2100])
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

plt.subplots_adjust(hspace=0.25)
plt.subplots_adjust(wspace=0.5)
sns.despine()
plt.show()

# %% plot avoided warming-related refugia loss due to CDR
paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
ar_removal = load_and_concat('ar_removal', paths)
beccs_removal = load_and_concat('beccs_removal', paths)

# calculate cumulative removals and avoided warming based on TCRE
ar_cum = cum_cdr_calc(ar_removal)
ar_cum['CoolAR'] = ar_cum['Cum'] * p50_est * 1000  # x1000 since Gt not Mt

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

# %% maps refugia land impact of CDR across SSP1-3 for a certain warming level

rcp_lvl = '26'  # select RCP level (without dot)
ssps = ['SSP1', 'SSP2', 'SSP3']

for model in models:

    if model == 'GLOBIOM':
        path = path_globiom
        removal_lvl = 3
    elif model == 'AIM':
        path = path_aim
        removal_lvl = 3
    elif model == 'IMAGE':
        path = path_image
        removal_lvl = 3

    for ssp in ssps:

        try:
            ar = f'{model}_Afforestation_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'
            be = f'{model}_BECCS_{ssp}-{rcp_lvl}_{removal_lvl}GtCO2.tif'

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

            fig = plt.figure(figsize=(10, 6.1))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())  # choose projection | LambertAzimuthalEqualArea())

            img_re = ax.imshow(data_refug, extent=extent_refug, transform=ccrs.PlateCarree(),
                               origin='upper', cmap='Greys', alpha=0.1)

            img_ar = ax.imshow(data_ar, extent=extent_ar, transform=ccrs.PlateCarree(),
                               origin='upper', cmap=cmap_ar, norm=norm_ar, alpha=1)

            img_be = ax.imshow(data_be, extent=extent_be, transform=ccrs.PlateCarree(),
                               origin='upper', cmap=cmap_be, norm=norm_be, alpha=0.7)

            ax.coastlines(linewidth=0.2)

            ax.set_extent([-167, 167, -58, 90])

            cbar_ar = plt.colorbar(img_ar, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
            cbar_be = plt.colorbar(img_be, ax=ax, orientation='horizontal', aspect=9, pad=0.16)
            cbar_ar.ax.set_position([0.35, 0, 0.1, 0.501])
            cbar_be.ax.set_position([0.576, 0, 0.1, 0.501])
            cbar_ar.ax.tick_params(labelsize=10)
            cbar_be.ax.tick_params(labelsize=10)
            cbar_ar.set_label(f'Forestation per grid cell \nfor removals of {removal_lvl} GtCO$_2$ [%]',
                              labelpad=1, fontsize=10.5)
            cbar_be.set_label(f'BECCS per grid cell \nfor removals of {removal_lvl} GtCO$_2$ [%]',
                              labelpad=1, fontsize=10.5)
            plt.title(f'{model} {ssp}-{rcp_lvl}', fontsize=8.6, x=0.04, y=0.2,
                      ha='left')
            plt.show()
        except Exception as e:
            print(f'Error processing {ssp}: {e}')
            continue

# %% calculate country burden for refugia
# read the administrative boundary shapefile data
admin_sf = shapefile.Reader(sf_path / 'world-administrative-boundaries.shp')

for model in models:
    if model == 'GLOBIOM':
        path = path_globiom
        removal_lvl = 3
    elif model == 'AIM':
        path = path_aim
        removal_lvl = 3
    elif model == 'IMAGE':
        path = path_image
        removal_lvl = 3

    for ssp in ssps:
        dfs = []

        try:
            intersect_src = rs.open(path / f'{ssp}_ar{removal_lvl}_bio_absolute.tif')
            df_ar = admin_bound_calculator(ssp, admin_sf, intersect_src)
            df_ar['option'] = 'AR'
            dfs.append(df_ar)
        except Exception as e:
            print(f'Error processing AR {ssp}: {e}')
            continue

        try:
            intersect_src = rs.open(path / f'{ssp}_be{removal_lvl}_bio_absolute.tif')
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

        cdr_sum = int(2 * removal_lvl)
        cbar.set_label(
            f'Share of national refugia covered by Forestation \nand BECCS for removals of {cdr_sum} GtCO$_2$ [%]',
            fontsize=15)
        plt.title(f'{model} {ssp}-{rcp_lvl}', fontsize=12.5, x=0.04, y=0.2, ha='left')
        plt.show()
