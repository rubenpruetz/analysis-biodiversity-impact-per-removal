
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

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_mi_cdr_df = pd.read_csv(path_all / 'lookup_table_ar_beccs_files_all_models.csv')
lookup_mi_cdr_df['year'] = lookup_mi_cdr_df['year'].astype(str)

tcre_df = pd.read_csv(path_ar6_data / 'tcre_estimates.csv')
p50_est = float(tcre_df[(tcre_df['Source'] == 'Own trans') &
                        (tcre_df['Estimate'] == 'point')]['Value'].iloc[0])

# %% choose model to run the script with
model = 'IMAGE'  # options: 'GLOBIOM' or 'AIM' or 'IMAGE'

if model == 'GLOBIOM':
    path = path_globiom
    model_setup = 'MESSAGE-GLOBIOM 1.0'
    removal_lvl = 3
elif model == 'AIM':
    path = path_aim
    model_setup = 'AIM/CGE 2.0'
    removal_lvl = 3
elif model == 'IMAGE':
    path = path_image
    model_setup = 'IMAGE 3.0.1'
    removal_lvl = 3

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

# %% choose between biodiv recovery or no recovery after peak warming

temperature_decline = 'not_allowed'  # options: 'allowed' or 'not_allowed'

if temperature_decline == 'allowed':
    warm_file = ar6_data_r.copy()
elif temperature_decline == 'not_allowed':
    warm_file = ar6_data_stab.copy()

bio_select = warm_file.set_index(['Model', 'Scenario'])
bio_select = 'bio' + \
    bio_select.select_dtypes(include=np.number).astype(str) + '_bin.tif'
bio_select.reset_index(inplace=True)

# %% calculate CDR land impact over time
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

area_df.to_csv(path / f'{model}_area_df_annex_group_temp_decline_{temperature_decline}.csv', index=False)

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

# %% plot avoided warming-related refugia loss due CDR
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
#%%
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
        globals()[f'df_{ssp}']['option'] = 'BECCS'
    except Exception as e:
        print(f'Error processing {ssp}: {e}')
        globals()[f'df_{ssp}'] = pd.DataFrame()
        continue
df_be = pd.concat([df_SSP1, df_SSP2, df_SSP3], axis=0)

# concat output dfs for AR and BECCS
df_options = pd.concat([df_ar, df_be], axis=0)

# calculate refugia area
land_area_calculation(path_uea, 'bio1.8_bin.tif', 'bio1.8_bin_km2.tif')

# use admin_bound_calculator to calc refugia area per country
intersect_src = rs.open(path_uea / 'bio1.8_bin_km2.tif')
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), subplot_kw={'projection': ccrs.Robinson()})

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

    ax.set_extent([-167, 167, -58, 90])

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        ax=ax, orientation='horizontal',
                        boundaries=bounds, ticks=bounds,
                        spacing='proportional', extend='max')
    cbar.ax.set_position([0.346, -0.175, 0.334, 0.5])
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label(f'Share of national refugia covered by Forestation \nand BECCS for removals of {cdr_sum} GtCO$_2$ [%]',
                   fontsize=15)
    plt.title(f'{model} {ssp}-{rcp_lvl}', fontsize=12.5, x=0.04, y=0.2, ha='left')
    plt.show()

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
