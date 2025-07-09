
# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from required_functions import *
plt.rcParams.update({'figure.dpi': 600})

path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')

path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
energy_crop_share = pd.read_csv(path_all / 'share_energy_crops_estimates.csv')

# %% plot supplementary figure on BECCS in Annex-I and Non-Annex I refugia

paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
area_df = load_and_concat('area_df_annex_group_temp_decline_not_allowed', paths)

rcps = ['19', '26', '45']  # specify RCPs that shall be plotted
area_df = area_df.loc[area_df['RCP'].isin(rcps)]

rcp_pal = {'19': '#00adcf', '26': '#173c66', '34': '#f79320',
           '45': '#e71d24', '60': '#951b1d', 'Baseline': 'dimgrey'}
all_rcps = sorted(area_df['RCP'].unique())

fig, axes = plt.subplots(3, 2, figsize=(8, 8), sharex=True, sharey=False)
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

axes[1, 0].legend(handles, new_labels, bbox_to_anchor=(-0.02, 2.7), loc='upper left',
                  ncols=3, columnspacing=1, handletextpad=0.4)

axes[0, 0].set_title('Annex I\n(BECCS)')
axes[0, 1].set_title('Non-Annex I\n(BECCS)')

axes[2, 0].set_xlabel('')
axes[2, 1].set_xlabel('')

axes[0, 0].set_ylabel('AIM')
axes[1, 0].set_ylabel('GLOBIOM')
axes[2, 0].set_ylabel('IMAGE')

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

fig.supylabel(f'Share of remaining refugia allocated for CDR [%] (SSP1-SSP3 range as shading)',
              x=0.05, va='center', ha='center')

for ax in axes.flat:
    ax.set_xlim(2020, 2100)
    ax.set_xticks([2020, 2100])
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

plt.subplots_adjust(hspace=0.25)
plt.subplots_adjust(wspace=0.2)
sns.despine()
plt.show()

# %% get AR6 land cover data for SSP-RCP combinations
years = ['2010', '2020', '2030', '2040', '2050', '2060', '2070', '2080',
         '2090', '2100']

models = ['MESSAGE-GLOBIOM 1.0', 'AIM/CGE 2.0', 'IMAGE 3.0.1']
scenarios = ['SSP1-19', 'SSP1-26', 'SSP1-45', 'SSP2-19', 'SSP2-26', 'SSP2-45',
             'SSP3-45']
variables = ['Land Cover|Built-up Area', 'Land Cover|Cropland',
             'Land Cover|Cropland|Energy Crops', 'Land Cover|Forest',
             'Land Cover|Pasture']
variables_adjust = ['Land Cover|Built-up Area', 'Other cropland',
                    'Land Cover|Forest', 'Energy cropland (for BECCS)',
                    'Energy cropland (not for BECCS)', 'Land Cover|Pasture']

ar6_db = ar6_db.loc[ar6_db['Model'].isin(models)]
ar6_db = ar6_db.loc[ar6_db['Scenario'].isin(scenarios)]
lc_data = ar6_db.loc[ar6_db['Variable'].isin(variables)]
lc_data = lc_data[['Model', 'Scenario', 'Variable'] + years].copy()

lc_data = pd.melt(lc_data, id_vars=['Model', 'Scenario', 'Variable'],
                  value_vars=years, var_name='Year', value_name='Value')

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
lc_data['Change'] = lc_data['Change'] / 100  # from ha to km2

# plot supplementary figure on land use changes based on AR6 Scenarios Database
lc_data['SSP'] = lc_data['Scenario'].str.split('-').str[0]
lc_data['RCP'] = lc_data['Scenario'].str.split('-').str[1]
lc_data['Year'] = lc_data['Year'].astype(int)

lc_data.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                           'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                           'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

lc_data.replace({'Variable': {'Land Cover|Built-up Area': 'Built-up',
                              'Land Cover|Forest': 'Forest',
                              'Land Cover|Pasture': 'Pasture'}},
                inplace=True)

var_pal = {'Built-up': 'dimgrey',
           'Energy cropland (for BECCS)': 'orangered',
           'Energy cropland (not for BECCS)': 'blue',
           'Forest': 'forestgreen',
           'Other cropland': 'brown',
           'Pasture': 'gold'}
all_vars = sorted(lc_data['Variable'].unique())

fig, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)
sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=True, ax=axes[0, 0])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 0])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "19"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 0])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 1])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 1])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "26"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 1])

sns.lineplot(data=lc_data.query('Model == "AIM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[0, 2])
sns.lineplot(data=lc_data.query('Model == "GLOBIOM" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[1, 2])
sns.lineplot(data=lc_data.query('Model == "IMAGE" & RCP == "45"'),
             x='Year', y='Change', hue='Variable', palette=var_pal,
             errorbar=('pi', 100), estimator='median', legend=False, ax=axes[2, 2])

for ax in axes.flat:
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xlim(2010, 2100)
    ax.set_xticks([2010, 2055, 2100])
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.8)

axes[0, 0].set_ylabel('AIM', fontsize=11)
axes[1, 0].set_ylabel('GLOBIOM', fontsize=11)
axes[2, 0].set_ylabel('IMAGE', fontsize=11)

axes[0, 0].set_title('1.5 °C')
axes[0, 1].set_title('2 °C')
axes[0, 2].set_title('Current Policies')

axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.43), loc='upper left', ncols=3,
                  columnspacing=1, handletextpad=0.4, fontsize=11)

fig.supylabel(f'Land cover change from 2010 [Mkm$^2$] (SSP1-SSP3 range as shading)',
              x=0.03, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.25)
sns.despine()
plt.show()

# %%
paths = {'GLOBIOM': path_globiom, 'AIM': path_aim, 'IMAGE': path_image}
ar_removal = load_and_concat('ar_removal', paths)
beccs_removal = load_and_concat('beccs_removal', paths)
