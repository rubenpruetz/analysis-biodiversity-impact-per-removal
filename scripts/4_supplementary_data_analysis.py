
# import libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')

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
             'Land Cover|Cropland|Energy Crops', 'Land Cover|Forest',
             'Land Cover|Pasture']

lc_data = ar6_db.loc[ar6_db['Variable'].isin(variables)]
lc_data = lc_data.loc[lc_data['Model'].isin(models)]
lc_data = lc_data.loc[lc_data['Scenario'].isin(scenarios)]
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

# calculate change from base year (2010)
lc_data = lc_data.loc[lc_data['Variable'].isin(variables_adjust)]
lc_2010 = lc_data.query('Year == "2010"').reset_index(drop=True)
lc_data = pd.merge(lc_data, lc_2010,
                   on=['Model', 'Scenario', 'Variable'],
                   suffixes=['', '_2010'])

lc_data['Change'] = lc_data['Value'] - lc_data['Value_2010']

# plot change in land cover
lc_data['SSP'] = lc_data['Scenario'].str.split('-').str[0]
lc_data['RCP'] = lc_data['Scenario'].str.split('-').str[1]
lc_data['Year'] = lc_data['Year'].astype(int)

lc_data.replace({'Model': {'AIM/CGE 2.0': 'AIM',
                            'MESSAGE-GLOBIOM 1.0': 'GLOBIOM',
                            'IMAGE 3.0.1': 'IMAGE'}}, inplace=True)

lc_data.replace({'Variable': {'Land Cover|Built-up Area': 'Built-up',
                              'Land Cover|Cropland|Energy Crops': 'Energy cropland',
                              'Land Cover|Forest': 'Forest',
                              'Land Cover|Pasture': 'Pasture'}},
                inplace=True)

var_pal = {'Built-up': 'red',
           'Energy cropland': 'blue',
           'Forest': 'green',
           'Other cropland': 'brown',
           'Pasture': 'orange'}
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

axes[0, 0].legend(bbox_to_anchor=(-0.05, 1.3), loc='upper left', ncols=5, 
                  columnspacing=1, handletextpad=0.4, fontsize=11)

fig.supylabel(f'Land cover change from 2010 [mio.ha] (SSP1-SSP3 range as shading)',
              x=0.02, va='center', ha='center')

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.25)
sns.despine()
plt.show()
