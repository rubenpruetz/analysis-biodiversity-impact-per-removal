
# import libraries
import pandas as pd
import rioxarray
import rasterio as rs
from required_functions import *
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
file_all = 'lookup_table_ar_bioenergy_files_all_models.csv'

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_names = pd.read_csv(path_globiom / 'lookup_table_ssp-rcp_names.csv')
energy_crop_share = pd.read_csv(path_all / 'share_energy_crops_estimates.csv')

# %% choose model to run the script with
model = 'IMAGE'  # options: 'GLOBIOM' or 'AIM' or 'IMAGE'

if model == 'GLOBIOM':
    path = path_globiom
    model_setup = 'MESSAGE-GLOBIOM 1.0'
elif model == 'AIM':
    path = path_aim
    model_setup = 'AIM/CGE 2.0'
elif model == 'IMAGE':
    path = path_image
    model_setup = 'IMAGE 3.0.1'

# land-per-removal curve calculation
# %% STEP1: calculate removal per scenario for 2010-2100
scenarios = ['SSP1-Baseline', 'SSP1-19', 'SSP1-26', 'SSP1-34', 'SSP1-45',
             'SSP2-Baseline', 'SSP2-19', 'SSP2-26', 'SSP2-34', 'SSP2-45',
             'SSP2-60', 'SSP3-Baseline', 'SSP3-34', 'SSP3-45', 'SSP3-60']

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

# calculate BECCS removal through energy crops only (no residues)
ec_share = energy_crop_share.loc[energy_crop_share['Model'].isin([model])]
beccs_removal = pd.merge(beccs_removal,
                         ec_share[['Scenario', 'Year', 'Share_energy_crops']],
                         on=['Scenario', 'Year'])
beccs_removal['Removal'] = beccs_removal['Removal'] * beccs_removal['Share_energy_crops']

# calculate share of bioenergy for BECCS based on biomass with/without CCS
bioeng_ncss = ar6_db.query('Variable == "Primary Energy|Biomass|Modern|w/o CCS"').reset_index(drop=True)
bioeng_ncss[numeric_cols20] = bioeng_ncss[numeric_cols20].round(2)
bioeng_tot = ar6_db.query('Variable == "Primary Energy|Biomass"').reset_index(drop=True)
bioeng_tot[numeric_cols20] = bioeng_tot[numeric_cols20].round(2)

bioeng_wccs = bioeng_tot[['Scenario']].copy()
bioeng_wccs[numeric_cols20] = 1 - (bioeng_ncss[numeric_cols20] / bioeng_tot[numeric_cols20])
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

# %% compute land-per-removal, removal, and land for AR and BECCS
lpr_ar = process_data(ar_land, ar_removal, 'AR')  # plot AR data
lpr_beccs = process_data(beccs_land, beccs_removal, 'BECCS')  # plot BECCS data

# %% impact-per-removal analysis (afforestation)

rcp_lvl = '26'  # select RCP level (without dot)

lpr_ar_strict = lpr_ar.loc[lpr_ar['RCP'].isin([rcp_lvl])]
removal_steps = [3]  # specify CDR levels (add more if required)

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
                # read raster data and geospatial information
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

lpr_beccs_strict = lpr_beccs.loc[lpr_beccs['RCP'].isin([rcp_lvl])]

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
                # read raster data and geospatial information
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

# %% generate spatial BECCS file per scenario per year

beccs_land_non_zero = beccs_land[beccs_land['Land'] > 0]

for index, row in beccs_land_non_zero.iterrows():
    scenario = row['Scenario']
    year = row['Year']
    fraction = row['Fraction']

    input_name = f'{model}_Bioenergy_{scenario}_{year}.tif'
    output_name = f'{model}_BECCS_{scenario}_{year}.tif'

    with rs.open(path / input_name) as src_input:
        # read raster data and geospatial information
        input_tiff = src_input.read(1)
        profile_input = src_input.profile

        # multiply bioenery cells by BECCS fraction assuming even distribution
        fract_tiff = input_tiff * fraction

        profile_updated = profile_input.copy()
        profile_updated.update(dtype=rs.float32)

        with rs.open(path / output_name, "w", **profile_updated) as dst:
            dst.write(fract_tiff.astype(rs.float32), 1)
