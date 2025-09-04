
# import libraries
import pandas as pd
import rioxarray
import rasterio as rs
from pathlib import Path
from required_functions import *

path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
file_all = 'lookup_table_ar_bioenergy_files_all_models.csv'

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
lookup_names = pd.read_csv(path_globiom / 'lookup_table_ssp-rcp_names.csv')
energy_crop_share = pd.read_csv(path_all / 'share_energy_crops_estimates.csv')

# %% specify models to run the script with
models = ['AIM', 'GLOBIOM', 'IMAGE']

for model in models:

    if model == 'GLOBIOM':
        path = path_globiom
        model_setup = 'MESSAGE-GLOBIOM 1.0'
    elif model == 'AIM':
        path = path_aim
        model_setup = 'AIM/CGE 2.0'
    elif model == 'IMAGE':
        path = path_image
        model_setup = 'IMAGE 3.0.1'

    # STEP1: calculate removal per scenario for 2010-2100
    scenarios = ['SSP1-19', 'SSP1-26', 'SSP1-45', 'SSP2-19', 'SSP2-26',
                 'SSP2-45', 'SSP3-45']

    numeric_cols20 = [str(year) for year in range(2020, 2110, 10)]

    if model == 'GLOBIOM':
        ar_variable = 'Carbon Sequestration|Land Use|Afforestation'
    elif model == 'AIM':
        ar_variable = 'Carbon Sequestration|Land Use|Afforestation'
    elif model == 'IMAGE':  # for IMAGE afforestation is not available
        ar_variable = 'Carbon Sequestration|Land Use'

    ar6_m = ar6_db.loc[ar6_db['Model'].isin([model_setup]) & ar6_db['Scenario'].isin(scenarios)]
    cdr = ar6_m[['Scenario', 'Variable'] + numeric_cols20].copy()
    cdr_array = ['Carbon Sequestration|CCS|Biomass', ar_variable]
    cdr = cdr[cdr['Variable'].isin(cdr_array)]
    cdr[numeric_cols20] = cdr[numeric_cols20].clip(lower=0)  # set negative values to zero
    cdr = cdr.melt(id_vars=['Scenario', 'Variable'], var_name='Year', value_name='Removal')
    cdr['Removal'] = cdr['Removal'] * 0.001  # Mt to Gt
    cdr['Year'] = pd.to_numeric(cdr['Year'])

    ar_removal = cdr[cdr['Variable'] == ar_variable]
    ar_removal['Variable'] = 'AR removal'
    ar_removal['Model'] = model
    ar_removal.to_csv(path /f'{model}_ar_removal.csv', index=False)

    # STEP2: calculate afforestation land per scenario in for 2020-2100
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

    # calculate BECCS removal and bioenergy land use fractions
    beccs_removal = cdr[cdr['Variable'] == 'Carbon Sequestration|CCS|Biomass']
    beccs_removal['Variable'] = 'BECCS removal'

    # calculate BECCS removal through energy crops only (no residues)
    ec_share = energy_crop_share.loc[energy_crop_share['Model'].isin([model])]
    beccs_removal = pd.merge(beccs_removal,
                             ec_share[['Scenario', 'Year', 'Share_energy_crops']],
                             on=['Scenario', 'Year'])
    beccs_removal['Removal'] = beccs_removal['Removal'] * beccs_removal['Share_energy_crops']
    beccs_removal['Model'] = model
    beccs_removal.to_csv(path /f'{model}_beccs_removal.csv', index=False)

    # filter BECCS related files from lookup table
    be_lookup = landfile_lookup[landfile_lookup['mitigation_option'] == 'Bioenergy plantation']
    beccs_lookup = be_lookup.copy()
    beccs_lookup['file_name'] = beccs_lookup['file_name'].str.replace('Bioenergy', 'BECCS')

    beccs_land = pd.DataFrame(columns=['Scenario', 'Year', 'Land'])

    for index, row in beccs_lookup.iterrows():  # calculate bioenergy land per scenario
        input_nc = row['file_name']
        scenario = row['scenario']
        year = row['year']

        try:
            land_use = rioxarray.open_rasterio(path / f'{model}_{input_nc}',
                                               masked=True)
            tot_cdr_area = pos_val_summer(land_use, squeeze=True)

            beccs_land = pd.concat([beccs_land, pd.DataFrame({
                'Scenario': [scenario],
                'Year': [year],
                'Land': [tot_cdr_area]
            })], ignore_index=True)
        except Exception as e:
            print(f'Error processing BECCS: {e}')
            continue

    beccs_land['Variable'] = 'Land demand'
    beccs_land['Year'] = beccs_land['Year'].apply(pd.to_numeric)

    # compute land-per-removal, removal, and land for AR and BECCS
    lpr_ar = process_data(ar_land, ar_removal)
    lpr_beccs = process_data(beccs_land, beccs_removal)

    # impact-per-removal analysis (afforestation)

    rcp_lvl = '26'  # select RCP level (without dot)

    lpr_ar_strict = lpr_ar.loc[lpr_ar['RCP'].isin([rcp_lvl])]
    removal_step = 3  # specify CDR levels (add more if required)

    # use yr_target_finder to find the year of a givenr removal
    ar_range = yr_target_finder(lpr_ar_strict, removal_step)

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

    # impact-per-removal analysis (BECCS)

    lpr_beccs_strict = lpr_beccs.loc[lpr_beccs['RCP'].isin([rcp_lvl])]

    # use yr_target_finder to find the year of a givenr removal
    beccs_range = yr_target_finder(lpr_beccs_strict, removal_step)

    # interpolate land use layers to yr_target

    for index, row in beccs_range.iterrows():
        ssp = row['SSP']
        rcp = row['RCP']
        yr_low = row['Year_low']
        yr_up = row['Year_up']
        yr_target = row['yr_target']

        lower_tiff = f'{model}_BECCS_{ssp}-{rcp}_{yr_low}.tif'
        upper_tiff = f'{model}_BECCS_{ssp}-{rcp}_{yr_up}.tif'
        output_name = f'{model}_BECCS_{ssp}-{rcp}_{removal_step}GtCO2.tif'

        with rs.open(path / lower_tiff) as src_low:
            with rs.open(path / upper_tiff) as src_up:
                # read raster data and geospatial information
                lower_tiff = src_low.read(1)
                upper_tiff = src_up.read(1)
                profile_lower = src_low.profile

                lower_beccs = lower_tiff * 0.000001  # km2 to Mkm2
                upper_beccs = upper_tiff * 0.000001  # km2 to Mkm2

                yr_diff = yr_up - yr_low  # diff of known years
                tiff_diff = upper_beccs - lower_beccs  # diff of known tiffs

                # lower tiff plus the fraction of tiff_diff for a given target yr
                tiff_target = lower_beccs + (tiff_diff * ((yr_target - yr_low) / yr_diff))

                profile_updated = profile_lower.copy()
                profile_updated.update(dtype=rs.float32)

                with rs.open(path / output_name, "w", **profile_updated) as dst:
                    dst.write(tiff_target.astype(rs.float32), 1)

    # do sensitivities for a certain removal, irrespective of CDR option split
    lpr_sum = pd.merge(lpr_ar, lpr_beccs, on=['SSP', 'RCP', 'Year'],
                       suffixes=['_ar', '_beccs'])
    lpr_sum['Removal'] = lpr_sum['Removal_ar'] + lpr_sum['Removal_beccs']
    lpr_sum = lpr_sum[['SSP', 'RCP', 'Year', 'Removal']].copy()

    removal_steps = [6, 10]
    for removal_step in removal_steps:
        cdr_sum_range = yr_target_finder(lpr_sum, removal_step)

        # interpolate land use layers to yr_target

        for index, row in cdr_sum_range.iterrows():
            ssp = row['SSP']
            rcp = row['RCP']
            yr_low = row['Year_low']
            yr_up = row['Year_up']
            yr_target = row['yr_target']

            lower_ar = f'{model}_Afforestation_{ssp}-{rcp}_{yr_low}.tif'
            upper_ar = f'{model}_Afforestation_{ssp}-{rcp}_{yr_up}.tif'
            output_ar = f'{model}_Afforestation_{ssp}-{rcp}_sum{removal_step}GtCO2.tif'

            with rs.open(path / lower_ar) as src_low:
                with rs.open(path / upper_ar) as src_up:
                    # read raster data and geospatial information
                    lower_ar = src_low.read(1)
                    upper_ar = src_up.read(1)
                    profile_lower = src_low.profile

                    lower_ar = lower_ar * 0.000001  # km2 to Mkm2
                    upper_ar = upper_ar * 0.000001  # km2 to Mkm2

                    yr_diff = yr_up - yr_low  # diff of known years
                    tiff_diff = upper_ar - lower_ar  # diff of known tiffs

                    # lower tiff plus the fraction of tiff_diff for a given target yr
                    tiff_target = lower_ar + (tiff_diff * ((yr_target - yr_low) / yr_diff))

                    profile_updated = profile_lower.copy()
                    profile_updated.update(dtype=rs.float32)

                    with rs.open(path / output_ar, "w", **profile_updated) as dst:
                        dst.write(tiff_target.astype(rs.float32), 1)

            lower_beccs = f'{model}_BECCS_{ssp}-{rcp}_{yr_low}.tif'
            upper_beccs = f'{model}_BECCS_{ssp}-{rcp}_{yr_up}.tif'
            output_name = f'{model}_BECCS_{ssp}-{rcp}_sum{removal_step}GtCO2.tif'

            with rs.open(path / lower_beccs) as src_low:
                with rs.open(path / upper_beccs) as src_up:
                    # read raster data and geospatial information
                    lower_beccs = src_low.read(1)
                    upper_beccs = src_up.read(1)
                    profile_lower = src_low.profile

                    lower_beccs = lower_beccs * 0.000001  # km2 to Mkm2
                    upper_beccs = upper_beccs * 0.000001  # km2 to Mkm2

                    yr_diff = yr_up - yr_low  # diff of known years
                    tiff_diff = upper_beccs - lower_beccs  # diff of known tiffs

                    # lower tiff plus the fraction of tiff_diff for a given target yr
                    tiff_target = lower_beccs + (tiff_diff * ((yr_target - yr_low) / yr_diff))

                    profile_updated = profile_lower.copy()
                    profile_updated.update(dtype=rs.float32)

                    with rs.open(path / output_name, "w", **profile_updated) as dst:
                        dst.write(tiff_target.astype(rs.float32), 1)
