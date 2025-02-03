# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import rioxarray
import rasterio as rs

import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import cmasher as cmr
from required_functions import *
from pathlib import Path
plt.rcParams.update({'figure.dpi': 600})

path_all = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity')
path_uea = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/uea_maps/UEA_20km')
path_ipl = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/ipl_maps/01_Data')
path_globiom = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/globiom_maps')
path_aim = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/aim_maps')
path_image = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/image_maps')
path_ig = Path('/Users/rpruetz/Documents/phd/primary/analyses/cdr_biodiversity/income')
path_ar6_data = Path('/Users/rpruetz/Documents/phd/datasets')

ar6_db = pd.read_csv(path_ar6_data / 'AR6_Scenarios_Database_World_v1.1.csv')
energy_crop_share = pd.read_csv(path_all / 'share_energy_crops_estimates.csv')
lookup_mi_cdr_df = pd.read_csv(path_all / 'lookup_table_ar_beccs_files_all_models.csv')
lookup_mi_cdr_df['year'] = lookup_mi_cdr_df['year'].astype(str)

# %% choose model to run the script with
model = 'GLOBIOM'  # options: 'GLOBIOM' or 'AIM' or 'IMAGE'

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

# %% resample land use to match IPL resolution
upscale_factor = 5  # equivalent to IPL resolution
land_infos = np.array(['Afforestation', 'BECCS'])

rcp_lvl = '34'  # select RCP level (without dot)
ssps = ['SSP1', 'SSP2', 'SSP3']

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

