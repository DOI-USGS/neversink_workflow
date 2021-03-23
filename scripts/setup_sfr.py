import os, sys
sys.path.append('../python_packages_static')
import json
import pandas as pd
import numpy as np
import flopy
import sfrmaker
import geopandas as gpd

setup_from_scratch = False
use_additional_flowlines = True  # option to use additional, manually-created sfr flowlines to reduce flooding in priority recharge areas

simname = 'mfsim.nam'
model_ws = '../neversink_mf6/'
nhd_dir = '../sciencebase/NHDPlus/'
processed_input = '../processed_data'

sim = flopy.mf6.MFSimulation.load(simname, 'mf6', sim_ws=model_ws)
m = sim.get_model()

delc = m.dis.delc.data
delr = m.dis.delr.data
top = m.dis.top.data
botm = m.dis.botm.data

with open (os.path.join(model_ws, 'neversink_grid.json')) as f:
    grid_info = json.load(f)

flopy_grid = flopy.discretization.structuredgrid.StructuredGrid(
    delc=delc,
    delr=delr,
    top=top,
    botm=botm,
    epsg=5070,
    xoff=grid_info['xoff'],
    yoff=grid_info['yoff'],
    lenuni=2
)
if setup_from_scratch:
    #  load NHD Flowlines
    rond_gdb = 'NHDPLUS_H_0202_HU4_GDB.gdb'
    nvsk_gdb = 'NHDPLUS_H_0204_HU4_GDB.gdb'

    buffer_extent = '../sciencebase/Shapefiles/Extents/Model_Extent_HUC12_2km_buffer.shp'

    # load flowlines
    print('loading Rondout NHDflowlines...')
    rond_flowlines = gpd.read_file(os.path.join(nhd_dir, rond_gdb), driver='FileGDB', layer='NHDFlowline')
    print('loading Neversink NHDflowlines...')
    nvsk_flowlines = gpd.read_file(os.path.join(nhd_dir, nvsk_gdb), driver='FileGDB', layer='NHDFlowline')

    # verify same crs before join
    assert nvsk_flowlines.crs == rond_flowlines.crs

    #  load buffer model area for clipping
    basin_buff = gpd.read_file(buffer_extent)
    basin_buff = basin_buff.to_crs(nvsk_flowlines.crs)

    # clip flowlines to basin + model + 2km
    print('clipping Rondout..')
    rond_flowlines_clip = gpd.clip(rond_flowlines, basin_buff)
    print('clipping Neversink...')
    nvsk_flowlines_clip = gpd.clip(nvsk_flowlines, basin_buff)

    rond_flowlines_clip.to_file(os.path.join(processed_input, 'rond_flowlines_clip.shp'))
    nvsk_flowlines_clip.to_file(os.path.join(processed_input, 'nvsk_flowlines_clip.shp'))

    # concat flowlines
    print('merging flowlines...')
    nsro_flowlines_clip = gpd.GeoDataFrame(pd.concat([rond_flowlines_clip, nvsk_flowlines_clip]), crs=nvsk_flowlines.crs)

    #  drop pipeline fcodes from flowlines

    drop_fcodes = [42803, 42814]
    nsro_flowlines_clip = nsro_flowlines_clip.loc[~nsro_flowlines_clip.FCode.isin(drop_fcodes)]

    # load flvaas
    print('loading neversink NHDPlusFlowlineVAA')
    nvsk_flvaa = gpd.read_file(os.path.join(nhd_dir, nvsk_gdb), driver='FileGDB', layer='NHDPlusFlowlineVAA')
    print('loading rondout NHDPlusFlowlineVAA')
    rond_flvaa = gpd.read_file(os.path.join(nhd_dir, rond_gdb), driver='FileGDB', layer='NHDPlusFlowlineVAA')

    # concat flvaas
    flvaa = gpd.GeoDataFrame(pd.concat([nvsk_flvaa, rond_flvaa]))

    # join to clipped, merged, flowlines
    nsro_flowlines_clip = nsro_flowlines_clip.merge(flvaa[['NHDPlusID', 'ArbolateSu','StreamOrde', 'MaxElevSmo', 'MinElevSmo', 'Divergence']], 
                    on='NHDPlusID', how='left'
                )

    # load plusflows
    print('loading neversink NHDPlusFlow')
    nvsk_pf = gpd.read_file(os.path.join(nhd_dir, nvsk_gdb), driver='FileGDB', layer='NHDPlusFlow')
    print('loading rondout NHDPlusFlow')
    rond_pf = gpd.read_file(os.path.join(nhd_dir, rond_gdb), driver='FileGDB', layer='NHDPlusFlow')

    # concat plusflows
    pf = pd.DataFrame(pd.concat([nvsk_pf, rond_pf]))
    pf = pf.merge(nsro_flowlines_clip[['Divergence', 'NHDPlusID']], left_on='ToNHDPID', right_on = 'NHDPlusID', how='outer')
    pf.rename(columns={'Divergence':'Divergence_ToNHDPID'}, inplace=True)

    #  create routing dict, excluding flowlines that have ToNHDPID Divergence == 2
    pf_dict = dict(zip(pf.loc[pf.Divergence_ToNHDPID != 2, 'FromNHDPID'], pf.loc[pf.Divergence_ToNHDPID != 2, 'ToNHDPID']))

    #  using routing dict to define ToNHDPID rounding values
    nsro_flowlines_clip['ToNHDPID'] = [pf_dict[i] for i in nsro_flowlines_clip.NHDPlusID]

    nsro_flowlines_final = nsro_flowlines_clip.copy()
    #  fill na's with zeros for sfrmaker
    nsro_flowlines_final['ToNHDPID'].fillna(0., inplace=True)

    cols = ['NHDPlusID', 'ToNHDPID']

    ## reduce length of NHDPlusIDs for SFRmaker
    print('reducing length of NHDPlusIDs for SFRmaker')
    for col in cols:
        vals = nsro_flowlines_final[col].values
        print(len(vals))
        updated_vals = []
        for val in vals:
            if val == 0.0:
                updated_vals.append(0)
            else:
                val = str(val)
                updated_vals.append(int(val[5:-2]))
        print(len(updated_vals))
        
        nsro_flowlines_final[col] = updated_vals

    #  write to shapefile
    nsro_flowlines_final = nsro_flowlines_final.to_crs(epsg=5070)
    nsro_flowlines_final.to_file(os.path.join(processed_input, 'nsro_flowlines.shp'))
    print('wrote {}'.format(os.path.join(processed_input, 'nsro_flowlines.shp')))

if use_additional_flowlines is True:
    print('adding additional flowlines to NHD...')
    #  read in addional flowlines
    add_flowlines = gpd.read_file(os.path.join(processed_input, 'additional_flowlines.shp'))
    add_flowlines['NHDPlusID'] = add_flowlines['NHDPlusID'].astype(np.int64) # adjust to int
    add_flowlines['ToNHDPID'] = add_flowlines['ToNHDPID'].astype(np.int64) # adjust to int

    #  merge additional flowlines with NHD and save to shapefile
    nsro_flowlines = gpd.read_file(os.path.join(processed_input, 'nsro_flowlines.shp'))
    add_flowlines = add_flowlines.to_crs(nsro_flowlines.crs)
    nsro_add_flowlines_final = gpd.GeoDataFrame(pd.concat([nsro_flowlines, add_flowlines]), crs=nsro_flowlines.crs)
    nsro_add_flowlines_final = nsro_add_flowlines_final.to_crs(epsg=5070)
    nsro_add_flowlines_final.to_file(os.path.join(processed_input, 'nsro_add_flowlines.shp'))

#  make sfr lines
print('building sfrmaker lines from shapefile...')

if use_additional_flowlines is True:
    lines = sfrmaker.Lines.from_shapefile(shapefile=os.path.join(processed_input, 'nsro_add_flowlines.shp'),
                                                id_column='NHDPlusID',  # arguments to sfrmaker.Lines.from_shapefile
                                                routing_column='ToNHDPID',
                                                arbolate_sum_column2='ArbolateSu',
                                                up_elevation_column='MaxElevSmo',
                                                dn_elevation_column='MinElevSmo',
                                                name_column='GNIS_Name',
                                                attr_length_units='meters',  # units of source data
                                                attr_height_units='meters',  # units of source data
                                                epsg=5070 
                                                )
if not use_additional_flowlines is True:
    lines = sfrmaker.Lines.from_shapefile(shapefile=os.path.join(processed_input, 'nsro_flowlines.shp'),# 'nsro_add_flowlines.shp'),
                                                id_column='NHDPlusID',  # arguments to sfrmaker.Lines.from_shapefile
                                                routing_column='ToNHDPID',
                                                arbolate_sum_column2='ArbolateSu',
                                                up_elevation_column='MaxElevSmo',
                                                dn_elevation_column='MinElevSmo',
                                                name_column='GNIS_Name',
                                                attr_length_units='meters',  # units of source data
                                                attr_height_units='meters',  # units of source data
                                                epsg=5070 
                                                )                            

#  make sfrdata instance using flopy grid, modflow model, and flopy grid
sfrdata = lines.to_sfr(grid=flopy_grid, model=m, model_length_units='meters', epsg=5070, consolidate_conductance=True, one_reach_per_cell=True)

print('setting streambed top elevations from dem...')
sfrdata.set_streambed_top_elevations_from_dem('../source_data/Shapefiles/top_50m_from_lidar.tif', dem_z_units='meters')

sfrdata.assign_layers('../neversink_mf6')
#sfrdata.run_diagnostics(verbose=False)

#  add sfr observations
print('adding sfr observations...')
sfrdata.add_observations('../processed_data/NWIS_DV_STREAMSTATS_SITES.csv',
                         obstype='downstream-flow',
                         x_location_column='x',
                         y_location_column='y',
                         obsname_column='site_id'                     
                        )

#  sfr inflows added in setup_model.py

#  write sfr files
sfrdata.write_package(external_files_path='.', version='mf6', )
