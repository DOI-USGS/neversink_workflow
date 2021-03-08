import os
import flopy
import flopy.utils as fu
mf6 = flopy.mf6
from mfsetup import MF6model
from mfsetup.fileio import load, dump
from mfsetup.utils import get_input_arguments
import numpy as np
import geopandas as gp
from shapely.geometry import Point
import pandas as pd


def setup_model():
    m = MF6model.setup_from_yaml('../neversink_full.yml')
    m.write_input()
    # write the SFR package again using SFRmaker,
    # because flopy doesn't check the packagedata against the idomain before writing
    if hasattr(m, 'sfr'):
        sfr_package_filename = os.path.join(m.model_ws, m.sfr.filename)
        m.sfrdata.write_package(sfr_package_filename,
            idomain=m.idomain,
            version='mf6',
            options=['save_flows',
                    'BUDGET FILEOUT {}.sfr.cbc'.format(m.name),
                    'STAGE FILEOUT {}.sfr.stage.bin'.format(m.name),
            		],
            external_files_path=m.external_path
            )
        
    return m

def reassign_well_pumping():
    # load up well pumping and metadata
    welldata = pd.read_csv('../neversink_mf6/wel_000.dat', delim_whitespace=True)
    metadata = pd.read_csv('../processed_input/2009-2016_ave_pumping.csv')
    welldata=welldata.merge(metadata[['ID_Well','Comments']], left_on = 'boundname', right_on="ID_Well")
    
    # pulling the layering info from the metadata comments which indicate layering in the following cases
    wellid = ['WWR0000401_Davos Well #3', 'WWR0000401_Riverside Well #3',
       'WWR0000506_Fallsburg Well #3', 'WWR0000506_Fallsburg Well #6',
       'WWR0000506_Fallsburg Well #7', 'WWR0001055_Mountaindale Well #2',
       'WWR0001734_52912000-1', 'WWR0001738_Well #1', 'WWR0001748_Well 3']
    layer = [3,1,3,3,1,3,1,1,1]
    updates = dict(zip(wellid,layer))
    
    # update the layers using the info from above
    for w,l in updates.items():
        welldata.loc[welldata.boundname==w,'#k'] = l
    
    # move a couple specific wells that need adjusting because they are in streams
    #welldata.loc[welldata.boundname=='WWR0000479_Center St. Well', 'j'] += 1
    welldata.loc[welldata.boundname=='WWR0001748_Well 3', 'j'] -= 1
    welldata.loc[welldata.boundname=='WWR0000506_Fallsburg Well #7', 'j'] += 1
    welldata.loc[welldata.boundname=='WWR0000506_Woodbourne Well #1A', 'j'] += 1
    welldata.loc[welldata.boundname=='WWR0000506_Woodbourne Well #2', 'j'] -= 1
    welldata.loc[welldata.boundname=='WWR0001734_52912000-6', 'j'] -= 1
    welldata.loc[welldata.boundname=='WWR0001738_Well #1', 'j'] += 1
    welldata.loc[welldata.boundname=='WWR0001738_Well #2', 'j'] += 1
    
    # and this annoying well prevents convergence of the whole model. But.....it's up on a knob so q->0
    welldata.loc[welldata.boundname=='WWR0001734_52912000-3','q']=0
    #  this well too
    welldata.loc[welldata.boundname=='WWR0001734_52912000-7','q']=0
    # well, while we are at it, this one as well
    welldata.loc[welldata.boundname=='WWR0001738_Well #3','q']=0


    # save out the results
    welldata[['#k','i','j','q', 'boundname']].to_csv('../neversink_mf6/wel_000.dat', sep=' ', index=None )

def trim_CHDs(m):
    
    # read in the valley extent shapefile
    val = gp.read_file('../sciencebase/Shapefiles/Extents/Valley_Extent_extended.shp')
    # set up a spatial reference object to be able to locate the CHD cells to intersect with the valley
    xul = 1742955.0
    yll = 2258285.0
    sr = fu.SpatialReference(
        xul=xul,
        yll=yll, 
        delr=m.dis.delr.array, 
        delc=m.dis.delc.array
    )
    X,Y = sr.get_xcenter_array(),sr.get_ycenter_array()
    # why doesn't sr get the offsets right?
    X+=xul
    Y+=yll
    # read the chd cells from the model object m
    chd_cells = pd.DataFrame.from_records(m.chd.stress_period_data.data[0])
    # strip out the k,i,j from cell ids
    chd_cells['#k'] = [ccel[0] for ccel in chd_cells.cellid]
    chd_cells['i'] = [ccel[1] for ccel in chd_cells.cellid]
    chd_cells['j'] = [ccel[2] for ccel in chd_cells.cellid]
    # set up geometry for the intersection
    chd_cells['x'] = X[chd_cells.j]
    chd_cells['y'] = Y[chd_cells.i]
    chd_cells['geometry'] = [Point(x,y) for x,y in zip(chd_cells.x,
                                                    chd_cells.y)]
    # intersect each CHD cell with the valleys
    chd_cells['in_valley'] = [i.within(val.geometry[0]) for i in chd_cells.geometry]
    # trim to only cells in the valleys
    chd_cells=chd_cells.loc[chd_cells.in_valley]
    # special location of the boundary near the neversink reservoir dam
    chd_dam = chd_cells.loc[(chd_cells.i<300) & (chd_cells.j<300)]  
    # find the bottom of the layer the CHD cells are in for the dam cells and set CHD to bot+1m 
    chd_dam['bot'] = [m.dis.botm.array[k][i,j] for 
                      k,i,j in zip(chd_dam['#k'],chd_dam['i'],chd_dam['j'])]
    chd_cells.loc[(chd_cells.i<300) & (chd_cells.j<300), 'head'] = chd_dam.bot+1

    # get back to one-based to write out
    chd_cells['#k'] += 1
    chd_cells['i'] += 1
    chd_cells['j'] += 1

    # save the CHD file back out again
    chd_cells[['#k','i','j','head']].to_csv('../neversink_mf6/chd_000.dat',
                                       sep=' ',
                                       index=None)


def add_inflows():
    df = pd.read_csv('../neversink_mf6/neversink_packagedata.dat', delim_whitespace=True)
    rno = int(df.loc[df['line_id'] == 200105086].iloc[0]['#rno']
)
    pdata = ['', 'BEGIN PERIOD 1', '# rno sfrsetting', '  {} inflow 48932'.format(rno), 'END PERIOD']
    cont = [i.rstrip() for i in open('../neversink_mf6/neversink.sfr', 'r').readlines()]

    if 'inflow' not in ' '.join(cont):
        print('adding inflow information to SFR file')
        cont.extend(pdata)

    with open('../neversink_mf6/neversink.sfr', 'w') as ofp:
        [ofp.write('{}\n'.format(line)) for line in cont]    

def adjust_model_botm():
    inlines = open('../neversink_mf6/neversink.dis', 'r').readlines()
    with open('../neversink_mf6/neversink.dis', 'w') as ofp:
        [ofp.write(line.replace('botm_003.dat', 'neversink_layer_5_new_botm_elevations.dat')) for line in inlines]
    
def setup_npf():
    model = MF6model.load('../neversink_full.yml')
    model.setup_npf()

def adjust_sfr_paths():

    sfr_output_files = ['neversink.sfr.cbc', 'neversink.sfr.stage.bin', 'neversink.sfr.obs']

    inlines = open('../neversink_mf6/neversink.sfr', 'r').readlines()
    with open('../neversink_mf6/neversink.sfr', 'w') as ofp:
        for line in inlines:
            outfile_present = False
            for outfile in sfr_output_files:
                if outfile in line:
                    prepend_words = line.split()[:2]
                    ofp.write('{} {} {}\n'.format(prepend_words[0], prepend_words[1],outfile))
                    outfile_present = True
            if outfile_present is False:
                ofp.write(line)

def add_sfr_to_nam_file():
    namfile = open('../neversink_mf6/neversink.nam', 'r').readlines()
    with open('../neversink_mf6/neversink.nam', 'w') as ofp:
        for line in namfile:
            if 'end packages' not in line.lower():
                ofp.write(line)
            else:
                ofp.write('  SFR6  neversink.sfr  sfr_0\n')
                ofp.write(line)
          


if __name__ == '__main__':
    setup_all = True
    setup_npf_only = False

    if setup_all:
        stdir = os.getcwd()
        m = setup_model()
        os.chdir(stdir)
        os.system('python setup_sfr.py')
        add_inflows()
        adjust_model_botm()
        adjust_sfr_paths()
        os.system('python neg_k_processing.py')
        
        # call function to put wells in correct layers
        reassign_well_pumping()
    
        # also trim CHDs to only be on the boundaries
        trim_CHDs(m)

        # need to add the SFR file to the model namefile since built externally
        add_sfr_to_nam_file()
        
    if setup_npf_only:
        print('Setting up NPF only...')
        stdir = os.getcwd()
        setup_npf()
        os.chdir(stdir)
        os.system('python neg_k_processing.py')
    

    
