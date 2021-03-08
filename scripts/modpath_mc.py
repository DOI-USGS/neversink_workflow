import sys, os
import zipfile
import numpy as np
import pandas as pd
import pyemu
import flopy.utils as fu
from get_endpoints import get_endpoints

#  set path
run_dir = '.'

# get the run index from the command line
runindex = int(sys.argv[1])

#  read in modpath parameter ensemble, pst control file to modify for modpath
mp_par = pd.read_csv(os.path.join(run_dir, 'modpath_par_ens.csv'), index_col=0)
pst = pyemu.Pst(os.path.join(run_dir, 'prior_mc_wide.pst'))
pst.control_data.noptmax = 0

#  check that indicies are in same order
assert np.sum(pst.parameter_data.index == mp_par.T.index) == len(pst.parameter_data)

#  set parvals using runindex value and write modpath pest file
pst.parameter_data.parval1 = mp_par.iloc[runindex].T
pst.write(os.path.join(run_dir, 'modpath.pst'))

# run pest/modflow to get phi
runstring = './pestpp-ies modpath.pst'
print(runstring)
os.system(runstring)

# get water table array and save as txt file
h = fu.binaryfile.HeadFile(os.path.join(run_dir, 'neversink.hds')).get_data()
wt = fu.postprocessing.get_water_table(h, 1e+30)
np.savetxt('wt_array.txt', wt, fmt='%.4e')
print('wt_array.txt created')

# run modpath
mp_zone_files = ['neversink_mp_forward_weak_NE', 'neversink_mp_forward_weak_W', 'neversink_mp_forward_weak_S']

for zone in mp_zone_files:
    runstring = './mp7 {}.mpsim'.format(zone)
    print(runstring)
    os.system(runstring)

    # get endpoints
    get_endpoints('{}.mpend'.format(zone), zone[26:])

#  zip up results
with zipfile.ZipFile('mp_results_{}.zip'.format(runindex), 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('modpath.phi.actual.csv')
    zf.write('endpoint_cells_NE.csv')
    zf.write('endpoint_cells_W.csv')
    zf.write('endpoint_cells_S.csv')
    zf.write('wt_array.txt')