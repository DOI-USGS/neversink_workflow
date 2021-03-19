import sys
sys.path.append('../python_packages_static')
import pandas as pd
import os
import numpy as np
import flopy as fp
import flopy.utils as fu

# define a MF6 capable T calculation function (thanks Andy!)
def get_transmissivities(heads, hk, top, botm,
                         r=None, c=None, x=None, y=None, modelgrid=None,
                         sctop=None, scbot=None, nodata=-999):
    """
    Computes transmissivity in each model layer at specified locations and
    open intervals. A saturated thickness is determined for each row, column
    or x, y location supplied, based on the open interval (sctop, scbot),
    if supplied, otherwise the layer tops and bottoms and the water table
    are used.

    Parameters
    ----------
    heads : 2D array OR 3D array
        numpy array of shape nlay by n locations (2D) OR complete heads array
        of the model for one time (3D)
    hk : 3D numpy array
        horizontal hydraulic conductivity values.
    top : 2D numpy array
        model top elevations.
    botm : 3D numpy array
        layer botm elevations.
    r : 1D array-like of ints, of length n locations
        row indices (optional; alternately specify x, y)
    c : 1D array-like of ints, of length n locations
        column indices (optional; alternately specify x, y)
    x : 1D array-like of floats, of length n locations
        x locations in real world coordinates (optional).
        If x and y are specified, a modelgrid must also be provided.
    y : 1D array-like of floats, of length n locations
        y locations in real world coordinates (optional)
        If x and y are specified, a modelgrid must also be provided.
    modelgrid_transform : affine.Affine instance, optional
        Only required for getting i, j if x and y are specified.
    sctop : 1D array-like of floats, of length n locations
        open interval tops (optional; default is model top)
    scbot : 1D array-like of floats, of length n locations
        open interval bottoms (optional; default is model bottom)
    nodata : numeric
        optional; locations where heads=nodata will be assigned T=0

    Returns
    -------
    T : 2D array of same shape as heads (nlay x n locations)
        Transmissivities in each layer at each location

    """
    
    
    if r is not None and c is not None:
        pass
    elif x is not None and y is not None:
        # get row, col for observation locations
        r, c = get_ij(modelgrid, x, y)
    else:
        raise ValueError('Must specify row, column or x, y locations.')

    # get k-values and botms at those locations
    # (make nlayer x n sites arrays)
    hk2d = hk[:, r, c]
    botm2d = botm[:, r, c]

    if len(heads.shape) == 3:
        heads = heads[:, r, c]

    msg = 'Shape of heads array must be nlay x nhyd'
    assert heads.shape == botm2d.shape, msg

    # set open interval tops/bottoms to model top/bottom if None
    if sctop is None:
        sctop = top[r, c]
    if scbot is None:
        scbot = botm[-1, r, c]

    # make an nlayers x n sites array of layer tops
    tops = np.empty_like(botm2d, dtype=float)
    tops[0, :] = top[r, c]
    tops[1:, :] = botm2d[:-1]

    # expand top and bottom arrays to be same shape as botm, thickness, etc.
    # (so we have an open interval value for each layer)
    sctoparr = np.zeros(botm2d.shape)
    sctoparr[:] = sctop
    scbotarr = np.zeros(botm2d.shape)
    scbotarr[:] = scbot

    # start with layer tops
    # set tops above heads to heads
    # set tops above screen top to screen top
    # (we only care about the saturated open interval)
    openinvtop = tops.copy()
    openinvtop[openinvtop > heads] = heads[openinvtop > heads]
    openinvtop[openinvtop > sctoparr] = sctoparr[openinvtop > sctop]

    # start with layer bottoms
    # set bottoms below screened interval to screened interval bottom
    # set screen bottoms below bottoms to layer bottoms
    openinvbotm = botm2d.copy()
    openinvbotm[openinvbotm < scbotarr] = scbotarr[openinvbotm < scbot]
    openinvbotm[scbotarr < botm2d] = botm2d[scbotarr < botm2d]

    # compute thickness of open interval in each layer
    thick = openinvtop - openinvbotm

    # assign open intervals above or below model to closest cell in column
    not_in_layer = np.sum(thick < 0, axis=0)
    not_in_any_layer = not_in_layer == thick.shape[0]
    for i, n in enumerate(not_in_any_layer):
        if n:
            closest = np.argmax(thick[:, i])
            thick[closest, i] = 1.
    thick[thick < 0] = 0
    thick[heads == nodata] = 0  # exclude nodata cells
    thick[np.isnan(heads)] = 0  # exclude cells with no head value (inactive cells)

    # compute transmissivities
    T = thick * hk2d
    return T
    
def get_heads(heads, r, c):
    """
    Extracts simulated head values at specified row, column.

    Parameters
    ----------
    heads : 2D array OR 3D array
        numpy array of shape nlay by n locations (2D) OR complete heads array
        of the model for one time (3D)
    r : 1D array-like of ints, of length n locations
        row indices 
    c : 1D array-like of ints, of length n locations
        column indices 

    Returns
    -------
    hds: 2D array of same shape as heads (nlay x n locations)
         in each layer at each location

    """
    if r is not None and c is not None:
        pass
    else:
        raise ValueError('Must specify row, column locations.')

    if len(heads.shape) == 3:
        hds = heads[:, r, c]
        
    else:
        hds = heads[r, c]

    return hds
    
#
# set some path information and flags NB - output_files order is important to maintain
#

#files
output_files = ['neversink.sfr.obs.output.csv', 'neversink.head.obs']
wkdir = '../neversink_mf6'
if len(sys.argv)>1:
    wkdir=sys.argv[1]
    if sys.argv[2].lower() == 'true':
        make_ins = True
    else:
        make_ins = False
else:
    wkdir = '../run_data'
    make_ins = True
obsfile = 'neversink.obs'

#flags


# load up dis and npf for layering and K for T calcs later
sim = fp.mf6.MFSimulation.load(sim_ws=wkdir, load_only=['DIS','npf'])
m = sim.get_model()

# grab the model info needed for T calcs
h = fu.binaryfile.HeadFile(os.path.join(wkdir, 'neversink.hds')).get_data()
hk = m.npf.k.array
top = m.dis.top.array
botm = m.dis.botm.array

# make a list of dfs for the model output files and read them in 
df_list = []

for file in output_files:
    df_list.append(pd.read_csv(os.path.join(wkdir, file)))


#
# process SFR observations first
#
sfr_out = df_list[0].T.drop('time').copy()
sfr_out.index = ['q_{}'.format(i) for i in sfr_out.index]
sfr_out.columns = ['obs']
# let's flip this to positive
sfr_out.obs *= -1
if make_ins:
    sfr_out['insline'] = ['l1 w !{}!'.format(i) for i in sfr_out.index]

#
# process heads with T weighting
#
df = df_list[1].T.drop('time').copy()
df.columns = ['head_mod']
# cast index to lower to make sure lining up with metadata capitalization
df.index = [i.lower() for i in df.index]

# pull out rootname from repeated multi-layer obs names
df['rootname'] = ['h_{}'.format(i.split('.')[0]) if '.' in i else 'h_{}'.format(i) for i in df.index]

# read in the obs file for metadata on heads

# navigate to make sure not reading in stuff we don't need
obsmeta = [i.strip().lower() for i in open(os.path.join(wkdir,obsfile), 'r').readlines()]
strow = [idx for idx,val in enumerate(obsmeta) if 'begin continuous' in val][0]
endrow = [idx for idx,val in enumerate(obsmeta) if 'end continuous' in val][0]

# then read in the metadata, and flip to zero-based indexing 
obsmetadata = pd.read_csv(os.path.join(wkdir,obsfile), skiprows=strow+1,
                          delim_whitespace=True,skipfooter=len(obsmeta)-endrow, engine='python',
                         names=['type','lay','row','col'], header=None)
obsmetadata['lay'] -= 1
obsmetadata['row'] -= 1
obsmetadata['col'] -= 1
# index lower like with obs output
obsmetadata.index = ['h_{}'.format(i.lower()) for i in obsmetadata.index]

#Make sure the metadata is lined up with the obs output
assert np.sum(obsmetadata.index.values == df.rootname.values) == len(df)

# then populate the heads df with lay,row,col
df['lay'] = obsmetadata.lay.values
df['row'] = obsmetadata.row.values
df['col'] = obsmetadata.col.values

# extract heads if from modflow head array if model did not converge
if df.head_mod.mean() == 0:
    print('model did not converge - extracting head observations from modflow head array')
    #df['extracted_head'] = np.nan
    for cn,cg in df.groupby('rootname'):
        heads = get_heads(heads=h,r=[cg.row.unique()],c=[cg.col.unique()])
        for clay in cg.lay:
            curr_head = np.squeeze(heads)[clay]
            df.loc[(df.rootname==cn) & (df.lay==clay), 'head_mod'] = curr_head
else:
    print('using heads from {}'.format(output_files[1]))

# calcualte T
print(':::getting Transmissivity values:::')
df['T'] = np.nan
df['T_frac'] = np.nan
for cn,cg in df.groupby('rootname'):
    T = get_transmissivities(h, hk, top, botm,r=[cg.row.unique()],c=[cg.col.unique()])
    T_sum = np.sum(np.squeeze(T))
    for clay in cg.lay:
        curr_T = np.squeeze(T)[clay]
        df.loc[(df.rootname==cn) & (df.lay==clay), 'T'] = curr_T
        df.loc[(df.rootname==cn) & (df.lay==clay), 'T_frac'] = curr_T/T_sum

# use T fraction and head to get fractional weighted head 
df['T_weighted_head'] = df.head_mod * df['T_frac']

#df['T_weighted_head'] = df['T_weighted_head'].astype(float)

# sum up the fractional results
head_out = df.groupby('rootname').sum()['T_weighted_head'].to_frame()
head_out = head_out.rename(columns = {'T_weighted_head':'obs'})
if make_ins:
    head_out['insline'] = ['l1 w !{}!'.format(i) for i in head_out.index]

# read in and report percent discrepancy
df_flux, _ = fu.Mf6ListBudget(os.path.join(wkdir,'neversink.list')).get_dataframes()
percdisc = df_flux.PERCENT_DISCREPANCY.values[0]
budget_df = pd.DataFrame({'obs':[percdisc]})
budget_df.index = ['PERC_DISC']
if make_ins:
    budget_df['insline'] = ['l1 w !{}!'.format(i) for i in budget_df.index]

#  read in land surface elevations
print('processing land surface observations')
ls_inds = pd.read_csv(os.path.join(wkdir, 'land_surf_obs-indices.csv'))
ls_inds['obs'] = h[ls_inds.k, ls_inds.i, ls_inds.j]
ls_out = ls_inds[['obs']]
ls_out.index = ls_inds.obsname 

if make_ins:
    ls_out['insline'] = ['l1 w !{}!'.format(i) for i in ls_out.index]

    
obs_out = pd.concat([sfr_out,head_out, budget_df, ls_out])

print('Writing output file to obs_mf6.dat')
# write out the results
obs_out['obs'].to_csv(os.path.join(wkdir,'obs_mf6.dat'), sep=' ')

# write out the instruction file if requested
if make_ins:
    print('Writing instruction file to obs_mf6.dat.ins')
    with open(os.path.join(wkdir,'obs_mf6.dat.ins'), 'w', newline='\n') as ofp:
        ofp.write('pif ~\n~obs~\n')
        obs_out.insline.to_csv(ofp, index=None, header=None)



