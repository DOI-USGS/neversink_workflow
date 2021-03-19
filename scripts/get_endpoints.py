import os, sys
sys.path.append('../python_packages_static')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import flopy as fp
import flopy.utils as fu

def get_endpoints(mpend_file, region):
    # set paths
    model_ws = os.getcwd()
    simname = 'mfsim.nam'

    # read DIS information
    sim = fp.mf6.MFSimulation.load(simname, 'mf6', sim_ws=model_ws, load_only='DIS')
    m = sim.get_model()

    # read endpoints from dest cells
    dest_nodes = [1042614, 214382, 562588, 574351, 575587, 574968, 1041449, 344623]
                

    epth = os.path.join(model_ws, mpend_file)
    e = fu.EndpointFile(epth)
    epd = e.get_destination_endpoint_data(dest_cells=dest_nodes)
    df = pd.DataFrame.from_records(epd)

    # compute i, j cells of endpoints
    df['i'] = (m.dis.nrow.data - (df['y0'].values - 25) / 50).astype(int)
    df['j'] = ((df['x0'] - 25) / 50).astype(int)

    # save csv of node, i cell, j cell
    df[['node','i', 'j']].to_csv('endpoint_cells_{}.csv'.format(region), index=False)

if __name__ == '__main__':
    mpend_file = sys.argv[1]
    region = sys.argv[2]
    get_endpoints(mpend_file, region)
