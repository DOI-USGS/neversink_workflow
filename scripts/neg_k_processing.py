import os
import numpy as np

ext_dir = '../neversink_mf6/'

# quick function to replace negative K values (from TIF sampling discrepancy)
# with max of surrouning 9-cell stencil
def replace_neg(idm, k):
    k[k == 0.] = -9999.
    neg_k = np.abs(idm) * k
    row, col = np.where(neg_k < -1)
    for r, c in zip(row, col):
        neg_k[r, c] = np.max(neg_k[r-1:r+2, c-1:c+2])
        
    return neg_k
    
    
if __name__ == "__main__": 
    for iteration in range(2): # added second iteration for cells that were not corrected by first pass, active cells more than 1 away from a k>0 cell
        for cl in range(4):
                idm = np.loadtxt(os.path.join(ext_dir, 'idomain_{0:03d}.dat'.format(cl)))
                print('idomain{}.dat'.format(cl))
                for k in ['k', 'k33']:
                    cf = os.path.join(ext_dir, '{0}{1}.dat'.format(k,cl))
                    print('updating {}'.format(cf))
                    k_data = np.loadtxt(cf)
                    k_updated = replace_neg(idm, k_data)
                    np.savetxt(cf, k_updated)
        
        
 
