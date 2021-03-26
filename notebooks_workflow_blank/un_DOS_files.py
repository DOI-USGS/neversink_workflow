import os

def un_DOS_files(modeldir='../mfsetup/LIRM_SS_TMR'):
    for root, dirs, files in os.walk(modeldir):
        for f in files:
            if (not f.startswith('.')) & ('postproc' not in root) & \
                    ('sfr' not in root) & ('.exe' not in f) & ('grb' not in f) & ('.shp' not in f) &\
                    ('shx' not in f) & ('dbf' not in f) & ('log' not in f) & ('bin' not in f) & \
                    ('hds' not in f) & ('mppth' not in f) & ('timeseries' not in f) & \
                    ('mpsim' not in f) & ('mp' not in f) & ('cbc' not in f):
                infilename = os.path.join(root,f)
                infile = open(infilename, 'r').readlines()
                print (infilename)
                with open(infilename, 'w') as ofp:
                    [ofp.write(line.rstrip().replace('\\','/') + '\n') for line in infile]

if __name__ == '__main__':
    un_DOS_files()
