import os
import glob
for cf in glob.glob('*.ipynb'):	
    print('jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {}'.format(cf))
    os.system('jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {}'.format(cf))

