# Workflow to accompany source water delineation for the Neversink-Rondout Basin, New York
This repository documents both the workflow and contains the dataset to repeat the model-building, data assimilation, and analysis in the two papers: 

Corson-Dosch, et al. _Areas contributing recharge to priority wells in valley-fill aquifers in the Neversink and Rondout Basins, New York_ in press as a USGS Scientific Investigations Report; and 

Fienen, et al. _Risk-based wellhead protection decision support: a repeatable workflow approach_ submitted to Groundwater

# Getting started
The workflow in this repository depends on a number of python packages. Static versions of the model-specific packages are included in the repository. However, a Conda environment must be installed to provide more general underlying packages.

To install, an environment file is provided in the root directory of the repository. Follow these steps:

* Install Anaconda python (if not already installed) from [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* In an Anaconda prompt on Windows, or in a terminal window on Mac or Linux, navigate to the top directory of this repository and type `conda env create -f environment.yml`
* This will create a `neversink` environment.
* Before running any notebooks or scripts, in an Anaconda prompt or terminal window, type `conda activate neversink` to activate the neversink environment.

The [Jupyter Notebook App](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html#notebook-app) can be opened from the Anaconda prompt or terminal window to run and explore the preprocessing and workflow notebooks. To launch the Jupyter Notebook App, follow these steps:
* Open an Anaconda prompt on Windows, or in a terminal window on Mac or Linux
* Navigate into this repository folder by typing `cd /path_to_repo_folder`. 
* Type `jupyter notebook` to launch the Jupyter Notebook App and the notebook interface will appear in a new browser window or tab. 

PEST++ and MODFLOW 6 executables are required to run the model locally. Executables for Windows, Mac, and Linux can be downloaded from these links: 
* [MODFLOW 6 (`mf6`)](https://github.com/MODFLOW-USGS/executables)
* [PEST++ (`pestpp-ies` and `pestpp-sen`)](https://github.com/usgs/pestpp/releases)

# Using the repository 
This repository is designed so that there are multiple paths a user can take though the workflow. Depending on user's objectives and level of interest, the workflow from the journal article and USGS report can be:
1. Replicated using supplied results, 
2. Rebuilt to generate local results, or 
3. Reviewed, as completed notebooks, without re-running any code. 

## Roadmap for each path 
Here we present the recommended order of operations for for each of the three paths through the workflow 
### 1. Replicated using supplied results

_Using this approach, the user can step through the  notebooks and apply the workflow described in the journal article and USGS report using supplied results. This approach does not require that MODFLOW or PEST be run locally._

1. (**Optional**) Run notebooks in `notebooks_preprocessing_blank` to generate MODFLOW  (notebooks starting with "0.") and PEST (notebooks starting with "1.") input data sets from source data. Complete preprocessing notebooks are aslo available to review in `notebooks_preprocessing_complete`
2. (**Optional**) Run `setup_model.py` in the `scripts` subdirectory to rebuild the modflow model from source data and processed data.
3. Run notebooks in `notebooks_workflow_blank` subdirectory in ascending order (start with 0.0, end with 4.2). Make sure the `run_ensemble` flag is set to `False` to use results supplied in the `output` subdirectory.  

### 2. Rebuild to generate local results
_Using this approach, the user can repeat the model building and data assimilation steps from scratch and step through the history matching workflow to generate results locally, running MODFLOW and PEST on their machine._

1. Run notebooks in `notebooks_preprocessing` to generate MODFLOW  (notebooks starting with "0.") and PEST (notebooks starting with "1.") input data sets from source data.
2. Run `setup_model.py` in the `scripts` subdirectory to rebuild the modflow model from source data and processed data.
3. Run notebooks in `notebooks_workflow_blank` subdirectory in ascending order (start with 0.0, end with 4.2). Make sure the `run_ensemble` flag is set to `True` to generate local results. 
4. The user will run the MODFLOW model locally using PESTPP-IES and PESTPP-SEN. Local results will be saved in the `run_data` subdirectory
5. Instructions to run the MODPATH Monte Carlo analysis (between workflow notebooks 4.1 and 4.2) are not included in this workflow. In the current setup, these workflow notebooks (4.0, 4.1 & 4.2) can only be run with supplied results (from `output/modpath`). The user could reproduce this step by running MODPATH once for each MODPATH zone (NE, W, and SE) and parameter realization in `modpath_par_ens.csv` (generated in workflow notebook `3.2_postproc_iES-LT_noise.ipynb`). In the paper, this was accomplished using the [HTCondor](https://research.cs.wisc.edu/htcondor/) run manager. HTCondor files used in this study are included in `output/modpath`  


### 3. Review completed notebooks
_The user can review completed Jupyter Notebooks that contain code, output, and text which describe the model building, data assimilation, and history matching workflows. This approach does not require running Notebooks, PESTPP, or MODFLOW._ 

1. Review completed preprocessing notebooks in the `notebooks_preprocessing_complete` subdirectory. 
2. Review completed workflow notebooks in the `notebooks_workflow_complete` subdirectory. These notebooks show the complete history-matching and MODPATH workflow.
3. MODFLOW and MODPATH input and output files can be reviewed in the `neversink_mf6` subdirectory.

# Description of repository subdirectories and top-level files
## /Figures
subdirectory containing figures generated during the neversink workflow
## /neversink_mf6
subdirectory containing MOFLOW files and subdirectories generated by `modflow-setup`. MODPATH files generated in the workflow are also saved here 
## /notebooks\_preprocessing\_blank
Blank preprocessing notebooks used to process source data for MODPATH and PEST. These notebooks can be run by users following path 1 or path 2 described in the **Using the repository** section above. 
## /notebooks\_preprocessing\_complete
Complete preprocessing notebooks used to process source data for MODPATH and PEST. These notebooks can be reviewed by users following path 3 described in the **Using the repository** section above. 
## /notebooks\_workflow\_blank
Blank workflow notebooks used for history matching and MODPATH. These notebooks can be run by users following path 1 or path 2 described in the **Using the repository** section above. 
## /notebooks\_workflow\_complete
Complete workflow notebooks sed for history matching and MODPATH. These notebooks can be reviewed by users following path 3 described in the **Using the repository** section above.
## /output
Results from the journal article, can be used to run workflow notebooks by users following path 2 described in the **Using the repository** section above.. 
## /processed\_data
MODFLOW and PEST data sets developed in `notebooks_preprocessing`, derrived from `source_data`.
## /python\_pacakges\_static
Static versions of the model-specific Python packages.
## /scripts
Python scripts used during model building, history matching, and MODPATH post-processing
## /sourcedata
Unmanipulated MODFLOW and PEST input and assimilation data sets. 
## environment.yml
[Conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that can be used to create the `neversink`  environment.
## neversink\_full.yml
A [modflow-setup](https://github.com/aleaf/modflow-setup) YAML format [configuration file](https://aleaf.github.io/modflow-setup/latest/config-file.html) used to build the neversink MODFLOW6 model.


