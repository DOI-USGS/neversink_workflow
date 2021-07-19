# Workflow to accompany source water delineation for the Neversink-Rondout Basin, New York
This repository documents both the workflow and contains the dataset to repeat the model-building, data assimilation, and analysis in the two papers: 

Corson-Dosch, N.T., Fienen, M.N., Finklestein, J.S., Leaf, A.T., White, J.T., Woda, J. and Williams, J.H (2021) _Areas contributing recharge to priority wells in valley-fill aquifers in the Neversink and Rondout Basins, New York_ in press as a USGS Scientific Investigations Report; and 

Fienen, M.N., Corson-Dosch, N.T., White, J.T., Leaf, A.T., and Hunt, R.J. (2021) _Risk-based wellhead protection decision support: a repeatable workflow approach_ submitted to Groundwater
# MODEL ARCHIVE

Archive created: 2021-05-20

# DISCLAIMER
                                                                          
  THE FILES CONTAINED HEREIN ARE PROVIDED AS AN ARCHIVE OF THE GROUNDWATER FLOW
  AND PARTICLE TRACKING SIMULATIONS DESCRIBED IN PUBLICATIONS BY 
  CORSON-DOSCH AND OTHERS (2021) AND FIENEN AND OTHERS (2021), LISTED ABOVE.
  THE FILES ARE ALSO PROVIDED AND AS A CONVENIENCE TO THOSE WHO WISH TO 
  REPLICATE SIMULATIONS AND REPEAT THE MODELING WORKFLOW. ANY CHANGES MADE TO 
  THESE FILES COULD HAVE UNINTENDED, UNDESIRABLE CONSEQUENCES. THESE 
  CONSEQUENCES COULD INCLUDE, BUT MAY NOT BE NOT LIMITED TO: ERRONEOUS MODEL 
  OUTPUT, NUMERICAL INSTABILITIES, AND VIOLATIONS OF UNDERLYING ASSUMPTIONS 
  ABOUT THE SUBJECT HYDROLOGIC SYSTEM THAT ARE INHERENT IN RESULTS PRESENTED IN 
  THE ASSOCIATED INTERPRETIVE REPORTS. THE U.S. GEOLOGICAL SURVEY ASSUMES NO 
  RESPONSIBILITY FOR THE CONSEQUENCES OF ANY CHANGES MADE TO THESE FILES.  IF 
  CHANGES ARE MADE TO THE MODEL, THE USER IS RESPONSIBLE FOR DOCUMENTING THE 
  CHANGES AND JUSTIFYING THE RESULTS AND CONCLUSIONS.     

--------------------------------------------------------------------------
# Getting started
The workflow in this repository depends on a number of python packages. Static versions of the model-specific packages are included in the repository. However, a Conda environment must be installed to provide more general underlying packages.

To install, an environment file is provided in the root directory of the repository. Follow these steps:

* Install Anaconda python (if not already installed) from [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* In an Anaconda prompt on Windows, or in a terminal window on Mac or Linux, navigate to the top directory of this repository and type `conda env create -f environment.yml`
* This will create a `neversink` environment.
* Before running any notebooks or scripts, in an Anaconda prompt or terminal window, type `conda activate neversink` to activate the neversink environment.

The [Jupyter Notebook App](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html#notebook-app) can be opened from the Anaconda prompt or terminal window to run and explore the preprocessing and workflow notebooks. To launch the Jupyter Notebook App, follow these steps:
* Open an Anaconda prompt on Windows, or in a terminal window on Mac or Linux
* Navigate into this repository folder by typing `cd /<path_to_repo_folder>`. 
* Type `jupyter notebook` to launch the Jupyter Notebook App and the notebook interface will appear in a new browser window or tab. 

PEST++ and MODFLOW 6 executables are required to run the model locally and must be copied into each run directory or, alternatively, added to the system path. Executables for Windows, Mac, and Linux can be downloaded from these links: 
* [MODFLOW 6 (`mf6`) and MODPATH 7 (`mp7`)](https://github.com/MODFLOW-USGS/executables)
* [PEST++ (`pestpp-ies` and `pestpp-sen`)](https://github.com/usgs/pestpp/releases)
Current versions are also included in `bin.zip`. 

# Using this archive 
This repository is designed so that there are multiple paths a user can take though the model and the workflow. Depending on the user's objectives and level of interest, the model and workflow from the journal article and USGS report can be accessed and used in the following ways:

1. Running and evaluating a single MODFLOW and/or MDOPATH simulation with a representative set of model parameters resulting from history matching
2. Replicating the workflow (pre- and post-processing) using supplied model and history matching results 
3. Rebuilding the model from data sources to generate local results and run the pre- and post-processing and all models locally
4. Reviewing completed notebooks, populated with complete ensemble results as generated for the article and report without re-running any code. 

## Roadmap for each path 
Here we present the recommended order of operations for each of the four paths through the workflow 

### 1. Running and evaluating a a single MODFLOW and/or MDOPATH simulation with a representative set of model parameters resulting from history matching  
_This approach is intended for users interested in a single, representative realization of model parameters and the associated smulation results. Optimal MODFLOW6 and MODFLOW7 files were created using the "base" parameters from the optimal (posterior) parameter ensemble from the PESTPP-IES history matching analysis. These optimal model files are supplied in the `neversink_optimal` subdirectory and can be run to replcate results supplied in the `output` subdirectory_  
1. Inspect the optimal MODFLOW6 and MODPATH7 files supplied in the `neversink_optimal` subdirectory.
2. (**Optional -- run MODFLOW6**) Open an Anaconda or Command prompt (if using Windows) or a terminal window (if using Mac or Linux) inside the `neversink_optimal` subdirectory. If MODFLOW6 is in the system path, simpy enter `mf6` to run the optimal MODFLOW6 simulation. If MODFLOW6 is not in the system path, use the MODFLOW6 executable supplied in the `bin` subdirectory for the appropreate operating system.
3. (**Optional -- run MODPATH7. Note: Step 2 must be completed first**) After MODFLOW6 has sucessfuly run (step 2), open an Anaconda or Command prompt (if using Windows) or a termanl window (if using Mac or Linux) inside the `neversink_optimal` subdirectory.
If MODPATH7 is in the system path, simpy enter `mp7` to start a MODPATH7 simulation. If MODPATH7 is not in the system path, use the MODPATH7 executable supplied in the `bin` subdirectory for the appropreate operating system. Three seperate MODPATH7 simulations will need to be run - one for each of the NE, S and W zones. For each run, use one of the three MODPATH simulation files provided in `neversink_optimal` (these files have an `.mpsim` file suffix. For example, the simulation file for the NE area is `neversink_mp_forward_weak_NE.mpsim`).
4. Output from the model runs are available in the `neversink_mf6_optimal_output` and `neversink_modpath_optimal_output` folders in the `output` folder for the MODFLOW 6 and MODPATH 7 model runs, respectively. These output results are provided for comparison with runs performed in the `neversink_optimal` folder (completed during optinal steps 2 and 3).

### 2. Replicated using supplied results

_Using this approach, the user can step through the  notebooks and apply the workflow described in the journal article and USGS report using supplied results. This approach does not require that MODFLOW or PEST be run locally._

1. (**Optional**) Run notebooks in `notebooks_preprocessing_blank` to generate MODFLOW  (notebooks starting with "0.") and PEST (notebooks starting with "1.") input data sets from source data. Complete preprocessing notebooks are also available to review in `notebooks_preprocessing_complete`
2. (**Optional**) Open an Anaconda prompt (if using Windows) or a termanl window (if using Mac or Linux), navigate into the `/scripts` subdirectory (by typing `cd scripts`), and activate the Conda neversink environment (by entering `activate neversink`). Run `setup_model.py` in the `scripts` subdirectory (by typing `python setup_model.py`) to rebuild the modflow model from source data and processed data.
3. Run the Jupyter Notebooks in the `notebooks_workflow_blank` subdirectory in ascending order (start with 0.0, end with 4.2). Make sure the `run_ensemble` flag is set to `False` to use results supplied in the `output` subdirectory.  

### 3. Replicating the workflow (pre- and post-processing) using supplied model and history matching results
_Using this approach, the user can repeat the model building and data assimilation steps from scratch and step through the history matching workflow to generate results locally, running MODFLOW and PEST on their machine._

1. Run notebooks in `notebooks_preprocessing` to generate MODFLOW  (notebooks starting with "0.") and PEST (notebooks starting with "1.") input data sets from source data.
2. Run `setup_model.py` in the `scripts` subdirectory to rebuild the modflow model from source data and processed data.
3. Run notebooks in `notebooks_workflow_blank` subdirectory in ascending order (start with 0.0, end with 4.2). Make sure the `run_ensemble` flag is set to `True` to generate local results. 
4. The user will run the MODFLOW model locally using PESTPP-IES and PESTPP-SEN. Local results will be saved in the `run_data` subdirectory
5. Instructions to run the MODPATH Monte Carlo analysis (between workflow notebooks 4.1 and 4.2) are not included in this workflow. In the current setup, these workflow notebooks (4.0, 4.1 and 4.2) can only be run with supplied results (from `output/modpath_mc`). The user could reproduce this step by running MODPATH once for each MODPATH zone (NE, W, and SE) and parameter realization in `modpath_par_ens.csv` (generated in workflow notebook `3.2_postproc_iES-LT_noise.ipynb`). In the paper, this was accomplished using the [HTCondor](https://research.cs.wisc.edu/htcondor/) run manager. 


### 4. Reviewing completed notebooks, populated with complete ensemble results as generated for the article and report without re-running any code. 
_The user can review completed Jupyter Notebooks that contain code, output, and text which describe the model building, data assimilation, and history matching workflows. This approach does not require running Notebooks, PESTPP, or MODFLOW._ 

1. Review completed preprocessing notebooks in the `notebooks_preprocessing_complete` subdirectory. 
2. Review completed workflow notebooks in the `notebooks_workflow_complete` subdirectory. These notebooks show the complete history-matching and MODPATH workflow.
3. MODFLOW and MODPATH input and output files can be reviewed in the `neversink_mf6` subdirectory.

# Description of subdirectories and top-level files
Each of the following directories can be uncompressed from a zip archive file with the same base name (e.g. `bin.zip` becomes `bin`).
## /bin
Executable files for running the models: 
 * MODFLOW 6 version 6.2.1
 * MODPATH 7 version 7.2001
 * PEST++ version 5.0.10


## /figures
Directory containing figures generated during the neversink workflow
## /neversink_mf6
Directory containing MOFLOW files and subdirectories generated by `modflow-setup`. MODPATH files generated in the workflow are also saved here 

## /neversink_optimal
MODFLOW and MODPATH files with properties assigned from the optimal "base" iteration of history matching. These can be considered as representative "best" model files. Output files are provided in the `/output` directory for comparison with the results written to this folder.

## /notebooks\_preprocessing\_blank
Blank preprocessing notebooks used to process source data for MODPATH and PEST. These notebooks can be run by users following path 1 or path 2 described in the **Using the repository** section above. 
## /notebooks\_preprocessing\_complete
Complete preprocessing notebooks used to process source data for MODPATH and PEST. These notebooks can be reviewed by users following path 3 described in the **Using the repository** section above. 
## /notebooks\_workflow\_blank
Blank workflow notebooks used for history matching and MODPATH. These notebooks can be run by users following path 1 or path 2 described in the **Using the repository** section above. 
## /notebooks\_workflow\_complete
Complete workflow notebooks sed for history matching and MODPATH. These notebooks can be reviewed by users following path 3 described in the **Using the repository** section above.
## /output
Results from the journal article, can be used to run workflow notebooks by users following path 2 described in the **Using the repository** section above.

Special cases are the `neversink_mp7_optimal_output` and `neversink_mf6_optimal_output` directories which contain static results from the optimal base models documented in the reports. These are provided for reference and are not overwritten by users rerunning the models but provided for comparison.

## /processed\_data
MODFLOW and PEST data sets developed in `notebooks_preprocessing`, derived from `source_data`.
## /python\_packages_\_static
Static versions of the model-specific Python packages.
## /scripts
Python scripts used during model building, history matching, and MODPATH post-processing
## /source
Source code for the MODFLOW 6, MODPATH 7, and PEST++ binaries.
## /source_data
Unmanipulated MODFLOW and PEST input and assimilation data sets. 
## /supplementary_tables
Excel spreadsheet with tables that provide prior and posterior information on the parameters and outline performance of observation groups in the iES runs documented in the paper. 
## environment.yml
[Conda environment file](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) that can be used to create the `neversink`  environment.
## neversink\_full.yml
A [modflow-setup](https://github.com/aleaf/modflow-setup) YAML format [configuration file](https://aleaf.github.io/modflow-setup/latest/config-file.html) used to build the neversink MODFLOW6 model.


