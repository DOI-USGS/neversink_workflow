# Workflow to accompany source water delineation for the Neversink-Rondout Basin, New York
This repository documents both the workflow and contains the dataset to repeat the model-building, data assimilation, and analysis in the two papers: 

Corson-Dosch, et al. _Areas contributing recharge to priority wells in valley-fill aquifers in the Neversink and Rondout Basins, New York_ in press as a USGS Scientific Investigations Report and 

Fienen, et al. _Risk-based wellhead protection decision support: a repeatable workflow approach_ submitted to Groundwater

# Getting started
The workflow in this repository depends on a number of python packages. Static versions of the model-specific packages are included in the repository. However, a Conda environment must be installed to provide more general underlying packages.

To install, an environment file is provided in the root directory of the repository. Follow these steps:

* Install Anaconda python (if not already installed) from [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* In an Anaconda prompt on Windows, or in a terminal window on Mac or Linux, navigate to the top directory of this repository and type `conda env create -f environment.yml`
* This will create a `neversink` environment.
* Before running any notebooks or scripts, in an Anaconda prompt of terminal window, type `conda activate neversink` to activate the neversink environment.

# Organization 

* Need to explain about starting with preprocessing, or model construction, or just the workflow
* Need to explain about the `run_ensemle` flag

## Preprocessing Notebooks

## Model Construction Script

## Main Workflow notebooks
### Completed notebooks with manuscript results
### Blank notebooks to run locally with a user-defined ensemble
### _description of the workflow within either of the two above_

