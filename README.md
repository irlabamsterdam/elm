# README

This repository contins the code and data for all the experiments for the IRJ submission "BCUBED Revisited: Elements Like Me".

## Overview

The repository is structured as follows:
- SyntheticExperiments
  - These contain the experiments from Section 2.4 of the paper in a notebook and require no dataset to run.
 - utils/metricutils.py
   - This file contains the implementations of both the ELM and BCUBED metrics, as well as some convenience functions for plotting and scores the metrics, and some convenience functions for converting between different cluster representation.
- BaselineExperiments
   - This contains the notebook with the baseline experiments from section 2.5 of the paper.
- BERTExperiments
   - This contains the notebook with the experiments with the BERT model for clustering, as presented in Section 2.6 of the paper
- data
   This folder contains the gold standard clustering of the PSS dataset described in the paper in JSON format, with separate partitions for train and test.
- images
   - This is the directory where all the images created by the notebooks from the different experiments will be saved. These are the exact pictures that are also included in the paper.


## Installation

We strongly recommend using anaconda to run this environment, as the experiments in are all in Jupyter Notebooks, and this will make running them a lot easier. If you really want to install via `pip`, we also supply a `requirements.txt` file for this purpose.


Below are the steps for installing the required dependencies:

- Step 1: Clone /Download the repository to your local computer, for example by using the command below: `test`

- Step 2: change into the directory in the terminal, for example `tst`

- Step 3: If you are using conda, you can use the command below to create a new conda environment with all the requirements you need to run the code in this repository.

`conda env create -f environment.yml`

This will create an environment called 'ELM_IRJ_paper_env', which can be used to run the experiments in this notebook. Note that this environment comes with `jupyter lab` already installed, but without a link to the evironment, so you can't select it in Jupyter Lab yet.
To do this, first activate the environment:

`conda activate ELM_IRJ_paper_env`

Then run the following command:

`ipython kernel install --name "ELM_IRJ_experiments_kernel" --user`

Now the kernel is linked to Jupyter Lab, and you can select it as the kernal to run the notebooks in Jupyter Lab. to start Jupyter Lab, make sure the environment is activated and type `Jupyter Lab` in the terminal.
