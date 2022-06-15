# Probability of default estimation
Estimation of probability of default on novel data from Orbis.

This project is done for the Phd course in Statistics for data science at University of Brescia.

## how to run:

You can use the `src/analysis.ipynb` and run all.
This notebook sources other relevant scripts with data preprocessing and other functions. 

It loads already pretrained models (XGB, MLPC and logistic regression). But if you are interested in how the models were trained, take a look at the models.py script.

## Strucutre of the project:

`orbis_data`:  folder with raw data sourced from Orbis private companies database

`models`: foleder with pretrained models (three models each trained on different resampled dataset)

`src`: folder with python scripts to run the modelling workflow

`latex`: latex folder with slides that presents the herein project 

## What else is there to do?:

- Make an EDA of the dataset
- Further develop the model diagnostics
- Finish slides