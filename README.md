# Probability of default estimation
Estimation of probability of default on novel real world data from Orbis.

This project is done for the Phd course in Statistics for data science at University of Brescia.

## how to run:

You can use the `analysis.ipynb` and run all.
This notebook sources other relevant scripts with data preprocessing and other functions. 

It loads already pretrained models (XGB, MLPC and logistic regression). But if you are interested in how the models were trained, take a look at the `models` folder with files for particular models.

The notebook comes with the description of the project and is 

Take a look also at the `latex/slides_PD.pdf` for slides that describes the project

## Strucutre of the project:

- `orbis_data`:  folder with raw data sourced from Orbis private companies database

- `models`: foleder with pretrained models (three models each trained on different resampled dataset)

- `src`: folder with python scripts to run the modelling workflow

- `latex`: latex folder with slides that presents the herein project 
    - `latex/img`: here are all of the charts/tables stored that were used in the slides and in the `analysis.ipynb`
