# Uncertainty Quantification in Subnetwork Ensemble Methods for Neural Networks
Repository for our master's thesis ⭐

Main papers: 

[Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424), by Blundell et al.

[Training independent subnetworks for robust prediction](https://arxiv.org/abs/2010.06610), by Havasi et al.

# Models
Currently, the following models are supported:
- Standard neural network
- Multi-input multi-output (MIMO), by Havasi et al.
- Naive multiheaded, by Havasi et al.
- Bayesian neural network (BNN), Blundell et al.
- Multi-input multi Bayesian output (MIMBO), ours.

With the following architectures:
- Wide ResNet (28-10 is default)
- MediumCNN

# Experiments
TBD

# Activate environment on HPC
$ are terminal commands
1. open terminal in same folder as this project and type the following commands (you can paste them into the terminal with middle mouse click)
2. ```$ module load python3/3.11.7```
3. ```$ module load cuda/11.8```
4. ```$ python3 -m venv MT```
5. ```$ source MT/bin/activate```
6. ```$ pip3 install -r requirements.txt```


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
