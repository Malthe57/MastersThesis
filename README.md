# Uncertainty Quantification in Subnetwork Ensemble Methods for Neural Networks
Repository for our master's thesis. Read our thesis [here](Master_thesis.pdf) ⭐

Main papers: 

[Weight Uncertainty in Neural Networks](https://arxiv.org/abs/1505.05424), by Blundell et al.

[Training independent subnetworks for robust prediction](https://arxiv.org/abs/2010.06610), by Havasi et al.

# Background
Uncertainty quantifiation in machine learning is the discipline that focuses on estimating and quantifying the uncertainty of machine learning models, such as deep neural networks. 
Modern deep neural networks tend to be overconfident even when they are incorrect and provide poorly calibrated
uncertainty estimates. Bayesian neural networks (BNN) and subnetwork
ensembles, such as MIMO, have been shown to improve the calibration and robustness
of uncertainty estimates for deep neural networks. We introduce the MIMBO
neural network, which is a combination of the two aforementioned methods that combines
the learnable weight posteriors of BNNs with the subnetwork ensemble of MIMO.

In our thesis, we apply these methods to supervised learning tasks and show that subnetwork ensembles give better calibrated uncertainty estimates without adding much computational cost. We demonstrate that when presented out-of-distribution (Ood) data, the uncertainty estimates reflect that the models become uncertain. **In summary**, we demonstrate subnetwork ensemble models *know* when they *don't know* anything.

# Models
Currently, the following models are supported:
- Standard neural network
- Multi-input multi-output (MIMO)
- Naive multiheaded
- Bayesian neural network (BNN)
- Multi-input multi Bayesian output (MIMBO)

With the following architectures:
- Wide ResNet (28-10 is default)
- MediumCNN

# Results

With $M=3$ subnetworks we achieve a similar accuracy and much better uncertainty estimates, in the form of Brier score, NLL and ECE (lower is better).

## CIFAR10
| Model         | Accuracy   | Brier score | NLL       | ECE        |
|---------------|------------|-------------|-----------|------------|
| Deterministic | **0.9576** | **0.00689** | 0.171     | 0.0258     |
| MIMO M=3      | 0.9555     | 0.00701     | **0.159** | **0.0133** |

## CIFAR100
| Model         | Accuracy   | Brier score | NLL       | ECE        |
|---------------|------------|-------------|-----------|------------|
| Deterministic | **0.7987** | 0.00290 | 0.803     | 0.0502     |
| MIMO M=3      | 0.7979     | **0.00289**     | **0.783** | **0.0299** |

# Visualisations
![Baseline](images/baseline.png) ![MIMO](images/mimo.png)
<p align="center">Deterministic neural networks are often overconfident, i.e. they are much more confident than they are accurate. On the other hand, the subnetwork ensemble MIMO model achieves much more well-calibrated uncertainty estimates. <p align="center">


# Activate environment on HPC
$ are terminal commands
1. open terminal in same folder as this project and type the following commands (you can paste them into the terminal with middle mouse click)
2. ```$ module load python3/3.11.7```
3. ```$ module load cuda/11.8```
4. ```$ python3 -m venv MT```
5. ```$ source MT/bin/activate```
6. ```$ pip3 install -r requirements.txt```

# Training
Configure the ``config.yaml`` file with the correct experiment settings and run the following in the terminal:
```
python src/train_classification.py
```
Alternatively on HPC, run the jobscript ``jobscript_classification.sh``:
```bash
bsub < jobscript_classification.sh
```

# Inference
Run the inference script in the terminal:
```bash
python src/infrence_classification.py
```
Alternatively on HPC, run the jobscript ``jobscript_inference_class.sh``:
```bash
bsub < jobscript_inference_class.sh
```




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
