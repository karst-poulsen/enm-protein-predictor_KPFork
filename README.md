# ENM-Protein-Predictor 
This project contains machine learning code for predicting nanoparticle binding. The project includes a package with classes for cleaning and formatting base data; building a random forest classifier on top of the data; and classifier optimization via grid search and recursive feature elimination. Forked from ENM-Protein-Predictor code: https://github.com/mfindlay23/ENM-Protein-Predictor. 

The project is structured as follows:
```angular2html
.
+-- config
|   +-- config-optimize.yml
|   +-- config-optimize-payne.yml
|   +-- config-rfecv.yml
|   +-- config-rfecv-payne.yml
|   +-- config-train.yml
|   +-- config-train-payne.yml
+-- predictionutils
|   +-- __init__.py
|   +-- data_utils.py
|   +-- pipeline.py
|   +-- predictor_utils.py
|   +-- validation_utils.py
+-- python
|   +-- optimize.py
|   +-- rfecv.py
|   +-- train.py
```
### config
The config directory holds configuration files for various run definitions.

### predictionutils
This package contains the following classes:
1) data_utils: various utilities for cleaning and splitting training data
2) pipeline: wrapper for other classes, used to create train, optimize, and rfecv pipelines
3) predictor_utils: wrapper for scikitlearn's RandomForestClassifier, including train, optimize, and rfecv methods
4) validation_utils: supplies method for easily calculating and displaying model metrics

### python
This directory holds wrapper scripts to initialize and run pipelines.

