## Classification Analysis of Parkinson Speech Dataset
### Authors: Elcin Ergin, Shu Hayakawa, Timardeep Kaur

#### About

In this study, we aim to analyze and diagnose patients with Parkinson Disease (PD) by applying Machine Learning Techniques (ML) on speech datasets. In particular, we focus on applying variations of Logistic Regression, Support Vector Machines (SVM) and K-Nearest-Neighbour (KNN). The study aims to work on a previous study conducted by Istanbul University. The same datasets were used for this study and were obtained from the following [UCI Database link](https://archive.ics.uci.edu/ml/datasets/Parkinson+Speech+Dataset+with++Multiple+Types+of+Sound+Recordings).

#### Disclaimer

The source code files in the src folder have hard coded file paths to the dataset files. After reorganizing the repository, some of the coded files may not be linked properly to the dataset locations so for users who are interested in testing the scripts, you will have to change the file paths in the source code accordingly. 

#### Folder Structure

The repository contains the scripts, datasets, and results obtained from this study.3 Machine Learning Techniques were applied to the datasets and the file structure is organized as follows:

* raw_data: Contains the raw .txt files obtained from the UCI Database.
* sloo_data: Contains data used for SLOO validation schemes.
* references: Contains the original research paper that applied ML techniques on the raw_datasets.
* presentation: Contains power point presentation for the research project.
* src: Contains scripts, results, and figures for ML Techniques conducted for this study.
	* loso: Contains the implementation and the results for ML Techniques conducting LOSO Validation.
	* sloo: Contains the implementations for SLOO Validation.
	* best_voice_samples: Contains the implementation to select the 3 best voice samples available from the 26.

#### Raw Datasets

[Download Raw Dataset from UCI Database](https://archive.ics.uci.edu/ml/machine-learning-databases/00301/)

#### Installation

Before installing dependencies, please [Install Python](https://www.python.org/downloads/) and [PIP](https://pip.pypa.io/en/stable/installing/).

Install scikit-learn ML Library:

	pip install -U scikit-learn

Install the scipy stack:
	
	pip install -U scipy
	pip install -U numpy
	pip install -U matplotlib

Install the pandas data frame:
	
	pip install -U pandas
	