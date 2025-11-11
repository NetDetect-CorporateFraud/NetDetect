# NetDetect for Corporate Fraud Detection
The materials in this repository are the codes and configuration files used in the MISQ paper "How a Rotten Apple May Spoil the Barrel: Corporate Fraud Detection via Dynamic Business Networks"
1. Computational Environment
The packages and corresponding versions used to run the code are listed as follows:
-	Python >= 3.8.0 
-	numpy >= 1.24.2
-	torch >= 1.12.0
-	dgl >= 1.1.2
-	sklearn >= 1.3.2
-	scipy >= 1.10.1
-	matplotlib>=3.7.5
-	re >= 2.2.1
-	json >= 2.0.9
-	argparse >= 1.1
-	shap>=0.44.1
-	xgboost>=1.6.2
-	yaml>=5.1.2
2. Programs/Code
All code is provided in the source code package. The functionality of each file or directory is summarized below:
-	attention.py: implements the basic attention mechanism and mean pooling module.
-	hgan.py: implements the Hierarchical Attention-based Network Embedding (HANE) module in the Static Network Feature Extraction (SNFE).
-	loss.py: implements multiple loss functions used in training NetDetect, including Focal Loss, Reweight Loss, and Reweight Revision Loss.
-	main.py: implements the training, validating, testing, and evaluation processes of experiments.
-	model.py: implements the NetDetect framework.
-	utils.py: provides basic functions required for model training, such as generating masks, setting random seeds, and early stopping.
-	the directory of "config": a set of .yaml files that define the parameter settings of all models for CN/US datasets

This algorithm is being used for commercial applications. If needed, please contact me (ahduwei@ruc.edu.cn) to request it.
