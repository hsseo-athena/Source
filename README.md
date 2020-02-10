# Overview
This project provides the deep-learning framework for a Generalized loss function (GLF) in segmentation task.

# Repository Contents
**batch.py**: code for batch normalization in the network  
**Network_D.py**: code for network including GLF (Deterministic model)  
**main_Network_D.py**: main code to run evaluation (Deterministic model)  
**Network_E.py**: code for network including GLF (Exploratory model)  
**main_Network_E.py**: main code to run evaluation (Exploratory model)  

# Installation Guide  
Install Tensorflow  
https://www.tensorflow.org/install  

# Instruction for Use  
**Data availability**  
The lung and liver tumor datasets are publicly available at https://medicaldecathlon.com and https://competitions.codalab.org/competitions/17094, respectively.  

Please use both network input and output data are .mat file. Deep learning models were built using standard libraries and scripts that are publicly available in TensorFlow r1.9. The custom codes were written in Python 3.5.2.  

Running a model inference on one data sample should take approximately 5 mins using a computer with a NVIDIA Tesla V100 GPU.
