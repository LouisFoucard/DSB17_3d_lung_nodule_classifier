## DSB17 3d lung nodule classifier

This repository contains the first stage of my solution for the 2017 Kaggle data science bowl (ranked in the top 3%). It consists in a 3d convnet for the classification of lung proposed tissue regions for nodules/tumor in lung CT scans. This classifier is then used to select proposed region in the DSB17 and increase the training signal for the cancerous/benign classification task. 

The model was trained on the Luna16 dataset (https://luna16.grand-challenge.org/), and its objective is to classify lung ct scan cubes of size 64x64x64 as containing a nodule/potential tumor or not. This repository only contains the nodule classifier, not the remaining of the pipeline needed for predicting the probablity of a patient of having lung cancer. 

Below is a representation of one full lung scan. The video goes down the z axis, from the neck to the abdomen. A yellow filter is overlayed on top of a potential tumor.

<img src="https://github.com/LouisFoucard/DSB17_3d_lung_nodule_classifier/blob/master/data/ezgif-1-65620bd01e.gif" height="300">

The Luna dataset contains some 1000 such scans. In order to build the training data for the tissue classifier, each one of these scan is diced into small 64x64x64 cubes. About 8000 cubes with positive samples are extracted, and the same number of negative examples are sampled at random from a total of 2 million cubes extracted from lung tissue in patients with no nodule.
Below are two example of cubes from the positive class:

<img src="https://github.com/LouisFoucard/DSB17_3d_lung_nodule_classifier/blob/master/data/ezgif-3-0802fafde5.gif" height="200"><img src="https://github.com/LouisFoucard/DSB17_3d_lung_nodule_classifier/blob/master/data/ezgif-3-9727116d29.gif" height="200">

Heavy 3d data augmentation (shear, stretch, rotation, translation, axes flip) and batch normalization achieves 0.95 F1 score on a held out test set (90/10 split). 

