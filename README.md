## DSB17 3d lung nodule classifier

This repository contains the first stage of my solution for the 2017 Kaggle data science bowl (ranked in the top 3%). It consists in a 3d convnet for the classification of proposed regions for nodules/tumor in lung CT scans. The model was trained on the Luna16 dataset (https://luna16.grand-challenge.org/), and its objective is to classify lung ct scan cubes of size 64x64x64 as containing a nodule/potential tumor or not. This repository only contains the nodule classifier, not the remaining of the pipeline needed for predicting the probablity of a patient of having lung cancer. 

About 8000 cubes with positive samples are extracted from the Luna dataset, and the same number of negative examples are sampled at random from a total of 2 million cubes extracted from lung tissue in patients with no nodule. Heavy 3d data augmentation (shear, stretch, rotation, translation, axes flip) and batch normalization achieves 0.95 F1 score on a held out test set (90/10 split). The video below shows an example of nodule (yellow filter overlay) extracted from a LUNA16 lung ct scan.

<img src="https://github.com/LouisFoucard/DSB17_3d_lung_nodule_classifier/blob/master/data/ezgif-1-65620bd01e.gif" height="300">

