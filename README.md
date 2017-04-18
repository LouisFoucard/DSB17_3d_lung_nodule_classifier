## DSB17 3d lung nodule classifier

3d convnet for the classification of nodules/tumor in lung CT scans. This model was trained on Luna16 for Kaggle's 2017 data science bowl competition (result in top 5%), and is capable of classifying lung ct scan cubes of size 64x64x64 as containing a nodule/tumor or not. This repository only contains the nodule classifier, not the remaining of the pipeline needed for predicting the probablity of a patient of having lung cancer. 

About 8000 cubes with positive samples are extracted from the LUNA dataset (https://luna16.grand-challenge.org/), and the same number of negative examples are sampled at random from lung tissue in patients with no nodule. The model is trained using binary cross entropy for the loss function, an achieves 0.11/0.13 train/test loss, which translates to 0.95 F1 score on the test set (90/10 split). The video below shows an example of nodule (yellow filter overlay) extracted from a LUNA16 lung ct scan.

<img src="https://github.com/LouisFoucard/DSB17_3d_lung_nodule_classifier/blob/master/data/ezgif-1-65620bd01e.gif" height="300">

