# Advancing Skin Cancer Detection Integrating a Novel Unsupervised Classification and Enhanced Imaging Techniques

This repository contains the implementation code and pipeline for our novel unsupervised learning methodology for skin lesion classification. 

## Methodology
Our methodology is a combination of modified ESRGAN, a novel histogram feature extraction map, optimal cluster-number estimation, and the application unsupervised clustering algorithm.


## Dataset
The following two public datasets were used in our experiment:
- **ISIC 2019**: [andrewmvd/isic‐2019](https://www.kaggle.com/datasets/andrewmvd/isic‐2019)
- **HAM10000**: [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## ESRGAN

The detailed implentation of ESR‐Generative Adversarial Networks is available at [Code 📁](https://github.com/DrSA-2024/UnsupervisedSkinCancer/tree/main/ESRGAN), and experimented other GAN-based image enhancer pretrained model can be found [Here 📁](https://github.com/DrSA-2024/UnsupervisedSkinCancer/tree/main/Pretrained_GAN-based_Model).




## Histogram Feature Map
Histogram feature map generation and extraction details is available at [Code 📁](https://github.com/DrSA-2024/UnsupervisedSkinCancer/blob/main/Unsupervised_HistogramFreature.ipynb)



## Optimal Cluster Number
Experimented optimal number of cluster finding with DBI and SS score evaluation code is detailed at [Code 📁](https://github.com/DrSA-2024/UnsupervisedSkinCancer/blob/main/Unsupervised_HistogramFreature.ipynb)



## Clustering Performance Evaluation
k-Means clusteing algorithm was finally choosen for our appraoch after exploring several other clusting algorithm. Code is avalable at [Code 📁](https://github.com/DrSA-2024/UnsupervisedSkinCancer/blob/main/Unsupervised_HistogramFreature.ipynb)


## Requirements

The following key Python packages are required to run the code:

- TensorFlow
- PyTorch
- NumPy
- Keras
- Pandas
- Matplotlib


## Citation

Will be updated soon. 

