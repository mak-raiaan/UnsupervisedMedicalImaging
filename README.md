# Advancing Skin Cancer Detection Integrating a Novel Unsupervised Classification and Enhanced Imaging Techniques


## Paper

📄 This code implements the paper published in a Q1 Journal, CAAI Transactions on Intelligence Technology, with an impact factor of 8.4.

**Title**: [Advancing skin cancer detection integrating a novel unsupervised classification and enhanced imaging techniques]([https://www.sciencedirect.com/science/article/pii/S1746809424003379](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12410))

## Methodology
Our methodology is a combination of modified ESRGAN, a novel histogram feature extraction map, optimal cluster-number estimation, and the application unsupervised clustering algorithm.


## Dataset
The following two public datasets were used in our experiment:
- **ISIC 2019**: [andrewmvd/isic‐2019](https://www.kaggle.com/datasets/andrewmvd/isic‐2019)
- **HAM10000**: [kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## ESRGAN

The detailed implentation of ESR‐Generative Adversarial Networks is available at [Code 📁](https://github.com/mak-raiaan/UnsupervisedMedicalImaging/tree/main/ESRGAN), and experimented other pretrained model can be found [Here 📁](https://github.com/mak-raiaan/UnsupervisedMedicalImaging/tree/main/Contrastive%20Learning).




## Histogram Feature Map
Histogram feature map generation and extraction details is available at [Code 📁](https://github.com/mak-raiaan/UnsupervisedMedicalImaging/blob/main/Unsupervised_HistogramFreature.ipynb)



## Optimal Cluster Number
Experimented optimal number of cluster finding with DBI and SS score evaluation code is detailed at [Code 📁](https://github.com/mak-raiaan/UnsupervisedMedicalImaging/blob/main/Unsupervised_HistogramFreature.ipynb)



## Clustering Performance Evaluation
k-Means clusteing algorithm was finally choosen for our appraoch after exploring several other clusting algorithm. Code is avalable at [Code 📁](https://github.com/mak-raiaan/UnsupervisedMedicalImaging/blob/main/Unsupervised_HistogramFreature.ipynb)


## Requirements

The following key Python packages are required to run the code:

- TensorFlow
- PyTorch
- NumPy
- Keras
- Pandas
- Matplotlib


## Citation Request

If you find this work helpful for your research, please consider citing our paper:
- **Cite:**
```bibtex
@article{rahman2025advancing,
  title = {Advancing skin cancer detection integrating a novel unsupervised classification and enhanced imaging techniques},
  author = {Rahman, Md. Abdur and Fahad, Nur Mohammad and Raiaan, Mohaimenul Azam Khan and Jonkman, Mirjam and De Boer, Friso and Azam, Sami},
  journal = {CAAI Transactions on Intelligence Technology},
  pages = {1--20},
  year = {2025},
  doi = {10.1049/cit2.1241020},
  publisher = {Wiley}
}
