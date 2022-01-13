# Cubical homology-based Image Classification
To read more specific infromation, visit https://winnspace.uwinnipeg.ca/handle/10680/1981

## Abstract
Persistent homology is a powerful tool in topological data analysis (TDA) to compute, study and encode efficiently  multi-scale topological features and is being increasingly used in digital image classification.   The topological features represent number of connected components, cycles, and voids  that describe the shape of data.  Persistent homology extracts the birth and death of these topological features through a filtration process.  The lifespan of these features can represented using persistent diagrams (topological signatures).   Cubical homology  is a more efficient method for extracting topological features from a 2D image and uses a collection of cubes to  compute the homology, which fits the digital image structure of grids.  In this research, we propose a cubical homology-based algorithm for extracting topological features from 2D images to generate their topological signatures.  Additionally, we propose a score, which  measures the significance of each of the sub-simplices in terms of persistence.  Also, gray level co-occurrence matrix  (GLCM) and contrast limited adapting histogram equalization (CLAHE) are used as a supplementary method for extracting features.  Machine learning techniques are then employed to classify images using the topological signatures. Among the eight tested algorithms  with six published image datasets using varying pixel dimensions, classes, and distributions, our experiments demonstrate that cubical homology-based machine learning with deep residual network (ResNet 1D) and Light Gradient Boosting Machine (lightGBM) models outperform published results with four out of six image datasets using the extracted topological features.  

## Pipeline
![fig_CP](https://user-images.githubusercontent.com/55457315/149304133-6a314a1b-602d-4495-b8ab-4c5f0194bcf7.PNG)

## Results
![image](https://user-images.githubusercontent.com/55457315/149304824-3bd90360-fcfc-4bcd-b5d4-6a0f6b58dce7.png)
![image](https://user-images.githubusercontent.com/55457315/149305038-d8b27ea7-94e3-45ef-925c-eaaa2fa658c7.png)

## Acknowledgment
This research was funded by NSERC Discovery Grant\#194376 and University of Winnipeg Major research grant\#14977.

## Data sets
- Concrete: https://data.mendeley.com/datasets/5y9wdsg2zt/2
- Mangopest: https://data.mendeley.com/datasets/94jf97jzc8/1
- Indain Fruits: https://data.mendeley.com/datasets/bg3js4z2xt/1
- Fashion MNIST: https://github.com/zalandoresearch/fashion-mnist
- APTOS: https://www.kaggle.com/c/aptos2019-blindness-detection/overview
- Colorectal histology: https://zenodo.org/record/53169#.XGZemKwzbmG

## Multi Scale 1D ResNet
retrieved from https://github.com/geekfeiw/Multi-Scale-1D-ResNet
