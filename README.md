# Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision Making

Created by Iago Pachêco Gomes at USP - ICMC, University of São Paulo - Institute of Mathematics and Computer Science

**(waiting for the result of the submission to Internation Journal of Fuzzy Systems)**

## Introduction

T2FIS Driving Style is an implementation of a Driving Style Recognition using Interval Type-2 Fuzzy Inference System [1]. This repository has the codes to extract the data sequences from Argoverse's trajectory prediction dataset; the codes to calculate the features vectors; the implementation of clustering algorithms (Kmeans, Fuzzy C-means, Gaussian Mixture Models Clusteris, and Agglomerative Hierarchical Clustering) used to compare the results; and, the implementations of Type-1 and Type-2 Fuzzy Inference Systems.

## Abstract

Driving styles summarize different driving behaviors that reflect in the movements of the vehicles. These behaviors may indicate a tendency to perform riskier maneuvers, consume more fuel or energy, break traffic rules, or drive carefully. Therefore, this paper presents a driving style recognition using Interval Type-2 Fuzzy Inference System with Multiple Experts Decision-Making for classifying drivers into calm, moderate and aggressive. This system receives as input features longitudinal and lateral kinematic parameters of the vehicle motion. The type-2 fuzzy sets are more robust than type-1 fuzzy sets when handling noisy data, because their membership function are also fuzzy sets. In addition, a multiple experts approach can reduce the bias and imprecision while building the fuzzy rulebase, which stores the knowledge of the fuzzy system. The proposed approach was evaluated using descriptive statistics analysis, and compared with clustering algorithms and a type-1 fuzzy inference system. The results show the tendency to associate lower kinematic profiles for the driving styles classified with the type-2 fuzzy inference system when compared to other algorithms, which is in line with the more conservative approach adopted in the aggregation of the experts' opinions.

## License

Apache License 2.0

## Citation
``` 
@article{gomes2021driving,
  title={Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision Making},
  author={Gomes, Iago Pach{\^e}co and Wolf, Denis Fernando},
  journal={arXiv preprint arXiv:2110.13805},
  year={2021}
}
```
## Usage

### Requirements

- Python 3.8
- skfuzzy 0.2 (https://pythonhosted.org/scikit-fuzzy/)
- scikit-learn 0.23.2 (https://scikit-learn.org/stable/)
- PyIT2FLS 0.6.1 (https://haghrah.github.io/PyIT2FLS/examples.html)
- Argoverse API (https://github.com/argoai/argoverse-api)

### Features
#### Dataset
1) Follow the instructions to install the Argoverse dataset API at: https://github.com/argoai/argoverse-api
2) Download *training* and *validation* datasets for Motion Forecasting v1.1

#### Sequences Extraction 

1) at features/argoverse_template:

  ```
  python extract_sequences.py --data_dir <path to where the data is saved> --features_dir <path to where you want to save the sequences> --mode <train or val> --batch_size 500 --obs_len 5 --filter <ekf, none, or savgol>
  ```
  
2) at features:

  ```
  python compute_features.py --data_dir <path to where the sequences are saved> --features_fir <path to where you want to save the features> --mode <train or val> --batch_size 100 --obs_len 5 --filter <ekf, none, or savgol>
  ```
### Clustering

### Fuzzy Inference Systems

## References

[1] (under revision) GOMES, Iago Pachêco; WOLF, Denis Fernando. Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision Making. **Expert Systems with Applications**. 2021.



## Contact

If you find any bug or issue of the software, please contact 'iagogomes at usp dot br' or 'iago.pg00 at gmail dot com'
