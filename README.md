# Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision Making

Created by Iago Pachêco Gomes at USP - ICMC, University of São Paulo - Institute of Mathematics and Computer Science

**(waiting for the result of the submission to Internation Journal of Fuzzy Systems)**

## Introduction

T2FIS Driving Style is an implementation of a Driving Style Recognition using Interval Type-2 Fuzzy Inference System [1]. This repository has the codes to extract the data sequences from Argoverse's trajectory prediction dataset; the codes to calculate the features vectors; the implementation of clustering algorithms (Kmeans, Fuzzy C-means, Gaussian Mixture Models Clusteris, and Agglomerative Hierarchical Clustering) used to compare the results; and, the implementations of Type-1 and Type-2 Fuzzy Inference Systems.

## Abstract

Driving styles summarize different driving behaviors that reflect in the movements of the vehicles. These behaviors may indicate a tendency to perform riskier maneuvers, consume more fuel or energy, break traffic rules, or drive carefully. Therefore, this paper presents a driving style recognition using Interval Type-2 Fuzzy Inference System with Multiple Experts Decision-Making for classifying drivers into calm, moderate and aggressive. This system receives as input features longitudinal and lateral kinematic parameters of the vehicle motion. The proposed approach was evaluated using descriptive statistics analysis, and compared with clustering algorithms and a type-1 fuzzy inference system. The results show the tendency to associate lower and consistent kinematic profiles for the driving styles classified with the type-2 fuzzy inference system when compared to other algorithms.

## License

Apache License 2.0

## Citation
``` 
@article{pacheco2023driving,
  title={Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision-Making},
  author={Pach{\^e}co Gomes, Iago and Wolf, Denis Fernando},
  journal={International Journal of Fuzzy Systems},
  pages={1--19},
  year={2023},
  publisher={Springer}
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

1) at clustering/:
   
   1.1) model training (learning clustering parameters using the train dataset):
   
   1.1.1) Change the kmeans.py file for the other algorithms: fuzzy_c_means.py, gaussian_mixture.py, hierarchical_clustering.py

   ```
   python kmeans.py --data_dir <directory to where the features are saved (npy files)> --result_dir <directory to where the learned model should be saved> --mode train --obs_len <2 or 5> --filter <ekf, none, or savgol>
   ```

   1.2) model testing or validation
   
   ```
   python kmeans.py --data_dir <directory where the features are saved (npy files)> --model_dir <directory where the model is saved> --result_dir <path to where the results should be saved> --mode <test or val> --obs_len <2 or 5> --filter <ekf, none, or savgol>
   ```
   
### Fuzzy Inference Systems

#### Type-1 Fuzzy Inference System

1) at fuzzy_t1/
     
   ```
   python fuzzy_t1.py --data_dir <directory to where the features are saved (npy files)>  --rules_dir <directory where the FLS' rules are saved> --result_dir <path to where the results should be saved> --mode <train, val, or test> --obs_len <2 or 5> --filter <ekf, none, or savgol> --expert_mode <single, multiple>
   ```
#### Type-2 Fuzzy Inference System

1) at fuzzy_t2/
   ```
   python fuzzy_t2.py --data_dir <directory to where the features are saved (npy files)>  --rules_dir <directory where the FLS' rules are saved> --result_dir <path to where the results should be saved> --mode <train, val, or test> --obs_len <2 or 5> --filter <ekf, none, or savgol> --expert_mode <single, multiple>
   ```

## References

[1] GOMES, Iago Pachêco; WOLF, Denis Fernando. Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision Making. **Internation Journal of Fuzzy Systems**. 2023.



## Contact

If you find any bug or issue of the software, please contact 'iagogomes at usp dot br' or 'iago.pg00 at gmail dot com'
