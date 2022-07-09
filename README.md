# BSAL
The repository contains the code for paper "BSAL: A Framework of Bi-component Structure and Attribute Learning for Link Prediction" accepted by SIGIR 2022.


## Requirements:
* torch
* numpy
* torch_geometric
* sklearn
* scipy


## Runs:
The BSAL code is contained in the code dir and the data are downloaded and stored in the data dir.
### Run BSAL models:
  ```bash
  cd code
  python train.py --runs 10 --lr 0.001 --wd 5e-4 --epochs 401 --bs 32 --patience 50 --dynamic_train False --dynamic_val False --dynamic_test False --dataset disease --val_ratio 0.05 --test_ratio 0.10 --train_percent 1.0 --val_percent 1.0 --test_percent 1.0 --use_new_split False --use_feat False
  ```

### Run Heuristic models:
  ```bash
  cd code
  python heuristic.py --dataset disease --batch_size 32 --use_heuristic CN
  ```

### Run Emb_link_models:
  ```bash
  cd code
  python emb_link_pred.py
  ```

### Run GAE models:
 ```bash
 cd gaes
 python gae.py --dataset Telecom --encoder GCN -epochs 4001 --lr 0.0001 --val_ratio 0.05 --test_ratio 0.10 --patience 200
 ```

### Run SEAL models:
  ```bash
  cd code
  python train.py --runs 10 --lr 0.001 --wd 5e-4 --epochs 401 --bs 32 --patience 50 --dynamic_train False --dynamic_val False --dynamic_test False --dataset disease --val_ratio 0.05 --test_ratio 0.10 --train_percent 1.0 --val_percent 1.0 --test_percent 1.0 --use_new_split False --use_feat False
  ```

## Reference:
If you find the code useful, please cite our paper:
```bash
@inproceedings{10.1145/3477495.3531804,
author = {Li, Bisheng and Zhou, Min and Zhang, Shengzhong and Yang, Menglin and Lian, Defu and Huang, Zengfeng},
title = {BSAL: A Framework of Bi-Component Structure and Attribute Learning for Link Prediction},
year = {2022},
isbn = {9781450387323},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3477495.3531804},
doi = {10.1145/3477495.3531804},
location = {Madrid, Spain},
series = {SIGIR '22}
}
```
