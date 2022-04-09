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
  python train.py 
  ```

### Run GAE models:
 ```bash
 cd gaes
 python gae.py --dataset Telecom --encoder GCN -epochs 4001 --lr 0.0001 --val_ratio 0.05 --test_ratio 0.10 --patience 200
 ```



