# Deep Recommendation system

## install

```bash
# use any of your python env
source /venv/myenv/bin/activate
# install this package and dependencies via
pip install .
# OR, install with development flag if you want to modify the code, then do:
# pip install -e .
```

## Run reweight experiment

The core interface is the `run_reweight.py` file. Use `--simulate` flag to train
and evaluate the model in simulation mode (need to generate the simulation ground 
truth use `simulate.py`).

### Data preprocessing
The data preprocessing scripts are in each sub-folders inside the `data` folder. They include `book_data.py`, 
`lastfm.py`, and `ml-1m.py`. Following the steps below:
1. Download the data and unzip them under data folder
2. Make sure the data script and the data are in the same folder.
3. Run the script. Processed data are saved also in the same folder.


### Real datda

```bash
DATA_PATH='YOUR_DATA_PATH'
python run_reweight.py --data_path $DATA_PATH
```

### Simulation
Train models using semi-synthetic data and evaludate the model using the true
relevance. Following the steps as:
1. Generate ground-truth models as well as user-item interaction data.
2. Run the `run_reweight.py` with the same data path and use `--simulate` flag.

```bash
DATA_PATH='YOUR_DATA_PATH'
# generate ground-truth
python simulate.py --data_path $DATA_PATH

# run the model in simulation mode
python run_reweight.py --data_path $DATA_PATH --simulate
```

#### Key hyper-parameters for simulation

* `quantile`: Default is 0.9, use larger quantile means fewer items would be exposed to
users.

#### Key hyper-parameters for training


* `f_step`, `w_step`, and `g_step`: determines the size of each update step.

Training loop is essentially:
```python
for i in range(f_step):
    update(f)
    for j in range(w_step):
        update(w)
        for k in range(g_step):
            update(g)
```

* `model`, `w_model`, and `g_model`: The functional class for f, w and g respectively. Available function class includes
`{mf,mlp,seq}`
  
* `f_lr`, `w_lr`, and `g_lr`: Learning rate for each module.
  
* `lambda_`: Domain balance constraint strength.

* `w_lower_bound`: Setting the lower bound to higher value could prevent the model from 
collapsing to trivia solution. Setting it equals to 1.0 means no weight at all.
  
### Optional arguments

* Simulation

```bash
usage: simulate.py [-h] [--batch_size BATCH_SIZE] [--dim DIM]
                   [--epsilon EPSILON] [--epoch EPOCH] [--p P] [--decay DECAY]
                   [--quantile QUANTILE] [--data_path DATA_PATH]
                   [--data_name DATA_NAME] [--sample_sim]
                   [--item_sample_size ITEM_SAMPLE_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
  --dim DIM             Dimension of embeddings
  --epsilon EPSILON     Simulation parameters, refer to paper for details
  --epoch EPOCH         Simulation parameters, refer to paper for details
  --p P                 Simulation parameters, refer to paper for details
  --decay DECAY         L2 penalty for embeddings
  --quantile QUANTILE   Only items in top x percent quantile are exposed to
                        users
  --data_path DATA_PATH
  --data_name DATA_NAME
  --sample_sim
  --item_sample_size ITEM_SAMPLE_SIZE
```

* Training
```bash
usage: run_reweight.py [-h] [--batch_size BATCH_SIZE] [--dim DIM]
                          [--epoch EPOCH] [--decay DECAY]
                          [--cuda_idx CUDA_IDX] [--data_path DATA_PATH]
                          [--simulate] [--data_name DATA_NAME]
                          [--lambda_ LAMBDA_] [--prefix PREFIX] [--tune_mode]
                          [--lr LR] [--f_lr F_LR] [--g_lr G_LR] [--w_lr W_LR]
                          [--max_len MAX_LEN] [--num_neg NUM_NEG]
                          [--w_lower_bound W_LOWER_BOUND]
                          [--model {mf,mlp,seq}] [--g_model {mf,mlp,seq}]
                          [--w_model {mf,mlp,seq}] [--share_f_embed]
                          [--f_step F_STEP] [--w_step W_STEP]
                          [--g_step G_STEP] [--eval_topk EVAL_TOPK]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Number of pairs per batch
  --dim DIM             Dimension of the embedding
  --epoch EPOCH         Number of epochs
  --decay DECAY         l2 regularization strength for sparse model
  --cuda_idx CUDA_IDX   Which GPU to use, default is to use CPU
  --data_path DATA_PATH
                        Directory that contains all the data
  --simulate            Run the code using simulated data
  --data_name DATA_NAME
                        Observation data after standardization
  --lambda_ LAMBDA_     Lambda as defined in the min-max objective
  --prefix PREFIX
  --tune_mode           Use validation data as testing data.
  --lr LR               Learning rate of SGD optimizer for baseline models
  --f_lr F_LR           Learning rate for f model
  --g_lr G_LR           Learning rate for g model
  --w_lr W_LR           Learning rate for w model
  --max_len MAX_LEN     Maximum length of sequence
  --num_neg NUM_NEG     Number of random negative samples per real label
  --w_lower_bound W_LOWER_BOUND
                        Lower bound of w(u, i), set it 1 will disable
                        reweighitng
  --model {mf,mlp,seq}  Base model used for f
  --g_model {mf,mlp,seq}
                        Base model used for g
  --w_model {mf,mlp,seq}
                        Base model used for w
  --share_f_embed
  --f_step F_STEP       Number of steps for the outer loop to update f
  --w_step W_STEP       Number of steps for the middle loop to update w
  --g_step G_STEP       Number of steps for the inner loop to update g
  --eval_topk EVAL_TOPK
                        top k items in full evaluations



```
