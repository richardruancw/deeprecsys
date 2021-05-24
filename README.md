# Deep Recommendation system

## install

```bash
# use any of your python env
source /venv/myenv/bin/activate
# install this package and dependencies via
pip install -e .
```

## Run reweight experiment

The core interface is the `run_reweight.py` file. Use `--simulate` flag to train
and evaluate the model in simulation mode (need to generate the simulation ground 
truth use `simulate.py`).

### Real datda

```bash
DATA_PATH='YOUR_DATA_PATH'
python run_reweight.py --data_path $DATA_PATH
```

### Simulation
Train models using semi-synthetic data and evaludate the model using the true
relevance.

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

* `min_step` and `max_step`: Number of batches per min and max step respectively.

* `min_lr` and `max_lr`: Learning rate for min and max step respectively.

* `lambda_`

* `w_lower_bound`: Setting the lower bound to higher value could prevent the model from 
collapsing to trivia solution.
  
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
  --min_lr MIN_LR       Learning rate for maximization step
  --max_lr MAX_LR       Learning rate for minimization step
  --max_len MAX_LEN     Maximum length of sequence
  --num_neg NUM_NEG     Number of random negative samples per real label
  --w_lower_bound W_LOWER_BOUND
                        Lower bound of w(u, i), set it 1 will disable
                        reweighitng
  --model {mf,mlp,seq}  Base model used in min-max training
  --max_step MAX_STEP   number of batches per maximization step
  --min_step MIN_STEP   number of batches per minimization step
  --eval_topk EVAL_TOPK
                        top k items in full evaluations

```