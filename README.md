# Deep Recommendation system

## install

```bash
# use any of your python env
source /venv/myenv/bin/activate
# install this package and dependencies via
pip install -e .
```

## Run reweight experiment

```bash
# start tensorboard
tensorboard --logdir=runs

python run_reweight --model seq --cuda_idx 0
```