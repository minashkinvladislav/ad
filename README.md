# Algorithm Destillation & Multi-Armed Bandits

This repository contents source code used to train & evaluate
Transformer with Algorithm Destillation in a multi-armed bandit task.

## Local installation

First of all, make sure that you have `python>=3.10`installed.

```bash
cd $HOME && git clone https://github.com/minashkinvladislav/ad.git
cd $HOME/ad
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

To run experiments with pretrained model:
```bash
cd ./experiments
python run.py --mode=load
```

To train model by yourself and then run experiments:
```bash
cd ./experiments
python run.py --mode=train
```

Custom model will be saved at `saved_models/custom`.

## Results

Results can be found in `figures` folder .
