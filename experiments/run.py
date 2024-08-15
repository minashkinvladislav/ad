# %%
# -----------------------IMPORTS-----------------------------
import time
import torch
import argparse
import numpy as np
from tqdm import tqdm

import utils
from envs import BernoulliBandit
from decision_transformer import DecisionTransformer
from agents import UCBAgent, RandomAgent, ThompsonSamplingAgent, DTAgent

# %%
# ------------------META RL HYPERPARAMETERS--------------------
n_actions = 10 # number of arms
n_states = 1 # actually, our problem is stateless
n_steps = 50
n_tasks = 5000

# -------------TRANSFORMER HYPERPARAMETERS---------------------
batch_size = 32
block_size = 4 * n_steps # context length
max_iters = 15 * 10**3
eval_interval = 500
learning_rate = 3e-4
eval_iters = 100 # how often do we want to evaluate a performance
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0

# ------------------DEVICE SELECTION-------------------
if torch.cuda.is_available():
    device = 'cuda:0'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'


# ------------------ CLI ------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    '--mode',
    type=str,
    default='load',
    help='Choose mode: "load" – will load existing pretrained model,' + \
    '"train" – will train a model from scratch. Default -- "load"'
)
args = parser.parse_args()


# %%
model = DecisionTransformer(n_embd, n_head, block_size, n_layer, dropout, n_steps, n_states, n_actions)
model = model.to(device)

if args.mode == 'load':
    model.load_state_dict(torch.load('../saved_models/TS_distilled_10_arms_even', map_location=device))
elif args.mode == 'train':
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    agents = [
        UCBAgent(),
        ThompsonSamplingAgent(),
        RandomAgent(),
    ]
    # collect interaction data
    regrets, actions, rewards, states, timesteps = utils.get_regret(
        BernoulliBandit(n_actions),
        agents,
        n_steps=n_steps,
        n_tasks=n_tasks
    )

    agent_to_distill = 'ThompsonSamplingAgent'
    data = torch.stack((
        torch.tensor(states[agent_to_distill], dtype=torch.long)[:, :],
        torch.tensor(actions[agent_to_distill], dtype=torch.long)[:, :],
        torch.tensor(rewards[agent_to_distill], dtype=torch.long)[:, :],
        torch.tensor(timesteps[agent_to_distill], dtype=torch.long)[:, :],
    ), dim=0)

    for iter in tqdm(range(max_iters + 1)):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            losses = utils.estimate_loss(model, eval_iters, data, n_tasks, block_size, batch_size, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        # sample a batch of data
        xb, yb = utils.get_batch(data, 'train', n_tasks, block_size, batch_size, device)
        logits, loss = model(xb[:, 0, :], xb[:, 1, :], xb[:, 2, :], xb[:, 3, :], yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), '../saved_models/custom')

# %%
model.eval()
for policy in ['greedy', 'sample']:
    agents = [
        UCBAgent(),
        ThompsonSamplingAgent(),
        RandomAgent(),
        DTAgent(model, block_size, policy, device)
    ]
    for mode in ['even', 'odd', 'uniform']:
        regrets, actions, rewards, states, timesteps = utils.get_regret(
            BernoulliBandit(n_actions, mode=mode), agents, n_steps=n_steps, n_tasks=100
        )
        utils.plot_regret(
            agents, regrets, n_steps=n_steps,
            title='DTAgent distilled from ' + 'ThompsonSamplingAgent\n' + 'Generation mode: ' + policy + ', bandit mode: ' + mode)
model.train()
