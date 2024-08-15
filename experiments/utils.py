import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


@torch.no_grad()
def get_regret(env, agents, n_steps=50, n_tasks=100):
    regrets = {agent.name: np.zeros((n_tasks, n_steps)) for agent in agents}
    actions = {agent.name: np.zeros((n_tasks, n_steps)) for agent in agents}
    rewards = {agent.name: np.zeros((n_tasks, n_steps)) for agent in agents}
    states = {agent.name: np.zeros((n_tasks, n_steps)) for agent in agents}
    timesteps = {agent.name: np.array([np.arange(n_steps) for _ in range(n_tasks)]) for agent in agents}

    for task in tqdm(range(n_tasks)):
        initial_state = env.reset()
        optimal_reward = env.optimal_reward()

        for agent in agents:
            agent.init_actions(env.action_count, n_steps, initial_state)

        for step in range(n_steps):
            for agent in agents:
                action = agent.get_action()
                state, reward, _, _ = env.step(action)
                agent.update(action, state, reward)

                regrets[agent.name][task][step] += optimal_reward - reward
                actions[agent.name][task][step] += action
                rewards[agent.name][task][step] += reward

    for agent in agents:
        regrets[agent.name] = np.cumsum(regrets[agent.name], axis=1)

    return regrets, actions, rewards, states, timesteps


def plot_regret(agents, regrets, n_steps, title=None):
    plt.clf()
    for agent in agents:
        mean = np.mean(regrets[agent.name], axis=0)
        confidence_interval = np.std(regrets[agent.name], axis=0) / 2
        plt.plot(np.arange(n_steps), mean, label=agent.name)
        plt.fill_between(
            np.arange(n_steps),
            (mean - confidence_interval),
            (mean + confidence_interval),
            alpha=.2,
            label='_' + agent.name
        )

    plt.legend()
    plt.ylabel("regret")
    plt.xlabel("steps")
    if title is None:
        plt.show()
    else:
        plt.title(title)
        plt.savefig('../figures/' + title + '.png')


@torch.no_grad()
def estimate_loss(model, eval_iters, data, n_tasks, block_size, batch_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, split, n_tasks, block_size, batch_size, device)
            logits, loss = model(X[:, 0, :], X[:, 1, :], X[:, 2, :], X[:, 3, :], Y)
            losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


def get_batch(data, split, n_tasks, block_size, batch_size, device):
    # shuffle data and split into train/val
    # by the way there is no need in doing so, at least for now
    # task_indices = torch.randperm(n_tasks)
    # data = data[:, task_indices]
    n = int(0.8*n_tasks)
    data = data[:, :n] if split == 'train' else data[:, n:]
    n_modality, n_tasks, n_steps = data.shape
    data = data.reshape(n_modality, n_tasks*n_steps)
    ix = torch.randint(n_tasks*n_steps - block_size, (batch_size,))
    x = torch.stack([data[:, i:i+block_size] for i in ix])
    y = torch.stack([data[1, i:i+block_size] for i in ix]) # choose only actions to predict
    x, y = x.to(device), y.to(device) # (Batch, Modality, BlockSize (context length))
    return x, y
