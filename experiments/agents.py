# %%
import torch
from torch.nn import functional as F
import numpy as np
from abc import ABCMeta, abstractmethod, abstractproperty


class AbstractAgent(metaclass=ABCMeta):
    def init_actions(self, n_actions, n_steps, initial_state):
        self._n_actions = n_actions
        self._successes = np.zeros(n_actions)
        self._failures = np.zeros(n_actions)
        self._total_pulls = 0

    def update(self, action, state, reward):
        """Observe reward from action and update agent's internal parameters
        """
        self._total_pulls += 1
        if reward == 1:
            self._successes[action] += 1
        else:
            self._failures[action] += 1

    @property
    def name(self):
        return self.__class__.__name__


class RandomAgent(AbstractAgent):
    def get_action(self):
        return np.random.randint(0, self._n_actions)


class ThompsonSamplingAgent(AbstractAgent):
    def init_actions(self, n_actions, n_steps, initial_state):
        self._n_actions = n_actions
        self._successes = np.ones(n_actions)
        self._failures = np.ones(n_actions)
        self._total_pulls = 2 * n_actions

    def get_action(self):
        ts = np.random.beta(self._successes, self._failures, size=self._n_actions)
        return np.argmax(ts)


class UCBAgent(AbstractAgent):
    def init_actions(self, n_actions, n_steps, initial_state):
        self._n_actions = n_actions
        self._successes = np.ones(n_actions)
        self._failures = np.ones(n_actions)
        self._total_pulls = 2 * n_actions

    def get_action(self):
        ucb = self._successes / (self._successes + self._failures) + np.sqrt(2*np.log(self._total_pulls + 1)/(self._successes + self._failures))
        return np.argmax(ucb)


class DTAgent(AbstractAgent):
    def __init__(self, model, block_size, policy, device):
        self.model = model
        self.device = device
        self.block_size = block_size
        self.policy = policy

    def init_actions(self, n_actions, n_steps, initial_state):
        self.states = torch.zeros(1, n_steps, dtype=torch.int, device=self.device)
        self.actions = torch.zeros(1, n_steps, dtype=torch.int, device=self.device)
        self.returns = torch.zeros(1, n_steps, dtype=torch.int, device=self.device)
        self.time_steps = torch.arange(n_steps, dtype=torch.int, device=self.device)
        self.time_steps = self.time_steps.view(1, -1)

        self.states[:, 0] = torch.as_tensor(initial_state, dtype=torch.int, device=self.device)
        self.step = 0

    def get_action(self):
        logits, loss = self.model(  # fix this noqa!!!
            self.states[:, :self.step + 1][:, -self.block_size:],
            self.actions[:, :self.step + 1][:, -self.block_size:],
            self.returns[:, :self.step + 1][:, -self.block_size:],
            self.time_steps[:, :self.step + 1][:, -self.block_size:],
        )
        logits = logits[:, -1, :] # becomes (B, C)
        if self.policy == 'sample':
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            predicted_action = torch.multinomial(probs, num_samples=1)
        elif self.policy == 'greedy':
            predicted_action = torch.argmax(logits)
        else:
            raise ValueError(
                'Incompatible generation policy for Decision Tranformer!'
            )
        return predicted_action

    def update(self, action, state, reward):
        self.actions[:, self.step] = torch.as_tensor(action, dtype=torch.int)
        self.states[:, self.step] = torch.as_tensor(state, dtype=torch.int)
        self.returns[:, self.step] = torch.as_tensor(reward, dtype=torch.int)
        self.step += 1
