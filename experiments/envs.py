# %%
import numpy as np

class BernoulliBandit:
    def __init__(self, n_actions, mode='even'):
        self._state = 0
        self.action_count = n_actions
        self.mode = mode

    def optimal_reward(self):
        """ Used for regret calculation
        """
        return np.max(self._probs)

    def step(self, action):
        """ Pull selected arm
        """
        if np.any(np.random.random() > self._probs[action]):
            reward = 0.0
        else:
            reward = 1.0
        return self._state, reward, False, None

    def reset(self):
        """ Assign new probabilities to the arms
        """
        self._probs = np.random.rand(self.action_count)
        # work-around for creating bandit with reward distribution on even/odd arms
        if self.mode == 'even':
            more_probable_indices = np.arange(0, self.action_count, 2)
            less_probable_indices = np.arange(1, self.action_count, 2)
        elif self.mode == 'odd':
            more_probable_indices = np.arange(1, self.action_count, 2)
            less_probable_indices = np.arange(0, self.action_count, 2)
        elif self.mode == 'uniform':
            more_probable_indices = np.arange(self.action_count)
            less_probable_indices = np.arange(self.action_count)
        else:
            raise ValueError('Wrong bandit mode!')

        argmax = np.argmax(self._probs)
        if np.random.random() > 0.05:
            idx = np.random.choice(more_probable_indices)
        else:
            idx = np.random.choice(less_probable_indices)

        self._probs[idx], self._probs[argmax] = self._probs[argmax], self._probs[idx]

        return self._state
