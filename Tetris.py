import gym
from gym.wrappers import TransformObservation
import numpy as np
from collections import deque

class TetrisPreprocesss(gym.Wrapper):
    def __init__(self, env, noop):
        super().__init__(env)
        self.env = env
        self._noop_max = noop
        self.obses = deque(maxlen=8)
        self.reset()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs2, reward2, done2, _ = self.env.step(0)
        done |= done2
        reward = max(reward, reward2)
        self.obses.append(obs)
        self.obses.append(obs2)
        if done:
            reward = -3
        else:
            if reward == 0:
                reward = .01
            elif reward == 1:
                reward = 27.18
            elif reward == 2:
                reward = 147.8
            elif reward == 3:
                reward = 602.6
            elif reward == 4:
                reward = 2183.93
        if len(self.obses) == 8:
            return np.stack((self.obses[1], self.obses[3], self.obses[5], self.obses[7])), reward, done, info
        return np.stack(self.obses, axis=0), reward, done, info

    def reset(self):
        obs = self.env.reset()
        noops = np.random.randint(8, self._noop_max + 1)
        for _ in range(noops):
            obs, _, done, _ = self.step(self.env.action_space.sample())
            if done:
                obs = self.env.reset()
        if len(self.obses) == 8:
            return np.stack((self.obses[1], self.obses[3], self.obses[5], self.obses[7]))
        return np.stack(self.obses, axis=0)
        
    
def transform(obs):
    obs = obs[28:204, 24:64]
    import cv2

    thresh = 111
    obs = cv2.threshold(obs, thresh, 255, cv2.THRESH_BINARY)[1]

    
    return obs


def create_tetris():
    tetris = gym.make(
        'ALE/Tetris-v5',
        obs_type='grayscale',
        frameskip=4,
        full_action_space=False,
        repeat_action_probability=0,
        render_mode=None
    )
    tetris = TransformObservation(
        tetris,
        transform
    )
    tetris = TetrisPreprocesss(
        tetris,
        174
    )
    return tetris
