from scipy.signal import convolve2d
import numpy as np
class Maze:
    def __init__(self,r_map):
        self.r_map = r_map

class Transition_kernels(dict):
    def __init__(self, key_val_pairs=None):
        super().__init__(key_val_pairs)

class Policy:
    def __init__(self, maze, actions):
        pass

class Q_function:
    def __init__(self, maze, H):
        self.r_map = maze.r_map
        self.H = H
        self.Q = [] # TODO: replicate r_map for H times

    def Q_learning(self, transition_kernels):
        for h in range(H-1,-1,-1):
            Qh_ = self.Q[:,:,h+1]
            for i in ...:
                for j in ...:
                    action_val = []
                    for key in transition_kernels:
                        conv_kernel = transition_kernels[key]
                        reward = ...
                        action_val.append(reward)
                    self.Q[i,j,h] = max(action_val)
        return self.Q

    def Q_eval(self, policy, transition_kernels):
        pass

    def Q_learning_exp(self, transition_kernels):
        pass