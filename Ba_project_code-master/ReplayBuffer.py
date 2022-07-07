'''https://github.com/sudharsan13296/Hands-On-Reinforcement-Learning-With-Python/blob/master/12.%20Capstone%20Project:%20Car%20Racing%20using%20DQN/12.3%20Replay%20Memory.ipynb'''
import pdb

import tensorflow as tf
import numpy as np
import csv
from dataReader import fileRead
import pandas as pd

class ReplayMemoryFast:

    # first we define init method and initialize buffer size
    def __init__(self, memory_size, batch_size, state_size, params_passed = 6):
        self.state_size = state_size
        # max number of samples to store
        self.memory_size = memory_size

        # mini batch size
        self.batch_size = batch_size
        self.experience = [[None]*params_passed] * self.memory_size
        self.current_index = 0
        self.size = 0

    # next we define the function called store for storing the experiences
    def store(self,**kwargs):
        size = len(list(kwargs.values())[0])

        # store the experience as a tuple (current state, action, reward, next state, is it a terminal state)
        for b in range(size):
            if self.current_index>=self.memory_size:
                self.current_index = 0
            self.experience[self.current_index] = [kwargs['observation'][b], kwargs['action'][b],kwargs['next_observation'][b],kwargs['diff_observation'][b],kwargs['scaled_inputs'][b],kwargs['timesteps'][b], kwargs['dones'][b]]
            self.current_index += 1

        self.size = min(self.size + size, self.memory_size)
        # if the index is greater than  memory then we flush the index by subtracting it with memory size



    # we define a function called sample for sampling the minibatch of experience
    #[[]],[np.array],[[]],[np.array],[np.array],[]
    def sample(self, sample_size = -1,replace=True, seed = None):
        if sample_size == 0:
            return None,None,None,None,None,None, None
        if sample_size == -1:
            sample_size = self.size-1
        # first we randomly sample some indices
        if sample_size>self.size:
            replace=True
        samples_index = np.random.choice(range(self.size),sample_size,replace = replace) # add seed?

        # select the experience from the sampled index
        samples = [self.experience[int(i)] for i in samples_index]

        observations, actions, next_observations, diff_observations, inputs, timesteps, dones = map(list,zip(*samples))
        return observations,actions,next_observations,diff_observations, inputs, timesteps, dones


    def clear(self):
        self.experience = [[None]*6] * self.memory_size
        self.current_index = 0
        self.size = 0

    def save_to_file(self,filename):
        data = np.array([self.experience[0]])
        for traj in range(1,self.size):
                data = np.vstack((data,self.experience[traj]))
        np.save(filename,data, allow_pickle=True)


    def load_from_file(self,filename, restore_upto = None, short_obs = True):
        try:
            file_load = np.load(filename, allow_pickle=True)
            if not restore_upto:
                restore_upto = file_load.shape[0]
            Observations = file_load[:restore_upto, 0]
            Actions = file_load[:restore_upto, 1]
            Next_Observations = file_load[:restore_upto, 2]
            Diff_Observations = file_load[:restore_upto, 3]
            Inputs_temp = file_load[:restore_upto, 4]
            if self.state_size < len(Observations[0]):
                for index in range(Observations.shape[0]):
                    Observations[index] = Observations[index][:26] + Observations[index][59:70] + Observations[index][
                                                                                                  82:]
                    Next_Observations[index] = Next_Observations[index][:26] + Next_Observations[index][59:70] + \
                                               Next_Observations[index][82:]
                    Diff_Observations[index] = np.concatenate(
                        (Diff_Observations[index][:26], Diff_Observations[index][59:70], Diff_Observations[index][82:]))
                    Inputs_temp[index] = np.concatenate(
                        (Inputs_temp[index][:26], Inputs_temp[index][59:70], Inputs_temp[index][82:]))
            Observations = Observations.tolist()
            Actions = Actions.tolist()
            Next_Observations = Next_Observations.tolist()
            Diff_Observations = Diff_Observations.tolist()
            Inputs_temp = Inputs_temp.tolist()
            Timesteps = file_load[:restore_upto, 5].tolist()
            Dones = file_load[:restore_upto, 6].tolist()
            self.store(observation=Observations, action=Actions, next_observation=Next_Observations,
                       diff_observation=Diff_Observations, scaled_inputs=Inputs_temp, timesteps=Timesteps, dones=Dones)
            return
        except:
            print(filename,"doesnt exist")
            return




