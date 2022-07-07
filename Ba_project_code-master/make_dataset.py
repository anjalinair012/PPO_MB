from osim.env import RUGTFPEnv
from numpy import random
import numpy as np
import os
import csv
import pdb
import pandas as pd


# Create environment
env = env = RUGTFPEnv(model_name="OS4_gait14dof15musc_2act_LTFP_VR_DynAct.osim",
        visualize=True, integrator_accuracy=1e-3, difficulty=1,
        seed=random.randint(1,1000), report=None, stepsize=0.01)
restore_model_from_file = False
stateDim=env.get_observation_space_size()
actionDim=env.get_action_space_size()
trajLength = 100
num_traj = 500
data = []


def add_noise(self, x, noiseToSignal=0.001):
	mean_data = np.mean(x, axis=0)
	std_of_noise = mean_data * noiseToSignal
	for j in range(mean_data.shape[0]):
		if (std_of_noise[j] > 0):
			x[:, j] = x[:, j] + np.random.normal(0, np.absolute(std_of_noise[j]), (x.shape[0],))
	return x



# for traj in range(num_traj):
# 	state = env.reset()
# 	print(traj)
# 	for step in range(trajLength):
# 		action = np.random.randint(2, size = 17)
# 		observation, reward, penalty, done = env.step(action)
# 		diff_obs = (np.array(observation) - np.array(state))
# 		inp = np.concatenate([state,action])
# 		if not np.any(data):
# 			data = np.array([state, action, observation,diff_obs, inp, env.t], dtype= object)
# 		else:
# 			data = np.vstack((data, np.array([state, action, observation,diff_obs, inp, env.t], dtype = object)))
# 		state = observation
# 		if done:
# 			break
# np.save("dataset_rand2.npy",data, allow_pickle=True)

#im_file = pd.read_csv("../EDITED-MATERIAL/175-FIX.csv", index_col=False)
#file_len = len(im_file)
#index_list = np.around(np.random.uniform(0.00,5.47,500), 2)

step_taken = 0

for i in range(num_traj):
	print("---------{}-----------".format(i))
	state = env.reset()
	step_in = 0
	while step_in<=trajLength:
		num_action = np.random.randint(1,3)
		action = np.random.randint(2, size = 17)
		while num_action>0:
			try:
				observation, reward, penalty, done = env.step(action)
			except:	
				print("exception: ", step_in)
				break
			diff_obs = (np.array(observation) - np.array(state))
			inp = np.concatenate([state,action])
			if not np.any(data):
				data = np.array([state, action, observation,diff_obs, inp, env.t, done], dtype= object)
			else:
				data = np.vstack((data, np.array([state, action, observation,diff_obs, inp, env.t, done], dtype = object)))
			num_action -=1
			step_in +=1
			if done:
				break
			state = observation

np.save("datasetRepeatAction.npy",data, allow_pickle=True)


