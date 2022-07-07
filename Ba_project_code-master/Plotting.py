import random
from itertools import count
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from statistics import mean
import pdb
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import numpy as np
import csv
import matplotlib.cm as cm
from neptune.new.types import File


class DynamicPlotting:
	def __init__(self, logger):
		self.fig = plt.figure()
		self.logger = logger
		
		
	def plotLosses(self,history, update_nn_every, path):
		path = os.path.join(path, "Losses.png")
		print("Plotting")

		plt.ylim([0, 15])
		plt.plot(history["train_loss"],label="train")
		plt.plot(history["val_loss"],label="validation")
		plt.ylabel('mse') #set the label for y axis
		plt.xlabel('Iteration') #set the label for x-axis
		plt.title("Average Loss every {} runs with buffer".format(update_nn_every)) #set the title of the graph
		plt.legend()
		plt.show()
		plt.savefig(path)
		plt.clf()
                

	def plotLossesFunc(self,history, update_nn_every):
		fig = plt.figure()
		ani = FuncAnimation(plt.gcf(), self.plotLosses, fargs = (history,update_nn_every))
		plt.tight_layout()
		plt.show()
		plt.clf()

	def plotDifferences(self, real_states, predicted_states, path =""):
		#plotpath1 = os.path.join(path, "ErrorDimInStep.png")
		plt.clf()
		diffSteps = np.array(real_states).reshape(-1,94) - np.array(predicted_states).reshape(-1,94)
		fig, axes = plt.subplots(nrows=5, ncols=4, figsize=(16, 8))
		#pdb.set_trace()
		for row, ax in zip(range(diffSteps.shape[0]), axes.flatten()):
		    ax.plot(diffSteps[row,:])
		    ax.set_title(row, fontsize=10)
		    #ax.set_xticks(range(diffSteps.shape[1]))
		    #ax.set_xticklabels(list(), rotation=90, fontsize=8)
		plt.show()
		plt.title("Error over Dimensions")
		plt.xlabel("Dimensions")
		plt.ylabel("Error")
		if path != "":
			plt.savefig(plotpath1)
		plt.clf()
		
	def plotMeanDifferences(self, real_states, predicted_states, path ="", iteration = 0):
		fig = plt.figure(figsize=(7, 9))
		#plotpath1 = os.path.join(path, "ErrorSteps.png")
		diffSteps = np.mean(np.abs(np.array(real_states).reshape(-1,94) - np.array(predicted_states).reshape(-1,94)), axis = 1)
		plt.title("Error over steps")
		plt.plot(diffSteps)
		plt.xlabel("Steps")
		plt.ylabel("MAE")
		if logger:
			self.logger["Training/ErrorHor{}".format(iteration)].upload(fig)
		#plt.savefig(plotpath1)
		#plotpath2 = os.path.join(path, "ErrorDim.png")
		fig = plt.figure(figsize=(7, 9))
		plt.cla()
		diffSteps = np.mean(np.abs(np.array(real_states).reshape(-1,94) - np.array(predicted_states).reshape(-1,94)), axis = 0)
		plt.title("Error over Dimensions")
		plt.plot(diffSteps)
		plt.xlabel("Dimensions")
		plt.ylabel("MAE")
		if logger:
			self.logger["Training/ErrorDim{}".fromat(iteration)].upload(fig)
		#plt.savefig(plotpath2)
	        
	        
	def PCA_visual(self, real_states, predicted_states, path = "", iteration = 0, header = "", logger = None):
		plotpath = os.path.join(path, "StatesPCA.pdf")
		pca = PCA()
		pipe1 = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
		pipe1.fit(real_states)
		real_transformed = pipe1.transform(real_states)
		pipe2 = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
		predicted_transformed = pipe1.transform(predicted_states)
		fig = plt.figure(figsize=(7, 9))
		colors = cm.rainbow(np.linspace(0, 1, len(real_states)))
		plt.subplot(2,1,1)
		plt.scatter(real_transformed[:,0], real_transformed[:,1], color=colors)
		plt.title("Real States")
		plt.subplot(2,1,2)
		plt.scatter(predicted_transformed[:,0], predicted_transformed[:,1], color=colors)
		plt.title("Predicted states")
		plt.suptitle(header)
		#self.logger["Training/state_visualise{}".format(iteration)].upload(fig)
		if path != "":
			if logger:
				logger["iter_{}/PCA_states".format(iteration)].upload(fig)
			plt.savefig(plotpath)
		plt.clf()
		plt.close()
		
	def compare_observations(self, real_states_dict, predicted_states_dict, path = "", header = "", called_from = 10, iteration = 0, logger = None):
		if called_from>1:
			plotpath = os.path.join(path, "Compare_positions_multi.png")
			name = "Compare_positions_multi.png"
		else:
			plotpath = os.path.join(path, "Compare_positions_step.png")
			name = "Compare_positions_step.png"
		plot_states = []
		pelvis_x = [[],[]]
		pelvis_y = [[],[]]
		pelvis_z =[[],[]]
		ankle_joint_l= [[],[]]
		ankle_joint_r = [[],[]]
		knee_joint_l = [[],[]]
		knee_joint_r = [[],[]]
		hip_joint_l =[[],[]]
		hip_joint_r = [[],[]]
		hip_abd_joint_l = [[],[]]
		hip_abd_joint_r = [[],[]]
		pelvis_pitch = [[],[]]
		pelvis_roll = [[],[]]
		pelvis_rot = [[],[]]
		for index in range(len(real_states_dict)):

			pelvis_x[0].append(real_states_dict[index]["pelvis"]["body_pos"][0])
			pelvis_x[1].append(predicted_states_dict[index]["pelvis"]["body_pos"][0])
			pelvis_y[0].append(real_states_dict[index]["pelvis"]["body_pos"][1])
			pelvis_y[1].append(predicted_states_dict[index]["pelvis"]["body_pos"][1])

			pelvis_z[0].append(real_states_dict[index]["pelvis"]["body_pos"][2])
			pelvis_z[1].append(predicted_states_dict[index]["pelvis"]["body_pos"][2])

			ankle_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['ankle'])
			ankle_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['ankle'])

			ankle_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['ankle'])
			ankle_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['ankle'])

			knee_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['knee'])
			knee_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['knee'])
			knee_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['knee'])
			knee_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['knee'])
			hip_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['hip'])
			hip_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['hip'])
			hip_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['hip'])
			hip_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['hip'])
			hip_abd_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['hip_abd'])
			hip_abd_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['hip_abd'])
			hip_abd_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['hip_abd'])
			hip_abd_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['hip_abd'])
			pelvis_pitch[0].append(real_states_dict[index]["pelvis"]["pitch"])
			pelvis_pitch[1].append(predicted_states_dict[index]["pelvis"]["pitch"])
			pelvis_roll[0].append(real_states_dict[index]["pelvis"]["roll"])
			pelvis_roll[1].append(predicted_states_dict[index]["pelvis"]["roll"])
			pelvis_rot[0].append(real_states_dict[index]["pelvis"]["joint_pos"][2])
			pelvis_rot[1].append(predicted_states_dict[index]["pelvis"]["joint_pos"][2])
		plot_states_1 = [pelvis_x ,pelvis_y ,pelvis_z ,ankle_joint_l,ankle_joint_r ,knee_joint_l ,knee_joint_r ,hip_joint_l, \
			hip_joint_r ,hip_abd_joint_l ,hip_abd_joint_r ,pelvis_pitch ,pelvis_roll ,pelvis_rot]
		plot_states_names = ["pelvis_x" ,"pelvis_y" ,"pelvis_z" ,"ankle_joint_l","ankle_joint_r" ,"knee_joint_l" ,"knee_joint_r" ,"hip_joint_l", \
			"hip_joint_r" ,"hip_abd_joint_l" ,"hip_abd_joint_r" ,"pelvis_pitch" ,"pelvis_roll" ,"pelvis_rot"]
		fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(16, 8))
		#pdb.set_trace()
		for obs_ind, ax in zip(range(len(plot_states_1)), axes.flatten()):
			ax.plot(plot_states_1[obs_ind][0][:100])
			ax.plot(plot_states_1[obs_ind][1][:100])
			ax.set_title(plot_states_names[obs_ind], fontsize=10)
		if logger:
			logger["iter_{}/{}".format(iteration, name)].upload(fig)
		#plt.savefig(plotpath)
		plt.close()
		if called_from>1:
			plotpath = os.path.join(path, "Compare_velocity_multi.pdf")
			name = "Compare_velocity_multi"
		else:
			plotpath = os.path.join(path, "Compare_velocity_step.pdf")
			name = "Compare_velocity_step"
		ankle_vel_l = [[],[]]
		ankle_vel_r = [[],[]]
		knee_vel_l = [[],[]]
		knee_vel_r = [[],[]]
		hip_vel_l = [[],[]]
		hip_vel_r = [[],[]]
		hipabd_vel_l = [[],[]]
		hipabd_vel_r =  [[],[]]
		for index in range(len(real_states_dict)):
			ankle_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['ankle'])
			ankle_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['ankle'])
			ankle_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['ankle'])
			ankle_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['ankle'])
			knee_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['knee'])
			knee_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['knee'])
			knee_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['knee'])
			knee_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['knee'])
			hip_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['hip'])
			hip_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['hip'])
			hip_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['hip'])
			hip_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['hip'])
			hipabd_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['hip_abd'])
			hipabd_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['hip_abd'])
			hipabd_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['hip_abd'])
			hipabd_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['hip_abd'])
		plot_states_2 = [ankle_vel_l , ankle_vel_r , knee_vel_l , knee_vel_r , hip_vel_l , hip_vel_r , hipabd_vel_l , hipabd_vel_r]
		plot_states_names = ["ankle_vel_l" , "ankle_vel_r" , "knee_vel_l" , "knee_vel_r" , "hip_vel_l" , "hip_vel_r" , "hipabd_vel_l" , "hipabd_vel_r"]

		fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
		#pdb.set_trace()
		for obs_ind, ax in zip(range(len(plot_states_2)), axes2.flatten()):
			ax.plot(plot_states_2[obs_ind][0][:100])
			ax.plot(plot_states_2[obs_ind][1][:100])
			ax.set_title(plot_states_names[obs_ind], fontsize=10)
		#plt.savefig(plotpath)
		if logger:
			logger["iter_{}/{}".format(iteration,name)].upload(fig2)
		plt.close()

	def compare_observations_drop_approx(self, real_states_dict, predicted_states_dict, predicted_states_std1,
										 predicted_states_std2, path="", header="", called_from=10, iteration=0,
										 logger=None):
		if called_from > 1:
			plotpath = os.path.join(path, "Compare_positions_multi.png")
			name = "Compare_positions_multi.png"
		else:
			plotpath = os.path.join(path, "Compare_positions_step.png")
			name = "Compare_positions_step.png"
		plot_states = []
		pelvis_x = [[], [], [], []]
		pelvis_y = [[], [], [], []]
		pelvis_z = [[], [], [], []]
		ankle_joint_l = [[], [], [], []]
		ankle_joint_r = [[], [], [], []]
		knee_joint_l = [[], [], [], []]
		knee_joint_r = [[], [], [], []]
		hip_joint_l = [[], [], [], []]
		hip_joint_r = [[], [], [], []]
		hip_abd_joint_l = [[], [], [], []]
		hip_abd_joint_r = [[], [], [], []]
		pelvis_pitch = [[], [], [], []]
		pelvis_roll = [[], [], [], []]
		pelvis_rot = [[], [], [], []]
		for index in range(len(real_states_dict)):
			pelvis_x[0].append(real_states_dict[index]["pelvis"]["body_pos"][0])
			pelvis_x[1].append(predicted_states_dict[index]["pelvis"]["body_pos"][0])
			pelvis_y[0].append(real_states_dict[index]["pelvis"]["body_pos"][1])
			pelvis_y[1].append(predicted_states_dict[index]["pelvis"]["body_pos"][1])

			pelvis_z[0].append(real_states_dict[index]["pelvis"]["body_pos"][2])
			pelvis_z[1].append(predicted_states_dict[index]["pelvis"]["body_pos"][2])

			ankle_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['ankle'])
			ankle_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['ankle'])

			ankle_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['ankle'])
			ankle_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['ankle'])

			knee_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['knee'])
			knee_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['knee'])
			knee_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['knee'])
			knee_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['knee'])
			hip_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['hip'])
			hip_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['hip'])
			hip_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['hip'])
			hip_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['hip'])
			hip_abd_joint_l[0].append(real_states_dict[index]['l_leg']['joint']['hip_abd'])
			hip_abd_joint_l[1].append(predicted_states_dict[index]['l_leg']['joint']['hip_abd'])
			hip_abd_joint_r[0].append(real_states_dict[index]['r_leg']['joint']['hip_abd'])
			hip_abd_joint_r[1].append(predicted_states_dict[index]['r_leg']['joint']['hip_abd'])
			pelvis_pitch[0].append(real_states_dict[index]["pelvis"]["pitch"])
			pelvis_pitch[1].append(predicted_states_dict[index]["pelvis"]["pitch"])
			pelvis_roll[0].append(real_states_dict[index]["pelvis"]["roll"])
			pelvis_roll[1].append(predicted_states_dict[index]["pelvis"]["roll"])
			pelvis_rot[0].append(real_states_dict[index]["pelvis"]["joint_pos"][2])
			pelvis_rot[1].append(predicted_states_dict[index]["pelvis"]["joint_pos"][2])

			'''adding std'''
			pelvis_x[2].append(predicted_states_std1[index]["pelvis"]["body_pos"][0])
			pelvis_x[3].append(predicted_states_std2[index]["pelvis"]["body_pos"][0])
			pelvis_y[2].append(predicted_states_std1[index]["pelvis"]["body_pos"][1])
			pelvis_y[3].append(predicted_states_std2[index]["pelvis"]["body_pos"][1])

			pelvis_z[2].append(predicted_states_std1[index]["pelvis"]["body_pos"][2])
			pelvis_z[3].append(predicted_states_std2[index]["pelvis"]["body_pos"][2])

			ankle_joint_l[2].append(predicted_states_std1[index]['l_leg']['joint']['ankle'])
			ankle_joint_l[3].append(predicted_states_std2[index]['l_leg']['joint']['ankle'])

			ankle_joint_r[2].append(predicted_states_std1[index]['r_leg']['joint']['ankle'])
			ankle_joint_r[3].append(predicted_states_std2[index]['r_leg']['joint']['ankle'])

			knee_joint_l[2].append(predicted_states_std1[index]['l_leg']['joint']['knee'])
			knee_joint_l[3].append(predicted_states_std2[index]['l_leg']['joint']['knee'])
			knee_joint_r[2].append(predicted_states_std1[index]['r_leg']['joint']['knee'])
			knee_joint_r[3].append(predicted_states_std2[index]['r_leg']['joint']['knee'])
			hip_joint_l[2].append(predicted_states_std1[index]['l_leg']['joint']['hip'])
			hip_joint_l[3].append(predicted_states_std2[index]['l_leg']['joint']['hip'])
			hip_joint_r[2].append(predicted_states_std1[index]['r_leg']['joint']['hip'])
			hip_joint_r[3].append(predicted_states_std2[index]['r_leg']['joint']['hip'])
			hip_abd_joint_l[2].append(predicted_states_std1[index]['l_leg']['joint']['hip_abd'])
			hip_abd_joint_l[3].append(predicted_states_std2[index]['l_leg']['joint']['hip_abd'])
			hip_abd_joint_r[2].append(predicted_states_std1[index]['r_leg']['joint']['hip_abd'])
			hip_abd_joint_r[3].append(predicted_states_std2[index]['r_leg']['joint']['hip_abd'])
			pelvis_pitch[2].append(predicted_states_std1[index]["pelvis"]["pitch"])
			pelvis_pitch[3].append(predicted_states_std2[index]["pelvis"]["pitch"])
			pelvis_roll[2].append(predicted_states_std1[index]["pelvis"]["roll"])
			pelvis_roll[3].append(predicted_states_std2[index]["pelvis"]["roll"])
			pelvis_rot[2].append(predicted_states_std1[index]["pelvis"]["joint_pos"][2])
			pelvis_rot[3].append(predicted_states_std2[index]["pelvis"]["joint_pos"][2])
		plot_states_1 = [pelvis_x, pelvis_y, pelvis_z, ankle_joint_l, ankle_joint_r, knee_joint_l, knee_joint_r,
						 hip_joint_l, \
						 hip_joint_r, hip_abd_joint_l, hip_abd_joint_r, pelvis_pitch, pelvis_roll, pelvis_rot]
		plot_states_names = ["pelvis_x", "pelvis_y", "pelvis_z", "ankle_joint_l", "ankle_joint_r", "knee_joint_l",
							 "knee_joint_r", "hip_joint_l", \
							 "hip_joint_r", "hip_abd_joint_l", "hip_abd_joint_r", "pelvis_pitch", "pelvis_roll",
							 "pelvis_rot"]
		fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(16, 8))
		# pdb.set_trace()
		for obs_ind, ax in zip(range(len(plot_states_1)), axes.flatten()):
			ax.plot(plot_states_1[obs_ind][0][:100])
			ax.plot(plot_states_1[obs_ind][1][:100], linestyle='--')
			ax.fill_between(range(0, 100), plot_states_1[obs_ind][2][:100], plot_states_1[obs_ind][3][:100])
			ax.set_title(plot_states_names[obs_ind], fontsize=10)
		if logger:
			logger["iter_{}/{}".format(iteration, name)].upload(fig)
		# plt.savefig(plotpath)
		plt.close()
		if called_from > 1:
			plotpath = os.path.join(path, "Compare_velocity_multi.pdf")
			name = "Compare_velocity_multi"
		else:
			plotpath = os.path.join(path, "Compare_velocity_step.pdf")
			name = "Compare_velocity_step"
		ankle_vel_l = [[], [], [], []]
		ankle_vel_r = [[], [], [], []]
		knee_vel_l = [[], [], [], []]
		knee_vel_r = [[], [], [], []]
		hip_vel_l = [[], [], [], []]
		hip_vel_r = [[], [], [], []]
		hipabd_vel_l = [[], [], [], []]
		hipabd_vel_r = [[], [], [], []]
		for index in range(len(real_states_dict)):
			ankle_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['ankle'])
			ankle_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['ankle'])
			ankle_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['ankle'])
			ankle_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['ankle'])
			knee_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['knee'])
			knee_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['knee'])
			knee_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['knee'])
			knee_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['knee'])
			hip_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['hip'])
			hip_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['hip'])
			hip_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['hip'])
			hip_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['hip'])
			hipabd_vel_l[0].append(real_states_dict[index]['l_leg']['d_joint']['hip_abd'])
			hipabd_vel_l[1].append(predicted_states_dict[index]['l_leg']['d_joint']['hip_abd'])
			hipabd_vel_r[0].append(real_states_dict[index]['r_leg']['d_joint']['hip_abd'])
			hipabd_vel_r[1].append(predicted_states_dict[index]['r_leg']['d_joint']['hip_abd'])
			'''add std'''
			ankle_vel_l[2].append(predicted_states_std1[index]['l_leg']['d_joint']['ankle'])
			ankle_vel_l[3].append(predicted_states_std2[index]['l_leg']['d_joint']['ankle'])
			ankle_vel_r[2].append(predicted_states_std1[index]['r_leg']['d_joint']['ankle'])
			ankle_vel_r[3].append(predicted_states_std2[index]['r_leg']['d_joint']['ankle'])
			knee_vel_l[2].append(predicted_states_std1[index]['l_leg']['d_joint']['knee'])
			knee_vel_l[3].append(predicted_states_std2[index]['l_leg']['d_joint']['knee'])
			knee_vel_r[2].append(predicted_states_std1[index]['r_leg']['d_joint']['knee'])
			knee_vel_r[3].append(predicted_states_std2[index]['r_leg']['d_joint']['knee'])
			hip_vel_l[2].append(predicted_states_std1[index]['l_leg']['d_joint']['hip'])
			hip_vel_l[3].append(predicted_states_std2[index]['l_leg']['d_joint']['hip'])
			hip_vel_r[2].append(predicted_states_std1[index]['r_leg']['d_joint']['hip'])
			hip_vel_r[3].append(predicted_states_std2[index]['r_leg']['d_joint']['hip'])
			hipabd_vel_l[2].append(predicted_states_std1[index]['l_leg']['d_joint']['hip_abd'])
			hipabd_vel_l[3].append(predicted_states_std2[index]['l_leg']['d_joint']['hip_abd'])
			hipabd_vel_r[2].append(predicted_states_std1[index]['r_leg']['d_joint']['hip_abd'])
			hipabd_vel_r[3].append(predicted_states_std2[index]['r_leg']['d_joint']['hip_abd'])
		plot_states_2 = [ankle_vel_l, ankle_vel_r, knee_vel_l, knee_vel_r, hip_vel_l, hip_vel_r, hipabd_vel_l,
						 hipabd_vel_r]
		plot_states_names = ["ankle_vel_l", "ankle_vel_r", "knee_vel_l", "knee_vel_r", "hip_vel_l", "hip_vel_r",
							 "hipabd_vel_l", "hipabd_vel_r"]

		fig2, axes2 = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))
		# pdb.set_trace()
		for obs_ind, ax in zip(range(len(plot_states_2)), axes2.flatten()):
			ax.plot(plot_states_2[obs_ind][0][:100])
			ax.plot(plot_states_2[obs_ind][1][:100])
			ax.fill_between(range(0, 100), plot_states_2[obs_ind][2][:100], plot_states_2[obs_ind][3][:100])
			ax.set_title(plot_states_names[obs_ind], fontsize=10)
		# plt.savefig(plotpath)
		if logger:
			logger["iter_{}/{}".format(iteration, name)].upload(fig2)
		plt.close()

	def compare_rew(self,rew_real,rew_predicted,header="", path="", called_from = 10, iteration = 0, logger = None):
		if called_from>1:
			plotpath = os.path.join(path, "Rewards_multistep.pdf")
			name = "Rewards_multistep"
		else:
			plotpath = os.path.join(path, "Rewards_step.pdf")
			name = "Rewards_step"
		fig = plt.figure()
		plt.plot(rew_real)
		plt.plot(rew_predicted)
		plt.title("Rewards")
		#plt.savefig(plotpath)
		if logger:
			logger["iter_{}/{}".format(iteration,name)].upload(fig)
		plt.close()

	def compare_done(self,done_real,done_predicted,header="", path="", called_from = 10, iteration = 0, logger = None):
		if called_from>1:
			plotpath = os.path.join(path, "Termination_multistep.pdf")
			name = "Termination_multistep"
		else:
			plotpath = os.path.join(path, "Termination_step.pdf")
			name = "Termination_step"
		fig = plt.figure()
		plt.plot(done_real)
		plt.plot(done_predicted)
		plt.title("Termination")
		#plt.savefig(plotpath)
		if logger:
			logger["iter_{}/{}".format(iteration,name)].upload(fig)
		plt.close()


	def data_explain(self, States = [], predicted_states = [], header = [], path =""):
		var = []
		var = np.std(States, axis = 0)
		mean = np.abs(np.mean(States, axis = 0))
		var = var/mean  #normalized variance
		plt.plot(var)
		plt.title("Normalized variance over dimensions")
		print(max(var))
		plt.xticks(ticks = range(84), labels = header[2:], rotation='vertical')
		#plt.show()
		
		plt.clf()
		minNum = np.amin(States, axis = 0)
		maxNum = np.amax(States, axis = 0)
		fig, axs = plt.subplots(10,9, figsize=(15, 6), facecolor='w', edgecolor='k')
		axis = axs.ravel()
		for col in range(2,States.shape[1]):
			axis[col-2].plot(States[:,col])
			axis[col-2].set_title(header[col])
		pdb.set_trace()
		fig.show()
		plt.clf()


def main():

	predicted = []
	real = []
	header = []
	ploting =DynamicPlotting()
	random = False
	if random == True:
		for i in range(20):
			predicted.append(np.random.rand(94).tolist())
			real.append(np.random.rand(94).tolist())
	else :	
		myfile = open("/home/anjali/Projects/rug-opensim-rl/previous-work/Aurelien_Adriaenssens_BSc-02-09-21/OpenSim-master/osim-rl/osim/data/175-FIX.csv")
		csvreader = csv.reader(myfile)
		header = next(csvreader)
		rows = []
		for row in csvreader:
			rows.append(row[2:])
		real = np.array(rows, dtype=float).reshape(-1,len(rows[0]))
		
	#ploting.plotDifferences(real,predicted)
	ploting.data_explain(States = real, header = header)


		
if __name__ == "__main__":
	main()


