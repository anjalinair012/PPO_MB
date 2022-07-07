import pdb
import time
import os
from dataReader import fileRead
import tensorflow as tf
from Ensemble import Ensemble
from baselines.common.mpi_running_mean_std import RunningMeanStd
import  numpy as np
from ReplayBuffer import ReplayMemoryFast
import baselines.common.tf_util as U
import matplotlib.pyplot as plt
from operator import sub

from matplotlib.pyplot import sca
from sklearn.preprocessing import StandardScaler
from utils import MinMaxScalerNew,Scaler,ManualScaler
import copy

'''https://www.analyticsvidhya.com/blog/2021/06/the-challenge-of-vanishing-exploding-gradients-in-deep-neural-networks/'''

class ModelBase:

    def __init__(self, env ="", observation_size = 94,activation_d="relu", activation_op="relu", epochs=20, nlayers=5, mb_max_steps=100,
                 capacity = 10000, batch_size = 64, networkUnits = 200, nmembers = 1, sample_ratio = 0.5, mb_ensemble = False, mb_scaler = "RuningMean",
                 logger = None):
        """create a model, replay buffer"""
        seed = 0
        self.env = env
        self.state_size = observation_size #env.get_observation_space_size()
        self.action_size = env.get_action_space_size()
        self.input_size = self.state_size + self.action_size
        self.stepsize = env.stepsize
        self.scale = mb_scaler
        with tf.variable_scope("Input_scaler"):
            if self.scale == "RuningMean":
                self.mb_rms = RunningMeanStd(shape=(self.input_size,))
            elif self.scale == "RuningMinMax":
                self.mb_rms = MinMaxScalerNew()
            elif self.scale == "StandardMinMax":
                self.mb_rms = ManualScaler(obs_dim=self.input_size)
        with tf.variable_scope("Output_scaler"):
            if self.scale == "RuningMean":
                self.op_mb_rms = RunningMeanStd(shape=(self.state_size,))
            elif self.scale == "RuningMinMax":
                self.op_mb_rms = MinMaxScalerNew()
            elif self.scale == "StandardMinMax":
                self.op_mb_rms = ManualScaler(obs_dim=self.state_size)
        self.horizon_len = mb_max_steps #not used
        self.batch_size = batch_size
        self.buffer_capacity = capacity

        self.replay_buffer_Drand = ReplayMemoryFast(memory_size=capacity,
                                       batch_size=batch_size,state_size=self.state_size,params_passed = 6)  # actual capacity = capacity*min_size
        self.replay_buffer_Drl = ReplayMemoryFast(memory_size=capacity,
                                       batch_size=batch_size,state_size=self.state_size,params_passed = 6)
        self.eval_buffer = ReplayMemoryFast(memory_size=1000,
                                       batch_size=batch_size,state_size=self.state_size,params_passed = 6)  # actual capacity = capacity*min_size
        self.epoch = epochs
        self.sample_ratio = sample_ratio
        self.logger = logger
        self._init(activation_d, activation_op, networkUnits, nlayers, nmembers, mb_ensemble)


    def _init(self, activation_d, activation_op, networkUnits, nlayers, nmembers, mb_ensemble):
        self.dyn_model = Ensemble(self.replay_buffer_Drand,self.replay_buffer_Drl,self.eval_buffer,
                                  self.mb_rms, self.op_mb_rms, self.scale,
                                  self.input_size, self.state_size, activation_d, activation_op, networkUnits,
                                 nlayers= nlayers, nmembers = nmembers, mb_ensemble = mb_ensemble)

        self.update_dataset(load_from_file = False)


    def restore_state(self, load_iters_mb, aggregate_every_iter):
        print("---restore model------")
        for index,member in enumerate(self.dyn_model.members):
            member.load_weights('best_model_{}'.format(str(index)))
        self.dyn_model.lr = self.dyn_model.lr * np.power(0.98, int(load_iters_mb/aggregate_every_iter)+1)
        self.dyn_model.log_counter += (int(load_iters_mb/aggregate_every_iter)+1)*10
        self.sample_ratio += 0.2*(int(load_iters_mb/aggregate_every_iter)+1)
        print("----restore buffers-----")
        self.update_dataset(load_from_file = True)

    def update_dataset(self,Observations = [], Actions = [], Next_Observations = [], Timesteps = [], end_states = [], Dones = [], load_from_file = False):
        buffer_overhead = time.time()
        if load_from_file:
            # self.replay_buffer_Drl.load_from_file("datasets/dataset_rl.npy", short_obs = False)
            # self.replay_buffer_Drand.load_from_file("datasets/best_trajectories.npy",short_obs = False)
            # self.eval_buffer.load_from_file("datasets/dataset_eval.npy", short_obs = False)
            self.mb_rms.load_from_file("Scalar_input.npy")
            self.op_mb_rms.load_from_file("Scalar_output.npy")
        else:
            if len(Observations) == 0:
                eval_index = 500
                Add_Rand = True
                self.replay_buffer_Drand.load_from_file("datasets/best_trajectories.npy", restore_upto=-eval_index)
                #Observations, Actions, Next_Observations, Diff_Observations, Inputs_temp, Timesteps = zip(*self.replay_buffer_Drand.experience)
                Observations, Actions, Next_Observations, Diff_Observations, Inputs_temp, Timesteps, Dones = self.replay_buffer_Drand.sample(replace=False)
            else:
                Add_Rand = False
                if isinstance(Observations, np.ndarray):
                    Diff_Observations = np.subtract(Next_Observations, Observations)
                    Inputs_temp = np.concatenate([Observations, Actions], axis=1)
                    Observations = Observations.tolist()
                    Actions = Actions.tolist()
                    Next_Observations = Next_Observations.tolist()
                    Timesteps = Timesteps.tolist()
                    Dones = Dones.tolist()
                    # Diff_Observations = np.delete(np.subtract(Next_Observations, Observations),end_states,axis=0)
                    # Inputs_temp = np.delete(np.concatenate([Observations, Actions], axis=1),end_states,axis=0)
                    # Observations = np.delete(Observations,end_states,axis=0).tolist()
                    # Actions = np.delete(Actions,end_states,axis=0).tolist()
                    # Next_Observations = np.delete(Next_Observations,end_states,axis = 0).tolist()
                    # Timesteps = np.delete(Timesteps,end_states,axis = 0).tolist()
                eval_index = int(len(Observations) * 0.20)
            '''Update scalar'''
            self.mb_rms.update(np.array(Inputs_temp[:-eval_index]))
            self.op_mb_rms.update(np.array(Diff_Observations[:-eval_index]))
            '''Save scalar'''
            self.mb_rms.save_to_file("Scalar_input.npy")
            self.op_mb_rms.save_to_file("Scalar_output.npy")


            self.eval_buffer.store(observation = Observations[-eval_index:],action = Actions[-eval_index:],next_observation = Next_Observations[-eval_index:],diff_observation = list(Diff_Observations[-eval_index:]), scaled_inputs = list(Inputs_temp[-eval_index:]), timesteps = Timesteps[-eval_index:], dones = Dones[-eval_index:])
            filename = "datasets/dataset_eval.npy"
            self.eval_buffer.save_to_file(filename)
            if Add_Rand:
                filename = "datasets/dataset_rand_mod.npy"
                self.replay_buffer_Drand.save_to_file(filename)
            else:
                #self.replay_buffer_Drl.clear()
                self.replay_buffer_Drl.store(observation = Observations[:-eval_index],action = Actions[:-eval_index],next_observation = Next_Observations[:-eval_index],diff_observation = list(Diff_Observations[:-eval_index]), scaled_inputs = list(Inputs_temp[:-eval_index]), timesteps = Timesteps[:-eval_index], dones = Dones[:-eval_index]) # stores converts to tensors
                filename = "datasets/dataset_rl.npy"
                self.replay_buffer_Drl.save_to_file(filename)
        buffer_overhead = time.time() - buffer_overhead
        m = open("Times.txt", "a+")
        m.write("Buffer Overhead %d\n" % buffer_overhead)
        m.close()
        #self.replay_buffer_Drl.clear()


    def train(self, epochs = None,minibatch_size = None, sample_size = 30000, load_iters_mb = None):
        if not epochs:
            epochs=self.epoch
        if not minibatch_size:
            minibatch_size = self.batch_size
        if self.epoch <= 0:
            print("Model trained")
            return
        self.dyn_model.train(epochs=epochs, batch_size=minibatch_size, prop_rl = self.sample_ratio, sample_size = sample_size, logger=self.logger)
        #self.dyn_model.train_doneModel(sample_size=int(sample_size/3),prop_rl=self.sample_ratio,replace=False, epochs=int(epochs/2), load_iters_mb = load_iters_mb, logger=self.logger)
        self.sample_ratio = min(1,self.sample_ratio+0.2)
        if self.sample_ratio == 1:
            self.replay_buffer_Drand.clear()

import pandas as pd
class LearnedEnvironment():
    #Used only for 94 obs space
    dict_muscle = {'abd_r': 'HAB_R',
                   'add_r': 'HAD_R',
                   'hamstrings_r': 'HAM_R',
                   'bifemsh_r': 'BFSH_R',
                   'glut_max_r': 'GLU_R',
                   'iliopsoas_r': 'HFL_R',
                   'rect_fem_r': 'RF_R',
                   'vasti_r': 'VAS_R',
                   'gastroc_r': 'GAS_R',
                   'soleus_r': 'SOL_R',
                   'tib_ant_r': 'TA_R',
                   'abd_l': 'HAB_L',
                   'add_l': 'HAD_L',
                   'glut_max_l': 'GLU_L',
                   'iliopsoas_l': 'HFL_L'}

    right_leg_MUS = ['HAB_R', 'HAD_R', 'HAM_R', 'BFSH_R', 'GLU_R', 'HFL_R', 'RF_R', 'VAS_R', 'GAS_R', 'SOL_R',
                     'TA_R']  # 11 muscles
    right_leg_mus = ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r',
                     'vasti_r', 'gastroc_r', 'soleus_r', 'tib_ant_r']

    left_leg_MUS = ['HAB_L', 'HAD_L', 'GLU_L', 'HFL_L']  # 4 muscles
    left_leg_mus = ['abd_l', 'add_l', 'glut_max_l', 'iliopsoas_l']

    act2mus = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 15 muscles + 2 Actuators
    dict_actuator = {'knee_actuator': "KNE_ACT",
                     'ankle_actuator': "ANK_ACT"}
    def __init__(self, mbAgent:ModelBase, difficulty:int):
        self.state_size = mbAgent.state_size
        self.action_size = mbAgent.action_size
        self.input_size = mbAgent.input_size
        self.model = copy.copy(mbAgent.dyn_model)
        self.state = self.reset()
        self.prev_observation_dic = None
        self.scale = mbAgent.scale
        self.difficulty = difficulty
        if mbAgent.scale == "RuningMean":
            self.Scalarmean = mbAgent.mb_rms.mean.eval()
            self.Scalarstd = mbAgent.mb_rms.std.eval()
            self.Scalarmean_op = mbAgent.op_mb_rms.mean.eval()
            self.Scalarstd_op = mbAgent.op_mb_rms.std.eval()
            self.ScalarIn = 0
            self.ScalarOut = 0
        elif mbAgent.scale == "RuningMinMax":
            self.Scalarmean = 0
            self.Scalarstd = 0
            self.Scalarmean_op = 0
            self.Scalarstd_op = 0
            self.ScalarIn = mbAgent.mb_rms
            self.ScalarOut = mbAgent.op_mb_rms
        elif mbAgent.scale == "StandardMinMax":
            self.Scalarmean = 0
            self.Scalarstd = 0
            self.Scalarmean_op = 0
            self.Scalarstd_op = 0
            self.ScalarIn = mbAgent.mb_rms
            self.ScalarOut = mbAgent.op_mb_rms
        self.t = 0
        self.stepsize = mbAgent.stepsize
        self.start_timestep = 0
        if self.scale == "RuningMean":
            action = tf.placeholder(shape = (self.action_size,), dtype=tf.float32, name = "action")
            mean = tf.placeholder(shape = self.Scalarmean.shape, dtype= tf.float32, name = "mean")
            std = tf.placeholder(shape = self.Scalarstd.shape, dtype = tf.float32, name = "std")
            state_tf = tf.placeholder(shape = (self.state_size,), dtype=tf.float32, name = "state")
            mean_op = tf.placeholder(shape = self.Scalarmean_op.shape, dtype= tf.float32, name = "mean")
            std_op = tf.placeholder(shape = self.Scalarstd_op.shape, dtype = tf.float32, name = "std")

            inputCalc = (tf.concat([state_tf, action], axis = -1) - self.Scalarmean)/self.Scalarstd
            self.input = U.function([state_tf,action, mean, std],inputCalc)
            output = tf.placeholder(shape = (self.state_size,), dtype=tf.float32, name = "diff")
            outputCalc = ((output * self.Scalarstd_op) + self.Scalarmean_op) + state_tf
            self.output = U.function([output, std_op, mean_op, state_tf],outputCalc)
        self.im_file = pd.read_csv("../EDITED-MATERIAL/175-FIX.csv", index_col=False)
        self.logger = mbAgent.logger



    def update_model(self, model, mb_rms, op_mb_rms, env = None, pi = None, evaluate = True, iteration_num = 0):
        path = "Results/iter_{}".format(iteration_num)
        try:
            os.mkdir(path)
        except OSError as error:
            print(error)
        if self.scale == "RuningMean":
            self.Scalarmean = mb_rms.mean.eval()
            self.Scalarstd = mb_rms.std.eval()
            self.Scalarmean_op = op_mb_rms.mean.eval()
            self.Scalarstd_op = op_mb_rms.std.eval()
        elif self.scale == "RuningMinMax":
            self.ScalarIn = mb_rms  #not required
            self.ScalarOut = op_mb_rms
        elif self.scale == "StandardMinMax":
            self.ScalarIn = mb_rms  #not required
            self.ScalarOut = op_mb_rms
        self.model = copy.copy(model)
        if evaluate:
            evaluate_model(env, self, 100, pi, path, iteration_num,self.state_size,self.logger)
            #state_evaluate_step(env, self, 100, pi, path, iteration_num,self.state_size,self.logger)


    def reset(self, state = [], t = -1):
        if state:
            self.state = state.copy()
        else:
            self.state = [0.93819803, -0.016427413, -0.005727512, 0.005555556, 0.93819803, 0.0, 0.0, 0.0, 0.0,
                          -0.016427413, -0.005727512, -0.003349662, 0.0, 0.0, 0.0, -3.365470066940035e-15,
                          -0.9639089373037147, 0.0, -0.01329794, 0.01204713, 0.0, -0.048406658, 0.0,
                          0.0, 0.0, 0.0, 0.0491786478971494, 0.9136660096887603, 1.3250152566044873e-12,
                          0.037889056926482326, 0.6467121526332598, 1.3231004358222596e-09, 0.04948033795645704,
                          0.9314330072530691, 3.834210532631098e-13, 0.08095135093985227, 1.2212967892198032,
                          8.674726349691641e-16, 0.0516825036160073, 1.027797021393286, 7.488036070204619e-12,
                          0.058627890420524464, 1.1080500915288654, 2.349806210194922e-10, 0.04509295458102523,
                          0.7843993391074274, 3.3313048745194203e-14, 0.04562651795800594, 0.7970461488540335,
                          6.21086383334892e-14, 0.056243573265007986, 1.0855050067611438, 3.0314948500868133e-10,
                          0.04994405857381857, 0.9775555197035398, 5.392349507346414e-15, 0.04979380244638792,
                          0.9568767451801227, 1.8358062548732654e-12, 0.0, 0.0, 0.0, -0.25020487, -0.73940182,
                          0.007071308, -0.0872665, 0.0, 0.0, 0.0, 0.0, 0.03768536831405808, 0.6432956487470146,
                          1.5267632502591518e-09, 0.04981366476746681, 0.9590103701671777, 1.8350141876472574e-12,
                          0.046165018517446525, 0.8104998651971789, 1.2827798676114216e-13, 0.12246622816727983,
                          1.3086070566910402, 9.391361392859058e-09, 300.0, 0.0, 1.0, 0.0, 0.01, 3.0, 300.0,
                          0.0, 1.0, 0.0, 0.01, 3.0]
        # else:#change to env reset state
        #     self.state = [0.93819803, -0.016427413, -0.005727512, 0.005555556, 0.93819803, 0.0, 0.0, 0.0, 0.0, -0.016427413, -0.005727512, -0.003349662, 0.0, 0.0, 0.0,
        #      -3.365470066940035e-15, -0.9639089373037147, 0.0, -0.01329794, 0.01204713, 0.0, -0.048406658, 0.0, 0.0, 0.0, 0.0,
        #      0.0, 0.0, 0.0, -0.25020487, -0.73940182, 0.007071308, -0.0872665, 0.0, 0.0, 0.0, 0.0,
        #      300.0, 0.0, 1.0, 0.0, 0.01, 3.0, 300.0, 0.0, 1.0, 0.0, 0.01, 3.0]
        if self.state_size < len(self.state): #for 49
            self.state = self.state[:26] + self.state[59:70] + self.state[82:]
        if t != -1:
            self.t = t
            self.start_timestep = t
        else:
            self.t = 0
            self.start_timestep = 0
        self.prev_observation_dic = self.get_state_dict(self.state)
        return self.state

    def is_done(self, state_desc = None, scaled_state= None, use_model = False):
        if use_model:
            done = self.model.predict_done(scaled_state)
            return done
        # To check if the model has fallen down or not
        if state_desc['pelvis']['height'] < 0.6:
            done = True  # the model has fallen
        else:
            done = False  # the model is standing tall
        return done

    def scale_features(self, x):
        if self.scale == "StandardMinMax":
            Input = self.ScalarIn.process(x)
            return Input
        elif self.scale == "RuningMinMax" :
            Input = self.ScalarIn.process(x.reshape(-1,x.shape[0]))
            diff, diff_std = self.model.predict(Input.reshape(-1,self.input_size))
            self.state = (np.array(self.state) + self.ScalarOut.inverse_process(diff)[0]).tolist()
        elif self.scale == "RuningMean":
            Input = (x - self.Scalarmean) / self.Scalarstd
            diff, diff_std = self.model.predict(Input.reshape(-1, self.input_size))
            # self.state = list(self.output(self.state,diff[0],self.Scalarstd_op,self.Scalarmean_op))
            self.state = list((diff * self.Scalarstd_op + self.Scalarmean_op)[0] + np.array(self.state))

    def step_eval(self,act):
        Input = np.concatenate([self.state, act])
        '''prepare prediction'''
        Input = self.scale_features(Input)  # self.state updated here
        diff, diff_std = self.model.predict(Input.reshape(-1, self.input_size), for_eval = True)
        diff_std = diff_std * 2
        std_1 = diff + diff_std
        std_2 = diff - diff_std
        std_state_1 = (self.ScalarOut.inverse_process(std_1) + np.array(self.state)).tolist()[0]
        std_state_2 = (self.ScalarOut.inverse_process(std_2) + np.array(self.state)).tolist()[0]
        self.state, reward, penalty, done = self.step(act)
        return self.state, std_state_1, std_state_2, reward, penalty,done

    def step(self,act):
        try:
            self.prev_observation_dic = self.get_state_dict(self.state)
            Input = np.concatenate([self.state, act])
            '''prepare prediction'''
            Input = self.scale_features(Input) #self.state updated here
            diff, _ = self.model.predict(Input.reshape(-1,self.input_size))
            self.state = (self.ScalarOut.inverse_process(diff) + np.array(self.state)).tolist()[0]
            next_state = self.get_state_dict(self.state)
            self.t += self.stepsize
            # if int(self.t*200) >= len(self.im_file)-10:
            #     print("file traversed")
            #     reward, penalty, _ = (0,0,False)
            #     done = True
            #     return self.state, reward, penalty, done
            # else:
            reward, penalty, _ = self.get_reward(next_state)
            if int(self.t*100) >= 500:
                done = True
            scaled_nxt_state = Input[:-17] + diff
            done = self.is_done(state_desc=next_state,scaled_state= scaled_nxt_state, use_model = False)
        except:
            pdb.set_trace()
        return self.state, reward, penalty, done

    def get_state_dict(self, state):
        ## Values in the observation vector
        # 'pelvis': height(1), pitch(1), roll(1),
            #           body_pos(3), body_vel(3),
            #           joint_pos(3), joint_vel(3) (total: 15 values)
        # for each 'r_leg' and 'l_leg' (*2)
        #   'ground_reaction_forces' (3 values)
        #   'joint' (4 values)
        #   'd_joint' (4 values)  --- 37
        # for each of the 15 muscles (*15)
        #    normalized 'f', 'l', 'v' (3 values)
        # actuators for knee and ankle (*2)
        #       force - instantaneous (1 value)
        #       speed (1 value)
        #       control (1 value)
        #       actuation (1 value)
        #       power (1 value)
        #       activation (1 value)
        # TOTAL = 94
        '''
        state['pelvis']:
         height(1), pitch(1), roll(1),
        #           body_pos(3), body_vel(3),
        #           joint_pos(3), joint_vel(3) (total: 15 values)
        state['r_leg']['ground_reaction_forces']  'ground_reaction_forces' (3 values)
        state['r_leg']['joint'] 'joint' (4 values)
        state['r_leg']['d_joint'] 'd_joint' (4 values)
        state['l_leg']['ground_reaction_forces']  'ground_reaction_forces' (3 values)
        state['l_leg']['joint'] 'joint' (4 values)
        state['l_leg']['d_joint'] 'd_joint' (4 values)
        state['l_leg']['actuator']:
           # actuators for knee and ankle (*2)
        #       force - instantaneous (1 value)
        #       speed (1 value)
        #       control (1 value)
        #       actuation (1 value)
        #       power (1 value)
        #       activation (1 value)
        '''
        obs_dict = {}
        counter = 0
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['height'] = state[counter]  # 1 value
        counter += 1
        obs_dict['pelvis']['pitch'] = state[counter]  # 1 value
        counter += 1
        obs_dict['pelvis']['roll'] = state[counter]  # 1 value
        counter += 1
        for pelvis_val in ['body_pos', 'body_vel', 'joint_pos', 'joint_vel']:  # 3*4
            obs_dict['pelvis'][pelvis_val] = state[counter:counter + 3]
            counter += 3

        for leg in ['r_leg', 'l_leg']:
            obs_dict[leg] = {}
            obs_dict[leg]['ground_reaction_forces'] = state[counter:counter + 3]
            counter += 3
            # joint angles
            obs_dict[leg]['joint'] = {}
            obs_dict[leg]['joint']['hip_abd'] = state[counter]
            counter += 1
            obs_dict[leg]['joint']['hip'] = state[counter]
            counter += 1
            obs_dict[leg]['joint']['knee'] = state[counter]
            counter += 1
            obs_dict[leg]['joint']['ankle'] = state[counter]
            counter += 1
            # joint angular velocities
            obs_dict[leg]['d_joint'] = {}
            obs_dict[leg]['d_joint']['hip_abd'] = state[counter]
            counter += 1
            obs_dict[leg]['d_joint']['hip'] = state[counter]
            counter += 1
            obs_dict[leg]['d_joint']['knee'] = state[counter]
            counter += 1
            obs_dict[leg]['d_joint']['ankle'] = state[counter]
            counter += 1
            if self.state_size == 94:
                if leg == 'r_leg':
                    muscle_list = self.right_leg_MUS
                else:
                    muscle_list = self.left_leg_MUS
                for MUS in muscle_list:
                    obs_dict[leg][MUS] = {}
                    obs_dict[leg][MUS]['f'] = state[counter]
                    counter +=1
                    obs_dict[leg][MUS]['l'] = state[counter]
                    counter += 1
                    obs_dict[leg][MUS]['v'] = state[counter]
                    counter += 1

            # Actuator states
        obs_dict['l_leg']['force'] = {}
        obs_dict['l_leg']['actuator'] = {}
        obs_dict['l_leg']['actuator']['knee'] = {}
        obs_dict['l_leg']['actuator']['ankle'] = {}
        obs_dict['l_leg']['force']['knee'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['knee']['speed'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['knee']['control'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['knee']['power'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['knee']['activation'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['knee']['actuation'] = state[counter]
        counter += 1

        obs_dict['l_leg']['force']['ankle'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['ankle']['speed'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['ankle']['control'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['ankle']['power'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['ankle']['activation'] = state[counter]
        counter += 1
        obs_dict['l_leg']['actuator']['ankle']['actuation'] = state[counter]
        return obs_dict

    def get_observation(self,state):
        obs_dict = self.get_state_dict(state)

        # Augmented environment from the L2R challenge

        res.append(obs_dict['pelvis']['height'])  # 1 value
        res.append(obs_dict['pelvis']['pitch'])  # 1 value
        res.append(obs_dict['pelvis']['roll'])  # 1 value
        res.extend(obs_dict['pelvis']['body_pos'])  # 3 values
        res.extend(obs_dict['pelvis']['body_vel'])  # 3 values
        res.extend(obs_dict['pelvis']['joint_pos'])  # 3 values
        res.extend(obs_dict['pelvis']['joint_vel'])  # 3 values

        for leg in ['r_leg', 'l_leg']:
            res += obs_dict[leg]['ground_reaction_forces']
            res.append(obs_dict[leg]['joint']['hip_abd'])
            res.append(obs_dict[leg]['joint']['hip'])
            res.append(obs_dict[leg]['joint']['knee'])
            res.append(obs_dict[leg]['joint']['ankle'])
            res.append(obs_dict[leg]['d_joint']['hip_abd'])
            res.append(obs_dict[leg]['d_joint']['hip'])
            res.append(obs_dict[leg]['d_joint']['knee'])
            res.append(obs_dict[leg]['d_joint']['ankle'])
            if self.state_size == 94:
                if leg == 'r_leg':
                    muscle_list = self.right_leg_MUS
                else:
                    muscle_list = self.left_leg_MUS
                for MUS in muscle_list:
                    res.append(obs_dict[leg][MUS]['f'])
                    res.append(obs_dict[leg][MUS]['l'])
                    res.append(obs_dict[leg][MUS]['v'])
        # Actuator states
        res.append(obs_dict['l_leg']['force']['knee'])
        res.append(obs_dict['l_leg']['actuator']['knee']['speed'])
        res.append(obs_dict['l_leg']['actuator']['knee']['control'])
        res.append(obs_dict['l_leg']['actuator']['knee']['power'])
        res.append(obs_dict['l_leg']['actuator']['knee']['activation'])
        res.append(obs_dict['l_leg']['actuator']['knee']['actuation'])

        res.append(obs_dict['l_leg']['force']['ankle'])
        res.append(obs_dict['l_leg']['actuator']['ankle']['speed'])
        res.append(obs_dict['l_leg']['actuator']['ankle']['control'])
        res.append(obs_dict['l_leg']['actuator']['ankle']['power'])
        res.append(obs_dict['l_leg']['actuator']['ankle']['activation'])
        res.append(obs_dict['l_leg']['actuator']['ankle']['actuation'])
        return res

    def compute_imitation_reward(self, t, im_file, state_desc):
        NotImplemented

    def get_reward(self, state_desc):
        if self.difficulty == 1:
            return self.get_reward_1(state_desc)
        elif self.difficulty == 2:
            return self.get_reward_2(state_desc)

    # def get_reward_1(self, state_desc):
    #     im_file = self.im_file
    #
    #     t = round(self.t * 200)
    #     #
    #     penalty = 0
    #     x_penalty = (state_desc["pelvis"]["body_pos"][0] - im_file['pelvis_tx'][t]) ** 2
    #     y_penalty = (state_desc["pelvis"]["body_pos"][1] - im_file['pelvis_ty'][t]) ** 2
    #     z_penalty = (state_desc["pelvis"]["body_pos"][2] - im_file['pelvis_tz'][t]) ** 2
    #     penalty += (x_penalty + y_penalty + z_penalty)
    #     #penalty += np.sum(np.array(self.get_activations()) ** 2) * 0.001  # reduce penalty of energy used.
    #
    #     goal_rew = np.exp(-8 * (x_penalty + y_penalty + z_penalty))
    #
    #     ankle_loss = ((state_desc['l_leg']['joint']['ankle'] - im_file['ankle_angle_l'][t]) ** 2 +
    #                   (state_desc['r_leg']['joint']['ankle'] - im_file['ankle_angle_r'][t]) ** 2)
    #     knee_loss = ((state_desc['l_leg']['joint']['knee'] - im_file['knee_angle_l'][t]) ** 2 +
    #                  (state_desc['r_leg']['joint']['knee'] - im_file['knee_angle_r'][t]) ** 2)
    #     hip_loss = ((state_desc["l_leg"]['joint']['hip'] - im_file['hip_flexion_l'][t]) ** 2 +
    #                 (state_desc['r_leg']['joint']['hip'] - im_file['hip_flexion_r'][t]) ** 2 +
    #                 (state_desc["l_leg"]['joint']['hip_abd'] - im_file['hip_adduction_l'][t]) ** 2 +
    #                 (state_desc['r_leg']['joint']['hip_abd'] - im_file['hip_adduction_r'][t]) ** 2)
    #     pelvis_angle_loss = ((state_desc["pelvis"]["pitch"] - im_file['pelvis_tilt'][t]) ** 2 +
    #                          (state_desc["pelvis"]["roll"] - im_file['pelvis_list'][t]) ** 2 +
    #                          (state_desc["pelvis"]["joint_pos"][2] - im_file['pelvis_rotation'][t]) ** 2)
    #     pelvis_pos_loss = ((state_desc['pelvis']['body_pos'][0] - im_file['pelvis_tx'][t]) ** 2 +
    #                        (state_desc['pelvis']['body_pos'][1] - im_file['pelvis_ty'][t]) ** 2 +
    #                        (state_desc['pelvis']['body_pos'][2] - im_file['pelvis_tz'][t]) ** 2)
    #
    #     total_position_loss = ankle_loss + knee_loss + hip_loss + pelvis_angle_loss
    #     pos_reward = np.exp(-4 * total_position_loss)
    #
    #     ankle_loss_v = ((state_desc['l_leg']['d_joint']['ankle'] - im_file['ankle_angle_l_speed'][t]) ** 2 +
    #                     (state_desc['r_leg']['d_joint']['ankle'] - im_file['ankle_angle_r_speed'][t]) ** 2)
    #     knee_loss_v = ((state_desc['l_leg']['d_joint']['knee'] - im_file['knee_angle_l_speed'][t]) ** 2 +
    #                    (state_desc['r_leg']['d_joint']['knee'] - im_file['knee_angle_r_speed'][t]) ** 2)
    #     hip_loss_v = ((state_desc['l_leg']['d_joint']['hip'] - im_file['hip_flexion_l_speed'][t]) ** 2 +
    #                   (state_desc['r_leg']['d_joint']['hip'] - im_file['hip_flexion_r_speed'][t]) ** 2 +
    #                   (state_desc['l_leg']['d_joint']['hip_abd'] - im_file['hip_adduction_l_speed'][t]) ** 2 +
    #                   (state_desc['r_leg']['d_joint']['hip_abd'] - im_file['hip_adduction_r_speed'][t]) ** 2)
    #
    #     total_velocity_loss = ankle_loss_v + knee_loss_v + hip_loss_v
    #     vel_reward = np.exp(-0.1 * total_velocity_loss)
    #
    #     im_rew = 0.9 * pos_reward + 0.1 * vel_reward
    #
    #     # print(f'im_rew: {im_rew},\t goal_rew: {goal_rew}')
    #     return 0.9 * im_rew + 0.1 * goal_rew, 10 - penalty, False

    def get_reward_1(self, state_desc):
        #state_desc = self.get_state_dict(self.state)
        # pelvis_reward = (self.prev_observation_dic["pelvis"]["body_pos"][0] - state_desc["pelvis"]["body_pos"][0])*2
        # if self.t>10:
        #     velocity_penalty = -(np.abs(state_desc['pelvis']["body_vel"][0] - 1.75))*1 + np.random.normal(0,0.18,)  #std taken from imitation file
        # else:
        #     velocity_penalty = 0
        # pelvis_rot_penalty = -np.abs(state_desc['pelvis']['roll'])*2 #+ np.random.normal(-0.013,0.041)*2 #hip rotation to be zero
        # #ankle cross penalty
        # hip_adduction_penalty = -(np.abs(state_desc['r_leg']['joint']['hip_abd'] + state_desc['l_leg']['joint']['hip_abd']))*2 #+ 0.13*2
        #
        # reward = pelvis_reward + velocity_penalty + pelvis_rot_penalty + hip_adduction_penalty
        pelvis_reward = (state_desc["pelvis"]["body_pos"][0] - self.prev_observation_dic["pelvis"]["body_pos"][0])*10
        alive_reward = 0
        if state_desc['pelvis']['body_pos'][1] >= 0.6:
            alive_reward = 0.5
        reward = pelvis_reward + alive_reward
        return reward, 0, False


def reduce_obs(state, state_size):
    if state_size<len(state):
        return state[:26] + state[59:70] + state[82:]
    else:
        return state

from Plotting import DynamicPlotting
plotter = DynamicPlotting(None)


def evaluate_model(env, model, horizon, pi=None, path="", iteration = 0, state_size = 94, logger = None):
    test_env = copy.copy(env)
    test_model = copy.copy(model)
    ob_env = test_env.reset()
    ob_model = test_model.reset(ob_env)
    ob_env_dict = test_env.get_observation_dict()
    ob_model_dict = test_model.get_state_dict(ob_model)
    States_env_dict = [ob_env_dict for _ in range(horizon)]
    States_model_dict = [ob_model_dict for _ in range(horizon)]
    States_model_std1 = [ob_model_dict for _ in range(horizon)]
    States_model_std2 = [ob_model_dict for _ in range(horizon)]
    ob_env_temp = reduce_obs(ob_env, state_size)
    States_env = [ob_env_temp for _ in range(horizon)]
    States_model = [ob_model for _ in range(horizon)]
    rew_env = [0]*horizon
    rew_model = [0]*horizon
    dones_env = [0]*horizon
    dones_model = [0]*horizon
    for h in range(1,horizon):
        if not pi:
            ac_env = np.random.randint(2, size = 17)
            ac_model = ac_env
        else:
            ac_env, _ = pi.act(False,ob_env + [test_env.t]) #reduce_obs(ob_env, state_size))
            ac_model, _ = pi.act(False, ob_model + [test_model.t])
        ob_env, rew, true_rew, done_env = test_env.step(ac_env)
        States_env[h] = reduce_obs(ob_env, state_size)
        States_env_dict[h] = test_env.get_observation_dict()
        rew_env[h] = rew
        dones_env[h] = done_env

        ob_model, ob_model_std1, ob_model_std2, rew, true_rew, done_model = test_model.step_eval(ac_model)
        States_model[h] = ob_model
        States_model_dict[h] = test_model.get_state_dict(ob_model)
        States_model_std1[h] = test_model.get_state_dict(ob_model_std1)
        States_model_std2[h] = test_model.get_state_dict(ob_model_std2)
        rew_model[h] = rew
        dones_model[h] = done_model
        if done_model or done_env:
            ob_env = test_env.reset()
            ob_model = test_model.reset(ob_env)
    plotter.PCA_visual(States_env,States_model, header = "multistep evaluation", path = path, iteration= iteration, logger=logger)
    plotter.compare_observations_drop_approx(States_env_dict,States_model_dict, States_model_std1, States_model_std2, header = "Multi step evaluation", path=path, iteration=iteration,logger=logger) #passed as dictionaries of observations
    plotter.compare_rew(rew_env,rew_model,header="rewards",path=path, iteration=iteration, logger=logger)
    plotter.compare_done(dones_env, dones_model, header = "Termination", path=path, iteration=iteration, logger=logger)
    #diff = np.mean(np.abs(np.subtract(np.array(States_env),np.array(States_model))))
    #print(diff)
    #model_evaluate_step(env,model,horizon,pi)
    #state_evaluate_step(env, model, horizon, pi)

def state_evaluate_step(env, model, horizon, pi, path, iteration,state_size, logger):
    horizon = 100
    short_hor = 5
    test_env = copy.copy(env)
    test_model = copy.copy(model)
    ob_env = test_env.reset()
    ob_model = test_model.reset()
    ob_env_dict = test_env.get_observation_dict()
    ob_model_dict = test_model.get_state_dict(ob_model)
    States_env = [ob_env_dict for _ in range(horizon)]
    States_model = [ob_model_dict for _ in range(horizon)]
    rew_env = [0]*horizon
    rew_model = [0]*horizon
    dones_env = [0]*horizon
    dones_model = [0]*horizon
    t=0
    h = 0
    while h <= horizon-short_hor:
        for _ in range(short_hor):
            States_env[h] = test_env.get_observation_dict()
            States_model[h] = test_model.get_state_dict(ob_model)
            if not pi:
                ac_env = ac_model = np.random.randint(2, size = 17)
            else:
                ac_env, _ = pi.act(False,reduce_obs(ob_env, state_size))
                ac_model, _ = pi.act(False, ob_model)

            ob_env, rew, true_rew, done_env = test_env.step(ac_env)
            rew_env[h] = rew
            dones_env[h] = done_env

            ob_model, rew, true_rew, done_model = test_model.step(ac_model)
            rew_model[h] = rew
            dones_model[h] = done_model
            t += test_env.stepsize
            h += 1
            if done_env:
                t=0
                ob_env = test_env.reset()
                ob_model = test_model.reset()
        ob_model = test_model.reset(ob_env, t)
    plotter.compare_observations(States_env,States_model, header = "One step evaluation",path = path, called_from = 1, iteration= iteration, logger=logger) #passed as dictionaries of observations
    plotter.compare_rew(rew_env,rew_model,header="rewards_one_step", path = path, called_from = 1, iteration=iteration, logger=logger)
    plotter.compare_done(dones_env,dones_model,header="termination_one_step", path = path, called_from = 1, iteration=iteration, logger=logger)

def model_evaluate_step(env, model, horizon, state_size, pi):
    test_env = copy.copy(env)
    test_model = copy.copy(model)
    ob_env = test_env.reset()
    ob_model = ob_env.copy()
    States_env = [ob_env for _ in range(horizon)]
    States_model = [ob_model for _ in range(horizon)]
    t=0
    for h in range(1,horizon):
        ac_env, _ = pi.act(False,reduce_obs(ob_env, state_size))
        ob_env, rew, true_rew, done_env = test_env.step(ac_env)
        States_env[h] = ob_env
        #ac_model,_ = pi.act(False,ob_model)
        ob_model, rew, true_rew, done_model = test_model.step(ac_env)
        t += test_env.stepsize
        States_model[h] = ob_model
        if done_env:
            t=0
            ob_env = test_env.reset()
        test_model.reset(ob_env, t)
    plotter.PCA_visual(States_env,States_model, header = "One step evaluation")

