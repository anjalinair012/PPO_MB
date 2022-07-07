import time

from mpi4py import MPI
import osim
from osim.env import RUGTFPEnv
from baselines.common import set_global_seeds
from baselines import bench
import os.path as osp
from baselines import logger
#from baselines.common.cmd_util import atari_arg_parser
import opensim
import pandas as pd
import numpy as np
import gym
import tensorflow
from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype, CategoricalPdType, MultiCategoricalPdType
import random
import neptune.new as neptune

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name


    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)
        #setting for testing
        num_of_categories = 3
        num_hid_layers = 228
        self.pdtype = pdtype = MultiCategoricalPdType(low=np.zeros_like(ac_space.low, dtype=np.int32),
                                                          high=np.ones_like(ac_space.high, dtype=np.int32)*num_of_categories)
        gaussian_fixed_var= True
                        
        sequence_length = None
        self.ac_space_low = ac_space.low
        self.ac_space_high = ac_space.high
        self.num_of_categories = num_of_categories

        #ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list((95,)))
        with tf.variable_scope("obfilter"):
            #self.ob_rms = RunningMeanStd(shape=ob_space.shape)
            self.ob_rms = RunningMeanStd(shape=(95,))

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
             
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0))) #tanh
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0))) #tanh
            pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, np.expand_dims(ob,0))
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
    def convert_act_to_category(self,ac):
        # ac_space_high = np.ones_like(self.ac_space_high, dtype=np.float32)
        return  self.ac_space_low + (ac*(self.ac_space_high-self.ac_space_low))/self.num_of_categories


def train(num_timesteps, seed, save_model_with_prefix, restore_model_from_file, save_after, load_after_iters, load_model, viz=False, stochastic=True):

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=312, num_hid_layers=2)

    learned_model.update_model( mbAgent.dyn_model,mbAgent.mb_rms, mbAgent.op_mb_rms, env=env, evaluate = False, iteration_num = load_after_iters)  #update learned model
    pposgd_simple.learn(env, mbAgent, learned_model, workerseed, policy_fn,
                    timesteps_per_actorbatch=1526, clip_param=0.2,
                    entcoeff=0.01, optim_epochs=4, optim_stepsize=1e-3, optim_batchsize=512, gamma=0.999, lam=0.9,
                    save_model_with_prefix=save_model_with_prefix, save_prefix=env_string,
                    restore_model_from_file=restore_model_from_file, load_after_iters=load_after_iters,
                    save_after=save_after, max_timesteps=int(num_timesteps * 1.1), schedule='linear',
                    stochastic=stochastic, load_model=load_model, aggregate_every_iter = aggregate_every_iter,
                    collection_rollouts = mpc_max_rollot, collection_timesteps = mpc_collection_length,
                    mb_train_every_iter = mb_train_every_iter, number_of_aggregation = mb_num_aggregate,
                    timesteps_on_mb = timesteps_on_mb)
    restore_model_from_file = 1
    env.close()

# 5000000
import sys
import yaml
import ast

with open('Configs/config.yaml') as cf_file:
    config = yaml.safe_load(cf_file.read())
    load_model=ast.literal_eval(config.get("load_model"))
    mb_layers = int(config.get("mb_layers"))
    mb_members = int(config.get("mb_members"))
    mb_ensemble = ast.literal_eval(config.get("mb_ensemble"))
    mb_networkUnits = int(config.get("mb_networkUnits"))
    mb_batchSize = int(config.get("mb_batchSize"))
    mb_init_epochs = int(config.get("mb_init_epochs"))
    mb_scaler = str(config.get("mb_scaler"))
    mb_aggregate = ast.literal_eval(config.get("mb_aggregate"))
    #mb_aggregate_epochs = int(config.get("mb_aggregate_epochs"))
    mpc_max_rollot = int(config.get("mpc_max_rollot"))
    mpc_collection_length = int(config.get("mpc_collection_length"))
    mb_num_aggregate = int(config.get("mb_num_aggregate"))
    aggregate_every_iter = int(config.get("aggregate_every_iter"))
    fraction_use_new = float(config.get("fraction_use_new"))
    mb_train_every_iter = int(config.get("mb_train_every_iter"))
    timesteps_on_mb = int(config.get("timesteps_on_mb"))
    mb_epoch = int(config.get("mb_epoch"))
    title = str(config.get("title"))
    observation_size = int(config.get("observation_size"))
    difficulty = int(config.get("difficulty"))
    to_log = ast.literal_eval(config.get("to_log"))
    load_iters_mb = int(config.get("load_iters_mb"))
# 1 if restoring, 0 if not        print(state_desc.get('joint_pos').get('knee_l') , ',' , state_desc.get('joint_pos').get('knee_r'))
if load_iters_mb<0:
    load_iters_mb = None
if to_log:
    run = neptune.init(
    project="anjalinair012/Anjali-PPOMB",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI1ZWFjNmViZi00NmE5LTQzZjktYjcwNi03ZjU0MjBlY2M2NmEifQ==",
    )
    # add to neptune
    params = {"title": title,
              "load_model": load_model, "mb_layers": mb_layers,
              "mb_members": mb_members,
              "mb_ensemble": mb_ensemble,
              "mb_networkUnits": mb_networkUnits,
              "mb_batchSize": mb_batchSize,
              "mb_init_epochs": mb_init_epochs,
              "mb_scaler": mb_scaler,
              "mb_aggregate": mb_aggregate,
              "mpc_max_rollot": mpc_max_rollot,
              "mpc_collection_length": mpc_collection_length,
              "mb_num_aggregate": mb_num_aggregate,
              "aggregate_every_iter": aggregate_every_iter,
              "fraction_use_new": fraction_use_new,
              "mb_train_every_iter": mb_train_every_iter,
              "timesteps_on_mb": timesteps_on_mb,
              "mb_epoch": mb_epoch,
              "observation_size": observation_size,
              "difficulty": difficulty
              }
    run["parameters"] = params
else:
    run = None




restore = int(sys.argv[1])
# Run with mpirun -np 4 python main.py 450 to load after 450 iterations
# 1 is yes 0 is no
load_iters = int(sys.argv[2])
#train_iters = int(sys.argv[3])
if(load_iters == 1):
    with open('iterations.txt', 'r') as f:
        lines = f.read().splitlines()
        # Get the last line as the last stored iteration
        last_iter = int(lines[-1])
        load_iters = last_iter

seed = 999

'''moved from inside train()'''
from ModelBase import ModelBase,LearnedEnvironment
from baselines.ppo1 import pposgd_simple as pposgd_simple
import baselines.common.tf_util as U
rank = MPI.COMM_WORLD.Get_rank()  #rank of calling process
sess = U.single_threaded_session()  #single thread tf session returned
viz = False

sess.__enter__()
if rank == 0:
    logger.configure()
else:
    logger.configure(format_strs=[])
workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
set_global_seeds(workerseed)

g = tf.get_default_graph()
with g.as_default():
    tf.set_random_seed(workerseed)

env = RUGTFPEnv(model_name="OS4_gait14dof15musc_2act_LTFP_VR_DynAct.osim",
        visualize=viz, integrator_accuracy=1e-3, difficulty=difficulty,
        seed=random.randint(1,1000), report=None, stepsize=0.01)
env_string = str(env).split('<')[1]
with tf.variable_scope("ModelBase"):
    mbAgent = ModelBase(env=env, observation_size = observation_size, activation_d="lrelu", activation_op="linear", nlayers=mb_layers, epochs=mb_epoch,
                        mb_max_steps=10, capacity = 50000, batch_size = mb_batchSize, networkUnits = mb_networkUnits, sample_ratio = fraction_use_new, mb_ensemble = mb_ensemble, mb_scaler = mb_scaler, nmembers=mb_members,
                        logger = run)
    learned_model = LearnedEnvironment(mbAgent,difficulty)
U.initialize()
if load_model:
    print("----restore----")
    mbAgent.restore_state(load_iters_mb, aggregate_every_iter)
else:
    mbAgent.train(epochs = mb_init_epochs,minibatch_size = mb_batchSize, load_iters_mb = load_iters_mb)  #train environment model

train(5000000, 999, save_model_with_prefix=True, restore_model_from_file=restore, save_after=1, load_after_iters=load_iters, load_model=load_model, viz=viz, stochastic=True)
# train(50000, 999, save_model_with_prefix=True, restore_model_from_file=restore, save_after=5, load_after_iters=load_iters, viz=True, stochastic=True)

