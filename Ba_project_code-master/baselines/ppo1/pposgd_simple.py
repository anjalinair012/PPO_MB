from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np, pandas as pd
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
import opensim
import os
from numpy import genfromtxt
import random
import MPC
from random import randint




buffer_use = "Drand"
noiseToSignal = 0.001
add_noise = False

def test_action(t):
    data = genfromtxt('action.txt')
    print("Dataset!!")
    print(t)
    return data[t]

def traj_top_generator(pi, learned_model, mpc):
    # ac = env.action_space_gym.sample()  # not used, just so we have the datatype
    # done = False  # marks if we're on first timestep of an episode
    #
    # ob = env.reset()
    #
    # limit = collection_rollouts * horizon
    # # Initialize history arrays
    # obs = np.array([ob for _ in range(limit)])
    # next_obs = np.array([ob for _ in range(limit)])
    # rews = np.zeros(limit, 'float32')
    # news = np.zeros(limit, 'int32')
    # acs = np.array([ac for _ in range(limit)])
    # timesteps = np.zeros(limit, 'float32')
    # end_states = []
    # counter = 0
    # while counter< limit:
    #     # Change this to call a function if we want to test a certain behavior
    #     ac,_ = pi.act(stochastic,ob)
    #     obs[counter] = ob
    #     acs[counter] = ac
    #     timesteps[counter] = env.t
    #     ob, rew, _, done = env.step(ac)
    #     next_obs[counter] = ob
    #     if done:
    #         end_states.append(counter)
    #         ob = env.reset()
    #     counter += 1
        # also try with same index as before
    best_trans  = mpc.MPC_run(pi,learned_model, True)
    return best_trans


def traj_segment_generator_eval(pi, env, horizon, stochastic, discount,state_size):
    def reduce_obs(observation,state_size):
        if state_size<len(observation):
            return observation[:26] + observation[59:70] + observation[82:]
        else:
            return observation
    t = 0

    new = True  # marks if we're on first timestep of an episode

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    # cur_ep_true_ret = 0
    ep_rets = np.zeros(horizon, 'float32')  # returns of completed episodes in this segment
    ep_lens = np.zeros(horizon, 'float32')  # lengths of ...
    # ep_true_rets = []

    ac = env.action_space.sample()  # not used, just so we have the datatype
    ob = env.reset()
    # Initialize history arrays
    # obs = np.array([ob for _ in range(horizon)])
    # rews = np.zeros(horizon, 'float32')
    # true_rews = np.zeros(horizon, 'float32')
    # vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    i = 0
    while True:
        prevac = ac
        # Change this to call a function if we want to test a certain behavior
        ac, vpred = pi.act(stochastic, reduce_obs(ob, state_size))
        ob, rew, true_rew, done = env.step(ac)
        cur_ep_ret += discount*rew
        cur_ep_len+=1
        if done:
            ep_rets[i] = cur_ep_ret
            ep_lens[i] = cur_ep_len
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        if i>=horizon:
            return np.mean(ep_rets),np.mean(ep_lens)
        i += 1


        # # Slight weirdness here because we need value function at time T
        # # before returning segment [0, T-1] so we get the correct
        # # terminal value
        # if t > 0 and t % horizon == 0:
        #     return (ep_rets, ep_lens)
        #     # Be careful!!! if you change the downstream algorithm to aggregate
        #     # several of these batches, then be sure to do a deepcopy
        #     ep_rets = []
        #     ep_lens = []
        #     ep_true_rets = []
        #
        # i = t % horizon
        # obs[i] = ob
        # vpreds[i] = vpred
        # news[i] = new
        # acs[i] = ac
        # prevacs[i] = prevac
        #
        # # print(t , '    ' , ac)
        # # print(ac)
        # # z = open("training_rewards.txt","a+")
        # # z.write("Episode %d    " % ac)
        # # np.savetxt('array.csv', [ac], delimiter=',', fmt='%d')
        #
        # # print(t)
        # ob, rew, true_rew, new = env.step(ac)
        #
        # rews[i] = rew
        #
        # cur_ep_ret += rew
        # cur_ep_true_ret += true_rew
        #
        # cur_ep_len += 1
        # if new:
        #     ep_rets.append(cur_ep_ret)
        #     ep_lens.append(cur_ep_len)
        #     ep_true_rets.append(cur_ep_true_ret)
        #     cur_ep_ret = 0
        #     cur_ep_len = 0
        #     cur_ep_true_ret = 0
        #
        #     ob = env.reset()
        # # print(t)
        # t += 1


def traj_segment_generator_VE(pi, env, mbAgent, learned_model, horizon, stochastic, timesteps_on_mb, add_noise = False):
    t = 0
    #ac = env.action_space_gym.sample() # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    # ob = env.reset()
    '''running on modelbased instead of real env'''
    env_used = learned_model
    noiseToSignal = 0.001

    '''starting transition taken from buffer to ensure the model is trained for it'''
    if buffer_use == "Drand":
        index = random.randint(1, mbAgent.replay_buffer_Drand.size - 1)
        start_ob = mbAgent.replay_buffer_Drand.experience[index][0]
        start_ac = np.array(mbAgent.replay_buffer_Drand.experience[index][1])  # not used
        start_t = mbAgent.replay_buffer_Drand.experience[index][5]
    elif buffer_use == "Drl":
        index = random.randint(1, mbAgent.replay_buffer_Drl.size - 1)
        start_ob = mbAgent.replay_buffer_Drl.experience[index][0]
        start_ac = np.array(mbAgent.replay_buffer_Drl.experience[index][1])
        start_prev_ac = np.array(mbAgent.replay_buffer_Drl.experience[index - 1][1])
        start_t = mbAgent.replay_buffer_Drl.experience[index][5]
    if add_noise:
        mean_data = np.mean(start_ob, axis=0)
        std_of_noise = mean_data * noiseToSignal
        start_ob = start_ob + np.random.normal(0, np.absolute(std_of_noise), (start_ob.shape[0],))

    env_used.reset(start_ob, start_t)
    #start_ob = env_used.reset()

    ob = start_ob.copy()
    ac = start_ac.copy()

    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    ep_true_rets = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    next_obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    true_rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    short_horizon_t = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": np.nan_to_num(rews), "vpred": np.nan_to_num(vpreds), "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": np.nan_to_num(vpred) * (1 - new),
                   "ep_rets": np.nan_to_num(ep_rets), "ep_lens": ep_lens, "ep_true_rets": np.nan_to_num(ep_true_rets),
                   "next_ob": next_obs}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_true_rets = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # print(t , '    ' , ac)
        # print(ac)
        # z = open("training_rewards.txt","a+")
        # z.write("Episode %d    " % ac)
        # np.savetxt('array.csv', [ac], delimiter=',', fmt='%d')

        # print(t)

        ob, rew, true_rew, new = env_used.step(ac)

        # ob, reward, new = env_used.step(ac)
        if short_horizon_t % timesteps_on_mb == 0 and t > 0 :  #Model based value expansion
            _,rew = pi.act(stochastic, ob)
            new = True
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew

        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_true_rets.append(cur_ep_true_ret)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_true_ret = 0

            if randint(0, 10)%300 == 0:
                ob = env_used.reset()
                start_t = 0
            # ob = env_used.reset(start_ob, start_t)
            elif buffer_use == "Drand":
                index = random.randint(1, mbAgent.replay_buffer_Drand.size - 1)
                start_ob = mbAgent.replay_buffer_Drand.experience[index][0]
                start_ac = np.array(mbAgent.replay_buffer_Drand.experience[index][1])  # not used
                start_t = mbAgent.replay_buffer_Drand.experience[index][5]
            elif buffer_use == "Drl":
                index = random.randint(1, mbAgent.replay_buffer_Drl.size - 1)
                start_ob = mbAgent.replay_buffer_Drl.experience[index][0]
                start_ac = np.array(mbAgent.replay_buffer_Drl.experience[index][1])
                start_prev_ac = np.array(mbAgent.replay_buffer_Drl.experience[index - 1][1])
                start_t = mbAgent.replay_buffer_Drl.experience[index][5]
            if add_noise:
                mean_data = np.mean(start_ob, axis=0)
                std_of_noise = mean_data * noiseToSignal
                start_ob = start_ob + np.random.normal(0, np.absolute(std_of_noise), (start_ob.shape[0],))
            ob = env_used.reset(start_ob, start_t)
            short_horizon_t = 0

        # print(t)
        t += 1
        short_horizon_t += 1

def traj_segment_generator(pi, env, mbAgent, learned_model, horizon, stochastic, timesteps_on_mb, add_noise = False):
    t = 0
    new = True  # marks if we're on first timestep of an episode
    '''running on modelbased instead of real env'''
    env = learned_model
    ob = env.reset()
    ac = np.array(mbAgent.replay_buffer_Drand.experience[0][1]) #ac = pi.act(stochastic, ob)  # not used, just so we have the datatype
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode
    cur_ep_true_ret = 0
    ep_rets = []  # returns of completed episodes in this segment
    ep_lens = []  # lengths of ...
    ep_true_rets = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    true_rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        # Change this to call a function if we want to test a certain behavior
        ac, vpred = pi.act(stochastic, ob)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob": obs, "rew": rews, "vpred": vpreds, "new": news,
                   "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets": ep_rets, "ep_lens": ep_lens, "ep_true_rets": ep_true_rets}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            ep_true_rets = []

        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # print("ac: ",ac)
        # print(pi.convert_act_to_category(ac))
        # print("----------------------\n\n")
        # print(used_ac)
        ob, rew, true_rew, new = env.step(ac)

        # print("\nStepped observation: \n^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n",ob)

        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_true_ret += true_rew

        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            ep_true_rets.append(cur_ep_true_ret)
            cur_ep_ret = 0
            cur_ep_len = 0
            cur_ep_true_ret = 0

            ob = env.reset()

        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, mbAgent, learned_model, seed, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        save_model_with_prefix, # Save the model
        save_prefix,
        restore_model_from_file,# Load the states/model from this file.
        load_after_iters,
        save_after,
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        num_parts = 0,  #number of particles for MPC
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        stochastic= True, load_model = False,
        aggregate_every_iter = 4, collection_rollouts = 10, collection_timesteps = 1000, # for collecting data for buffer
        mb_train_every_iter = 4, number_of_aggregation = 6, timesteps_on_mb = 200
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    counter_aggregation = 0
    # print("action space:")
    # print(ac_space)

    g=tf.get_default_graph()
    with g.as_default():
        tf.set_random_seed(seed)

    
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy


    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()
    '''Initialise MPC controller'''
    mpc = MPC.MPC_planner(horizon = collection_timesteps, T = 500, max_trajectories = collection_rollouts, real_env = env)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, mbAgent, learned_model, timesteps_per_actorbatch, stochastic=stochastic, timesteps_on_mb = timesteps_on_mb)
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    truerewbuffer = deque(maxlen=100)

    if restore_model_from_file == 1:
        saver=tf.train.Saver()
        basePath=os.path.dirname(os.path.abspath(__file__))
        modelF= basePath + '/' +"_afterIter_"+str(load_after_iters) + '.model'
        saver.restore(tf.get_default_session(), modelF)
        logger.log("Loaded model from {}".format(modelF))
        # Restore the variables from file
        data = genfromtxt('test_afterIter_' + str(load_after_iters) + '.csv', delimiter=',')

        for i in range(len(data)):
            data_vector = data[i]
            episodes_so_far = int(data_vector[0])
            timesteps_so_far = int(data_vector[1])
            iters_so_far = int(data_vector[2])
            time_elapsed = int(data_vector[3])
            lenbuffer.append(int(data_vector[4]))
            rewbuffer.append(int(data_vector[5]))
            truerewbuffer.append(int(data_vector[6]))     


    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"
    learned_model.update_model(mbAgent.dyn_model, mbAgent.mb_rms, mbAgent.op_mb_rms, env, pi, True,
                               iters_so_far)
    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        random.seed(22 * iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]


        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        '''In-loop MB training'''
        # if counter_aggregation < number_of_aggregation and iters_so_far % aggregate_every_iter == 0:
        #     top_traj = traj_top_generator(pi,learned_model, mpc)
        #     mbAgent.update_dataset(Observations=top_traj[0], Actions=top_traj[1], Next_Observations=top_traj[2],
        #                            Timesteps=top_traj[3], end_states=-1)
        #     counter_aggregation += 1
            #if iters_so_far % mb_train_every_iter == 0 and iters_so_far > 0:
                #mbAgent.train(sample_size = 10000)
                #learned_model.update_model(mbAgent.dyn_model, mbAgent.mb_rms, mbAgent.op_mb_rms, env, pi, True,iters_so_far)
        if iters_so_far%10 == 0 and iters_so_far>0:
            top_traj = traj_top_generator(pi, learned_model, mpc)
            mbAgent.update_dataset(Observations=top_traj[0], Actions=top_traj[1], Next_Observations=top_traj[2],
                                    Timesteps=top_traj[3], Dones= top_traj[4], end_states=-1)
            mbAgent.train(sample_size=10000)
            learned_model.update_model(mbAgent.dyn_model, mbAgent.mb_rms, mbAgent.op_mb_rms, env, pi, True,
                                       iters_so_far)
            ep_mean_rewards,ep_mean_length = traj_segment_generator_eval(pi=pi, env=env,horizon=500,stochastic=False, discount = gamma, state_size = mbAgent.state_size)
            o = open("mean_reward_real.txt", "a+")
            p = open("mean_length_real.txt", "a+")
            o.write("Reward %d\n    " % ep_mean_rewards)
            p.write("Length %d\n    " % ep_mean_length)
            o.close()
            p.close()
        if iters_so_far > 1:  # experiment with this
            buffer_use = "Drl"
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses= np.mean(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)


        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"], seg["ep_true_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews, truerews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        truerewbuffer.extend(truerews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpTrueRewMean", np.mean(truerewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        prev_episodes_so_far = episodes_so_far
        episodes_so_far +=len(lens)
        timesteps_so_far += sum(lens)

        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        f= open("training_rewards.txt","a+")
        g= open("training_episode_lengths.txt","a+")
        h= open("training_mean_rewards.txt", "a+")
        k= open("training_mean_lengths.txt", "a+")
        l= open("iterations.txt", "a+")
        m= open("timesteps.txt", "a+")
        n= open("training_mean_truerewards.txt", "a+")
        h.write("Episode %d    " % episodes_so_far)
        h.write("Reward  %d\r\n" % np.mean(rews))
        k.write("Episode %d    " % episodes_so_far)
        k.write("Length  %d\r\n" % np.mean(lens))
        n.write("Episode %d    " % episodes_so_far)
        n.write("Reward  %d\r\n" % np.mean(truerews))
        if(iters_so_far % 5 == 0):
           l.write("%d\r\n" % iters_so_far)
           m.write("%d\r\n" % timesteps_so_far)
        for i in range(episodes_so_far - prev_episodes_so_far):
            f.write("Episode %d    " % (prev_episodes_so_far + i))
            f.write("Reward  %d\r\n" % rews[i])
            g.write("Episode %d    " % (prev_episodes_so_far + i))
            g.write("Length  %d\r\n" % lens[i])
        f.close()
        g.close()
        k.close()
        h.close()
        l.close()
        m.close()
        n.close()
        #print("rews:\n")
        #print(rews)

        #g= open("training_episode_lengths.txt","a+")
        #for i in range(episodes_so_far - prev_episodes_so_far):
         #   g.write("Episode %d    " % (prev_episodes_so_far + i))
         #   g.write("Length  %d\r\n" % lens[i])
        #g.write("Episode %d    " % episodes_so_far)
        #g.write("Length  %d\r\n" % np.mean(lens))
        #g.close()


        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

        if iters_so_far % save_after ==0:
            if save_model_with_prefix:
                basePath=os.path.dirname(os.path.abspath(__file__))
                modelF= basePath + '/' +"_afterIter_"+str(iters_so_far)+".model"
                U.save_state(modelF)
                logger.log("Saved model to file :{}".format(modelF))
                if(episodes_so_far < 100):
                   size = episodes_so_far
                else:
                   size = 100
                asd = np.zeros((size, 7), dtype = np.int32)
                for i in range(size):
                    asd[i] = [episodes_so_far, timesteps_so_far, iters_so_far, time.time() - tstart, lenbuffer[i], rewbuffer[i], truerewbuffer[i]]
                    np.savetxt('test_afterIter_' + str(iters_so_far) + '.csv', asd, delimiter = ",")

                


    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
