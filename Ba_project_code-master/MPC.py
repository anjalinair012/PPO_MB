import copy
import numpy as np
import pdb
import multiprocessing
import threading



class MPC_planner:

    def __init__(self, horizon, T, max_trajectories, real_env, outer_counter = 5):
        self.horizon = horizon
        self.T = T
        self.max_trajectories = max_trajectories
        self.real_env = copy.copy(real_env)
        self.start_ac = np.zeros((self.max_trajectories,self.real_env.get_action_space_size()))
        self.horizon_rew = np.zeros(self.max_trajectories)
        self.start_state = 0
        self.stepsize = self.real_env.stepsize
        self.State = np.zeros((self.T*outer_counter,94))
        self.Action = np.zeros((self.T*outer_counter, self.real_env.get_action_space_size()))
        self.Next_State = np.zeros((self.T*outer_counter, 94))
        self.Timesteps = np.zeros(self.T*outer_counter)
        self.Dones = np.zeros(self.T*outer_counter)
        self.stop = 5.475



    '''Note for report: adding individual transitions to D_rl and not trajectories makes the data for model IID. That's better in terms of assumptions'''
    def MPC_run(self,pi, env, stochastic = True):  #keep horizon small, num of trajectories laaaaargeeee
        outer_counter = 0
        while outer_counter<5:
            self.real_env.reset()
            ob = env.reset()
            counter = 0
            t = 0
            while counter<self.T:
                self.start_state = ob
                for traj_num in range(self.max_trajectories):
                    ob = env.reset(self.start_state, t)
                    ac = np.random.randint(2, size=self.start_ac.shape[1])
                    self.start_ac[traj_num] = ac
                    cum_rew = 0
                    local_horizon = int(min(t+self.horizon, self.T))
                    for h in range(local_horizon):
                        ob, rew, _, done = env.step(ac)
                        cum_rew +=rew
                        ac, _ = pi.act(stochastic, ob)
                    self.horizon_rew[traj_num] = cum_rew
                ob = self.add_best(outer_counter*10 + counter,t) #return best next_state
                counter += 1
                t += self.stepsize
            outer_counter +=1
        return (self.State,self.Action,self.Next_State,self.Timesteps, self.Dones)
        # pool = multiprocessing.Pool()
        # Rewards_obj = [pool.apply_async(CEM_Process, args=(action, state, Model)) for action in Actions]
        # Rewards = [r.get() for r in Rewards_obj]

    def add_best(self, counter,timestep_state):
        try:
            elite_idx = self.horizon_rew.argsort()[0]
            elite_action = self.start_ac[elite_idx]
            next_state, _,_,done = self.real_env.step(elite_action)
            self.State[counter] = self.start_state
            self.Action[counter] = elite_action
            next_state_reduced = reduce_obs(next_state, len(self.start_state))
            self.Next_State[counter] = next_state_reduced
            self.Timesteps[counter] = timestep_state
            self.Dones[counter] = done
            return next_state_reduced
        except:
            pdb.set_trace()

def reduce_obs(state, state_size):
    if state_size<len(state):
        return state[:26] + state[59:70] + state[82:]
    else:
        return state


def send_res(pi, learned_model, env, return_list):
    horizon = 5
    T = 20
    max_trajectories = 2
    real_env = env
    mpc = MPC_planner(horizon, T, max_trajectories, real_env)
    mpc.MPC_run(pi, learned_model, True)
    return_list.append(mpc.State)
    return_list.append(mpc.Action)
    return_list.append(mpc.Next_State)
    return_list.append(mpc.Timesteps)
