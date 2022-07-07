import concurrent.futures
import copy
import time
import threading
import numpy as np
import pdb
from multiprocessing import Process, Queue

State = []
Action = []
Next_State = []
Timesteps = []


class MPC_planner(Process):
    def __init__(self, T, policy, model, queue, start_state,t): #
        Process.__init__(self)
        self.queue = queue
        # self.horizon = horizon
        self.T = T
        # self.max_trajectories = max_trajectories
        # self.real_env = copy.copy(real_env)
        # self.start_ac = np.zeros((self.max_trajectories,self.real_env.get_action_space_size()))
        # self.horizon_rew = np.zeros(self.max_trajectories)
        self.start_state = start_state
        # self.stepsize = self.real_env.stepsize
        # self.State = np.zeros((self.T,self.real_env.get_observation_space_size()))
        # self.Action = np.zeros((self.T, self.real_env.get_action_space_size()))
        # self.Next_State = np.zeros((self.T, self.real_env.get_observation_space_size()))
        # self.Timesteps = np.zeros(self.T)
        # self.stop = 5.475
        self.pi = policy
        self.env = model
        self.t = t

    def MPC_run_func(self, pi, env,t, T, start_state): #
        print("hello there")
        ob = env.reset(start_state, t)
        ac = np.random.randint(2, size=17)
        if np.random.randint(100) % 2 == 0:
            ac = np.random.randint(2, size=17)
        else:
            ac, _ = pi.act(True, ob)
            print("after act")

        start_ac = ac
        cum_rew = 0
        # for h in range(10): #horizon of 10
        #     print("in horizon")
        #     if t + h >= T:
        #         break
        #     ob, rew, _, done = env.step(ac)
        #     if done:
        #         print("breaking")
        #         break
        #     cum_rew += rew
        #     ac, _ = pi.act(True, ob)
        return cum_rew
        #return 10
        #self.queue.put(cum_rew)

    def run(self):
        #global queue
        print("In run")
        res = self.MPC_run_func(self.pi, self.env, self.t, self.T, self.start_state)
        print("Process running")
        self.queue.put(res)

def add_best(rewards, start_acts, real_env, start_state,timestep_state):
    elite_idx = np.array(rewards).argsort()[0]
    elite_action = start_acts[elite_idx]
    next_state, _,_,_ = real_env.step(elite_action)
    State.append(start_state)
    Action.append(elite_action)
    Next_State.append(next_state)
    Timesteps.append(timestep_state)
    return next_state



def send_res(pi, learned_model, env):
    print("in send res")
    counter = 0
    t = 0
    stepsize = 0.01
    results = []
    traj_num = 4
    T = 500
    real_env = env
    real_env.reset()
    ob = learned_model.reset()
    start_time = time.time()
    queue = Queue()
    while counter < T:
        start_state = ob
        traj_num_temp = traj_num
        workers = [MPC_planner(T, pi, learned_model, queue, start_state,t) for i in range(traj_num)]
            # #(pi, learned_model, t, T, start_state)
            # for f in concurrent.futures.as_completed(workers):
            #     results.append(f.result())
            # #worker.mpc_run_sub(pi, learned_model, t, T, start_state, queue)
        for w in workers:
            w.start()
        horizon_rews =[]
        horizon_acs = []
        while traj_num_temp > 0:
            res = queue.get()
            horizon_rews.append(res)
            # horizon_acs.append(res[1])
            traj_num_temp -= 1
        for worker in workers:
            worker.join()
        ob = add_best(horizon_rews,horizon_acs, real_env,start_state, t)
        counter += 1
        t += stepsize
    print("Time take ", str(time.time() - start_time))
    # return np.array(State),np.array(Action),np.array(Next_State),np.array(Timesteps)

