"""This code was not gone through in the video, but essentially this code provides a useful example of how to 'Cook Chicken' and use a common image editing library to change colors."""
import multiprocessing
from multiprocessing import Process, Queue  # multiprocessing.Manager.Queue
import time
from osim.env import RUGTFPEnv

class Chef(Process):
    def __init__(self, queue, chicken, env):
        Process.__init__(self)
        self.queue = queue
        self.chicken = chicken
        self.env = env

    def cook_chicken(self, chicken, env):
        for i in range(len(chicken)):
            chicken[i] = 1
        return chicken

    def run(self):
        cooked_chicken = self.cook_chicken(self.chicken, self.env)
        print("Chef is finshed cooking")
        self.queue.put(cooked_chicken)


def start_run():
    ts = time.time()
    chefs = []
    queue = Queue()
    chickens_to_cook = 10
    env =  RUGTFPEnv(model_name="OS4_gait14dof15musc_2act_LTFP_VR_DynAct.osim",
        visualize=False, integrator_accuracy=1e-3, difficulty=1,
        seed=340, report=None, stepsize=0.01)
    for i in range(chickens_to_cook):
        raw_chicken = [0] * 10000000
        chefs.append(Chef(queue, raw_chicken,env))
    for chef in chefs:
        chef.start()

    while chickens_to_cook > 0:
        cooked_chicken = queue.get()
        print('cooked_chicken = ', cooked_chicken[59])
        chickens_to_cook -= 1
    print('All chefs have finished', round(time.time() - ts, 2), 'seconds')


start_run()