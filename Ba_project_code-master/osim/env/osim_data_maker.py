import math
import os
import random
from osim.utils.mygym import convert_to_gym
import gym
import numpy as np
import opensim
from opensim import (ActivationCoordinateActuator, CoordinateActuator,
                     ScalarActuator)
import pandas as pd

## OpenSim interface
# The main purpose of this class is to provide wrap all
# the necessery elements of OpenSim in one place
# The actual RL environment then only needs to:
# - open a model
# - actuate
# - integrate
# - read the high level description of the state
# The objective, stop condition, and other gym-related
# methods are enclosed in the OsimEnv class
class OsimModel(object):
    # Initialize simulation
    stepsize = 0.01

    model_path = None
    state = None
    state0 = None
    joints = []
    bodies = []
    brain = None
    verbose = False
    istep = 0
    start_point = 0
    timestep_limit = 300
    time_limit = 1e10

    state_desc_istep = None
    prev_state_desc = None
    state_desc = None
    integrator_accuracy = None

    visualize = False

    maxforces = []
    curforces = []

    def __init__(self, model_path, visualize, integrator_accuracy=5e-5, stepsize=0.01):
        self.integrator_accuracy = integrator_accuracy
        # self.load_model(model_path)
        self.model = opensim.Model(model_path)
        self.model_state = self.model.initSystem()
        self.brain = opensim.PrescribedController()

        # Enable the visualizer
        self.model.setUseVisualizer(visualize)

        self.muscleSet = self.model.getMuscles()
        self.forceSet = self.model.getForceSet()
        self.bodySet = self.model.getBodySet()
        self.jointSet = self.model.getJointSet()
        self.markerSet = self.model.getMarkerSet()
        self.contactGeometrySet = self.model.getContactGeometrySet()
        self.actuatorSet = self.model.getActuators()

        if self.verbose:
            self.list_elements()

        # Add actuators as constant functions. Then, during simulations
        # we will change levels of constants.
        # One actuartor per each muscle and actuator
        for j in range(self.actuatorSet.getSize()):
            func = opensim.Constant(1.0)
            self.brain.addActuator(self.actuatorSet.get(j))
            self.brain.prescribeControlForActuator(j, func)

        for j in range(self.muscleSet.getSize()):
            self.maxforces.append(self.muscleSet.get(j).getMaxIsometricForce())
            self.curforces.append(1.0)

        # Try to add constant functions for the motors as well

        self.noutput = self.actuatorSet.getSize()

        self.model.addController(self.brain)
        self.model_state = self.model.initSystem()

    def list_elements(self):
        print("JOINTS")
        for i in range(self.jointSet.getSize()):
            print(i, self.jointSet.get(i).getName())
        print("\nBODIES")
        for i in range(self.bodySet.getSize()):
            print(i, self.bodySet.get(i).getName())
        print("\nMUSCLES")
        for i in range(self.muscleSet.getSize()):
            print(i, self.muscleSet.get(i).getName())
        print("\nFORCES")
        for i in range(self.forceSet.getSize()):
            print(i, self.forceSet.get(i).getName())
        print("\nMARKERS")
        for i in range(self.markerSet.getSize()):
            print(i, self.markerSet.get(i).getName())
        print("\nACTUATORS")
        for i in range(self.actuatorSet.getSize()):
            print(i, self.actuatorSet.get(i).getName())

    def actuate(self, action):
        if np.any(np.isnan(action)):
            raise ValueError(
                "NaN passed in the activation vector. Values in interval [0,1] for muscles and [-1,1] for actuators are required.")

        # Clipping actions to ensure the values are within the range.
        action = np.array(action)
        action[0:15] = np.clip(action[0:15], 0.0, 1.0)
        action[15:17] = np.clip(action[15:17], -1.0, 1.0)
        self.last_action = action

        brain = opensim.PrescribedController.safeDownCast(self.model.getControllerSet().get(0))
        functionSet = brain.get_ControlFunctions()

        for j in range(functionSet.getSize()):
            func = opensim.Constant.safeDownCast(functionSet.get(j))
            func.setValue(float(action[j]))

    """
    Directly modifies activations in the current state.
    """

    def set_activations(self, activations):
        if np.any(np.isnan(activations)):
            raise ValueError("NaN passed in the activation vector. Values in [0,1] interval are required.")
        for j in range(self.muscleSet.getSize()):
            self.muscleSet.get(j).setActivation(self.state, activations[j])
        self.reset_manager()

    """
    Get activations in the given state.
    """

    def get_activations(self):
        return [self.muscleSet.get(j).getActivation(self.state) for j in range(self.muscleSet.getSize())]

    def compute_state_desc(self):
        self.model.realizeAcceleration(self.state)

        res = {}

        ## Joints
        res["joint_pos"] = {}
        res["joint_vel"] = {}
        res["joint_acc"] = {}
        for i in range(self.jointSet.getSize()):
            joint = self.jointSet.get(i)
            name = joint.getName()
            res["joint_pos"][name] = [joint.get_coordinates(i).getValue(self.state) for i in
                                      range(joint.numCoordinates())]
            res["joint_vel"][name] = [joint.get_coordinates(i).getSpeedValue(self.state) for i in
                                      range(joint.numCoordinates())]
            res["joint_acc"][name] = [joint.get_coordinates(i).getAccelerationValue(self.state) for i in
                                      range(joint.numCoordinates())]

        ## Bodies
        res["body_pos"] = {}
        res["body_vel"] = {}
        res["body_acc"] = {}
        res["body_pos_rot"] = {}
        res["body_vel_rot"] = {}
        res["body_acc_rot"] = {}
        for i in range(self.bodySet.getSize()):
            body = self.bodySet.get(i)
            name = body.getName()
            res["body_pos"][name] = [body.getTransformInGround(self.state).p()[i] for i in range(3)]
            res["body_vel"][name] = [body.getVelocityInGround(self.state).get(1).get(i) for i in range(3)]
            res["body_acc"][name] = [body.getAccelerationInGround(self.state).get(1).get(i) for i in range(3)]

            res["body_pos_rot"][name] = [
                body.getTransformInGround(self.state).R().convertRotationToBodyFixedXYZ().get(i) for i in range(3)]
            res["body_vel_rot"][name] = [body.getVelocityInGround(self.state).get(0).get(i) for i in range(3)]
            res["body_acc_rot"][name] = [body.getAccelerationInGround(self.state).get(0).get(i) for i in range(3)]

        ## Forces
        res["forces"] = {}
        for i in range(self.forceSet.getSize()):
            force = self.forceSet.get(i)
            name = force.getName()
            values = force.getRecordValues(self.state)
            res["forces"][name] = [values.get(i) for i in range(values.size())]

        ## Muscles
        res["muscles"] = {}
        for i in range(self.muscleSet.getSize()):
            muscle = self.muscleSet.get(i)
            name = muscle.getName()
            res["muscles"][name] = {}
            res["muscles"][name]["activation"] = muscle.getActivation(self.state)
            res["muscles"][name]["fiber_length"] = muscle.getFiberLength(self.state)
            res["muscles"][name]["fiber_velocity"] = muscle.getFiberVelocity(self.state)
            res["muscles"][name]["fiber_force"] = muscle.getFiberForce(self.state)
            # We can get more properties from here http://myosin.sourceforge.net/2125/classOpenSim_1_1Muscle.html

        ## Markers
        res["markers"] = {}
        for i in range(self.markerSet.getSize()):
            marker = self.markerSet.get(i)
            name = marker.getName()
            res["markers"][name] = {}
            res["markers"][name]["pos"] = [marker.getLocationInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["vel"] = [marker.getVelocityInGround(self.state)[i] for i in range(3)]
            res["markers"][name]["acc"] = [marker.getAccelerationInGround(self.state)[i] for i in range(3)]

        ## Other
        res["misc"] = {}
        res["misc"]["mass_center_pos"] = [self.model.calcMassCenterPosition(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_vel"] = [self.model.calcMassCenterVelocity(self.state)[i] for i in range(3)]
        res["misc"]["mass_center_acc"] = [self.model.calcMassCenterAcceleration(self.state)[i] for i in range(3)]

        return res

    def get_state_desc(self):
        if self.state_desc_istep != self.istep:
            self.prev_state_desc = self.state_desc
            self.state_desc = self.compute_state_desc()
            self.state_desc_istep = self.istep
        return self.state_desc

    def set_strength(self, strength):
        self.curforces = strength
        for i in range(len(self.curforces)):
            self.muscleSet.get(i).setMaxIsometricForce(self.curforces[i] * self.maxforces[i])

    def get_body(self, name):
        return self.bodySet.get(name)

    def get_joint(self, name):
        return self.jointSet.get(name)

    def get_muscle(self, name):
        return self.muscleSet.get(name)

    def get_marker(self, name):
        return self.markerSet.get(name)

    def get_contact_geometry(self, name):
        return self.contactGeometrySet.get(name)

    def get_force(self, name):
        return self.forceSet.get(name)

    def get_action_space_size(self):
        return self.noutput

    def set_integrator_accuracy(self, integrator_accuracy):
        self.integrator_accuracy = integrator_accuracy

    def reset_manager(self):
        self.manager = opensim.Manager(self.model)
        self.manager.setIntegratorAccuracy(self.integrator_accuracy)
        self.manager.initialize(self.state)

    def reset(self):
        self.state = self.initializeState()
        self.model.equilibrateMuscles(self.state)
        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

    def get_state(self):
        return opensim.State(self.state)

    def set_state(self, state):
        self.state = state
        self.istep = int(self.state.getTime() / self.stepsize)  # TODO: remove istep altogether
        self.reset_manager()

    def integrate(self):
        # Define the new endtime of the simulation
        self.istep = self.istep + 1

        # Integrate till the new endtime
        self.state = self.manager.integrate(self.stepsize * self.istep)

    def step(self, action):
        self.prev_state_desc = self.get_state_desc()
        self.actuate(action)
        self.integrate()

        obs = self.get_state_desc()

        return obs

    def initializeState(self):
        self.state = self.model.initializeState()

    def get_reward(self, t):
        return NotImplemented

    def is_done(self):
        return NotImplemented


## RUG TFP Model
# This environment provides basic interface to the transfemoral amputee model developed at RUG
class RUGTFPEnv(OsimModel):
    # to change later:
    # muscle v: normalize by max_contraction_velocity, 15 lopt / s
    # model = '3D'
    stepsize = 0.01
    model_path = None
    model = None

    timestep_limit = 300
    time_limit = 1e10
    # from gait14dof22musc_20170320.osim
    # 11.7769 + 9.30139 + 3.7075 + 0.1 + 1.25 + 0.21659 + 4.5 + 0.8199 + 0.77 + 0.710828809 + 34.2366 = 67.389708809
    MASS = 67.38971  # Add up all the body segement mass.
    G = 9.80665  # from gait1dof22muscle

    LENGTH0 = 1  # leg length

    footstep = {}
    footstep['n'] = 0
    footstep['new'] = False
    footstep['r_contact'] = 1
    footstep['l_contact'] = 1

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
    # muscle order in action
    # HAB_R, HAD_R, HAM_R, BFSH_R, GLU_R, HFL_R, RF_R, VAS_R, GAS_R, SOL_R, TA_R, HAB_L, HAD_L, GLU_L, HFL_L, KNE_ACT, ANK_ACT

    dict_actuator = {'knee_actuator': "KNE_ACT",
                     'ankle_actuator': "ANK_ACT"}

    # Initial pose should be an array of 17 values as shown below:
    # NOTE: The angles should be in radians
    # NOTE: The coordinates should be similar to the coordinates as seen in the OpenSim GUI Coordinates tab
    INIT_POSE = np.array([
        0.0,  # pelvis_tilt
        0.0,  # pelvis_list
        0.0,  # pelvis_rotation
        0.0,  # pelvis_tx
        0.94,  # pelvis_ty
        0.0,  # pelvis_tz
        0.0,  # hip_flexion_r
        0.0,  # hip_adduction_r
        0.0,  # hip_rotation_r
        0.0,  # knee_angle_r
        0.0,  # ankle_angle_r
        0.0,  # hip_flexion_l
        0.0,  # hip_adduction_l
        0.0,  # hip_rotation_l
        0.0,  # knee_angle_l
        0.0,  # ankle_angle_l
        0.0])  # lumbar_extension

    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
        self.model = OsimModel(self.model_path, self.visualize, integrator_accuracy=5e-5, stepsize=self.stepsize)

    def get_model_key(self):
        return self.model

    # Conditions to check if the simulations is done
    def is_done(self):
        raise NotImplementedError

    # Difficulty could be used for experimenting with different reward functions.
    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        if difficulty == 0:
            self.time_limit = 1000
            print("Difficulty 0, Choosing reward type 0")
        if difficulty == 1:
            self.time_limit = 1000
            print("Difficulty 1, Choosing reward type 1")
        if difficulty == 2:
            self.time_limit = 1000
            print("Difficulty 2, Choosing reward type 2")
        if difficulty == 3:
            self.time_limit = 2500  # 25 sec
            print("Difficulty 3, Choosing reward type 3")

    def __init__(self, model_name="", visualize=True, integrator_accuracy=5e-5, difficulty=1, stepsize=0.01, seed=None,
                 report=None):
        if difficulty not in [0, 1, 2, 3]:
            raise ValueError("difficulty level should be in [0, 1, 2, 3].")

        self.model_path = os.path.join(os.path.dirname(__file__), f"../models/{model_name}")
        super().__init__(visualize=visualize, model_path=self.model_path, integrator_accuracy=integrator_accuracy,
                         stepsize=stepsize)
        # super().load_model(model_path=self.model_path)
        self.Fmax = {}
        self.lopt = {}
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            self.Fmax[leg] = {}
            self.lopt[leg] = {}
            for MUS, mus in zip(
                    ['HAB_R', 'HAD_R', 'HAM_R', 'BFSH_R', 'GLU_R', 'HFL_R', 'RF_R', 'VAS_R', 'GAS_R', 'SOL_R', 'TA_R',
                     'HAB_L', 'HAD_L', 'GLU_L', 'HFL_L'],
                    ['abd_r', 'add_r', 'hamstrings_r', 'bifemsh_r', 'glut_max_r', 'iliopsoas_r', 'rect_fem_r',
                     'vasti_r', 'gastroc_r', 'soleus_r', 'tib_ant_r', 'abd_l', 'add_l', 'glut_max_l', 'iliopsoas_l']):
                try:
                    muscle = self.muscleSet.get('{}'.format(mus))
                    Fmax = muscle.getMaxIsometricForce()
                    lopt = muscle.getOptimalFiberLength()

                    self.Fmax[leg][MUS] = muscle.getMaxIsometricForce()
                    self.lopt[leg][MUS] = muscle.getOptimalFiberLength()
                except Exception as e:
                    # print(e) # Harmless exception to catch the unused muscles
                    pass

        # Actuator Optimal Force
        # Manual way of getting the optimal force from the knee and ankle actuators
        actuator_names = ['knee_actuator', 'ankle_actuator']
        self.Fmax['l_leg']['KNE_ACT'] = CoordinateActuator.safeDownCast(
            self.actuatorSet.get('{}'.format(actuator_names[0]))).getOptimalForce()
        self.Fmax['l_leg']['ANK_ACT'] = CoordinateActuator.safeDownCast(
            self.actuatorSet.get('{}'.format(actuator_names[1]))).getOptimalForce()
        self.action_space = (
                ([0.0] * (self.get_action_space_size() - 2)) + ([-1.0] * 2), [1.0] * self.get_action_space_size())
        self.observation_space = ([0] * self.get_observation_space_size(), [0] * self.get_observation_space_size())
        self.action_space = convert_to_gym(self.action_space)
        self.observation_space = convert_to_gym(self.observation_space)
        self.set_difficulty(difficulty)

        if report:
            bufsize = 0
            self.observations_file = open('%s-obs.csv' % (report,), 'w', bufsize)
            self.actions_file = open('%s-act.csv' % (report,), 'w', bufsize)
            self.get_headers()

        self.im_file = pd.read_csv("../EDITED-MATERIAL/175-FIX.csv", index_col=False)#normal_walking_data_edit2#ccw_data_walk_edit         175-FIX

    def reset(self, project=True, seed=None, init_pose=None, obs_as_dict=False, t = 0):
        self.t = t
        self.init_reward()

        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1

        # initialize state
        self.state = self.model.initializeState()
        if init_pose is None:
            init_pose = self.INIT_POSE
        state = self.get_state()
        QQ = state.getQ()
        QQDot = state.getQDot()
        for i in range(17):
            QQDot[i] = 0

        """
               This is the Layout of the Q vector:
               00: pelvis tilt 
               01: pelvis list
               02: pelvis rotation
               03: pelvis x
               04: pelvis y
               05: pelvis z
               06: hip flexion right 
               07: hip abduction right
               08: (hip ... right)
               09: hip flexion left
               10: hip abduction left
               11: (hip ... left)
               12: (lumbar ext)
               13: knee right 
               14: ankle right
               15: knee left
               16: ankle left
        """
        # Initial pose for the model loaded on reset
        t = int(t*200)
        QQ[0] = self.im_file['pelvis_tilt'][t]  ##+ random.uniform(-im_file['pelvis_tilt'][t],im_file['pelvis_tilt'][t])
        QQ[1] = self.im_file['pelvis_list'][t]  # + random.uniform(-im_file['pelvis_list'][t],im_file['pelvis_list'][t])
        QQ[2] = self.im_file['pelvis_rotation'][
            t]  ## + random.uniform(-im_file['pelvis_rotation'][t],im_file['pelvis_rotation'][t])# yaw
        QQ[3] = self.im_file['pelvis_tx'][t]  # if continue_imitation else 0] # x: (+) forward #B
        QQ[4] = self.im_file['pelvis_ty'][t]  # if continue_imitation  else 0]
        QQ[5] = self.im_file['pelvis_tz'][t]  # if continue_imitation else 0]# z: (+) right
        QQ[6] = self.im_file['hip_flexion_r'][
            t]  # + random.unifor#m(-im_file['hip_flexion_r'][t],im_file['hip_flexion_r'][t])
        QQ[7] = self.im_file['hip_adduction_r'][
            t]  # + random.unifo#rm(-im_file['hip_adduction_r'][t],im_file['hip_adduction_r'][t])
        QQ[8] = float(self.im_file['hip_rotation_r'][
                          t])  # + rando#m.uniform(-im_file['hip_rotation_r'][t],im_file['hip_rotation_r'][t])
        QQ[9] = self.im_file['knee_angle_r'][
            t]  # + random.uniform(-im_file['knee_angle_r'][t],im_file['knee_angle_r'][t])
        QQ[10] = self.im_file['ankle_angle_r'][
            t]  # + random.uniform(-im_file['ankle_angle_r'][t],im_file['ankle_angle_r'][t]
        QQ[11] = self.im_file['hip_flexion_l'][
            t]  # + random.uniform#(-im_file['hip_flexion_l'][t],im_file['hip_flexion_l'][t])
        QQ[12] = self.im_file['hip_adduction_l'][
            t]  # + random.uniform(-im_file['hip_adduction_l'][t],im_file['hip_adduction_l'][t])
        QQ[13] = float(self.im_file['hip_rotation_l'][
                           t])  # + random.uniform(-im_file['hip_rotation_l'][t],im_file['hip_rotation_l'][t])
        QQ[14] = self.im_file['knee_angle_l'][
            t]  # + random.uniform(-im_file['knee_angle_l'][t],im_file['knee_angle_l'][t])
        QQ[15] = self.im_file['ankle_angle_l'][
            t]  # + random.uniform(-im_file['ankle_angle_l'][t],im_file['ankle_angle_l'][t])
        QQ[16] = float(self.im_file['lumbar_extension'][
                           t])  # + random.uniform(-im_file['lumbar_extension'][t],im_file['lumbar_extension'][t])

        state.setQ(QQ)
        state.setU(QQDot)
        self.set_state(state)
        self.model.equilibrateMuscles(self.state)

        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

        d = self.get_state_desc()
        pose = np.array([d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])

        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        return self.get_observation()

    def step(self, action, project=True, obs_as_dict=True):
        action_mapped = [action[i] for i in self.act2mus]

        obs = super().step(action)
        self.t += self.stepsize
        self.update_footstep()

        d = super().get_state_desc()
        self.pose = np.array(
            [d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])

        # if project:
        #     if obs_as_dict:
        #         obs = self.get_observation_dict()
        #     else:
        #         obs = self.get_observation()
        # else:
        #     obs = self.get_state_desc()
        obs = self.get_observation()
        done = self.is_done()
        reward = self.get_reward()
        if round(self.t*200) >= len(self.im_file['pelvis_tilt']) - 10:
            # print("DONE!!!!!")
            done = True
        return obs, reward[0], reward[1], done

    # Returns boolean value for the model state
    def is_done(self):
        state_desc = self.get_state_desc()
        # To check if the model has fallen down or not
        if state_desc['body_pos']['pelvis'][1] < 0.6:
            done = True  # the model has fallen
        else:
            done = False  # the model is standing tall
        return done

    def update_footstep(self):
        state_desc = self.get_state_desc()

        # update contact
        r_contact = True if state_desc['forces']['foot_r'][1] < -0.05 * (self.MASS * self.G) else False
        l_contact = True if state_desc['forces']['foot_l'][1] < -0.05 * (self.MASS * self.G) else False

        self.footstep['new'] = False
        if (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def get_observation_dict(self):
        state_desc = self.get_state_desc()

        obs_dict = {}

        # pelvis state
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['height'] = state_desc['body_pos']['pelvis'][1]
        obs_dict['pelvis']['pitch'] = state_desc['joint_pos']['ground_pelvis'][0]  # (+) pitching forward
        obs_dict['pelvis']['roll'] = state_desc['joint_pos']['ground_pelvis'][
            1]  # (+) rolling around the forward axis (to the right)

        obs_dict['pelvis']['body_pos'] = state_desc['body_pos']['pelvis'][:3]
        obs_dict['pelvis']['body_vel'] = state_desc['body_vel']['pelvis'][:3]  # pelvis body velocity: tx, ty, tz
        obs_dict['pelvis']['joint_pos'] = state_desc['joint_pos']['ground_pelvis'][
                                          :3]  # pelvis joint: list, rotatation, tilt
        obs_dict['pelvis']['joint_vel'] = state_desc['joint_vel']['ground_pelvis'][
                                          :3]  # Pelvis joint velocity: list, rotatation, tilt

        # leg state
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            obs_dict[leg] = {}
            grf = [f / (self.MASS * self.G) for f in
                   state_desc['forces']['foot_{}'.format(side)][0:3]]  # forces normalized by bodyweight
            grm = [m / (self.MASS * self.G) for m in
                   state_desc['forces']['foot_{}'.format(side)][3:6]]  # forces normalized by bodyweight

            obs_dict[leg]['ground_reaction_forces'] = grf

            # joint angles
            obs_dict[leg]['joint'] = {}
            obs_dict[leg]['joint']['hip_abd'] = state_desc['joint_pos']['hip_{}'.format(side)][1]  # (+) hip abduction
            obs_dict[leg]['joint']['hip'] = state_desc['joint_pos']['hip_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['joint']['knee'] = state_desc['joint_pos']['knee_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['joint']['ankle'] = state_desc['joint_pos']['ankle_{}'.format(side)][0]  # (+) extension
            # joint angular velocities
            obs_dict[leg]['d_joint'] = {}
            obs_dict[leg]['d_joint']['hip_abd'] = state_desc['joint_vel']['hip_{}'.format(side)][1]  # (+) hip abduction
            obs_dict[leg]['d_joint']['hip'] = state_desc['joint_vel']['hip_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['d_joint']['knee'] = state_desc['joint_vel']['knee_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['d_joint']['ankle'] = state_desc['joint_vel']['ankle_{}'.format(side)][0]  # (+) extension

            # muscles
            if leg == 'r_leg':
                MUS_list = self.right_leg_MUS
                mus_list = self.right_leg_mus
            else:
                MUS_list = self.left_leg_MUS
                mus_list = self.left_leg_mus
            for MUS, mus in zip(MUS_list,
                                mus_list):
                obs_dict[leg][MUS] = {}
                obs_dict[leg][MUS]['f'] = state_desc['muscles']['{}'.format(mus)]['fiber_force'] / \
                                          self.Fmax[leg][MUS]
                obs_dict[leg][MUS]['l'] = state_desc['muscles']['{}'.format(mus)]['fiber_length'] / \
                                          self.lopt[leg][MUS]
                obs_dict[leg][MUS]['v'] = state_desc['muscles']['{}'.format(mus)]['fiber_velocity'] / \
                                          self.lopt[leg][MUS]
        # actuators
        obs_dict['l_leg']['force'] = {}
        obs_dict['l_leg']['actuator'] = {}
        obs_dict['l_leg']['actuator']['knee'] = {}
        obs_dict['l_leg']['actuator']['ankle'] = {}

        obs_dict['l_leg']['force']['knee'] = state_desc["actuators"]["knee"]["control"] * self.Fmax['l_leg'][
            'KNE_ACT']  # get instantaneous force
        obs_dict['l_leg']['actuator']['knee']['speed'] = state_desc["actuators"]["knee"]["speed"]
        obs_dict['l_leg']['actuator']['knee']['control'] = state_desc['actuators']['knee']['control']
        obs_dict['l_leg']['actuator']['knee']['power'] = state_desc['actuators']['knee']['power']
        obs_dict['l_leg']['actuator']['knee']['activation'] = state_desc['actuators']['knee']['activation']
        obs_dict['l_leg']['actuator']['knee']['actuation'] = state_desc['actuators']['knee']['actuation']

        obs_dict['l_leg']['force']['ankle'] = state_desc["actuators"]["ankle"]["control"] * self.Fmax['l_leg'][
            'ANK_ACT']  # get instataneous force
        obs_dict['l_leg']['actuator']['ankle']['speed'] = state_desc['actuators']['ankle']['speed']
        obs_dict['l_leg']['actuator']['ankle']['control'] = state_desc['actuators']['ankle']['control']
        obs_dict['l_leg']['actuator']['ankle']['power'] = state_desc['actuators']['ankle']['power']
        obs_dict['l_leg']['actuator']['ankle']['activation'] = state_desc['actuators']['ankle']['activation']
        obs_dict['l_leg']['actuator']['ankle']['actuation'] = state_desc['actuators']['ankle']['actuation']

        return obs_dict

    ## Values in the observation vector
    # 'pelvis': height(1), pitch(1), roll(1),
    #           body_pos(3), body_vel(3),
    #           joint_pos(3), joint_vel(3) (total: 15 values)
    # for each 'r_leg' and 'l_leg' (*2)
    #   'ground_reaction_forces' (3 values)
    #   'joint' (4 values)
    #   'd_joint' (4 values)
    #   for each of the 15 muscles (*15)
    #       normalized 'f', 'l', 'v' (3 values)
    # actuators for knee and ankle (*2)
    #       force - instantaneous (1 value)
    #       speed (1 value)
    #       control (1 value)
    #       actuation (1 value)
    #       power (1 value)
    #       activation (1 value)
    # TOTAL = 94
    def get_observation(self):
        obs_dict = self.get_observation_dict()

        # Augmented environment from the L2R challenge
        res = []

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

    def get_observation_clipped(self):
        obs = self.get_observation()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def get_observation_space_size(self):
        return 94

    def get_state_desc(self):
        d = super().get_state_desc()
        self.model.realizeAcceleration(self.state)

        ## Actuators
        try:
            d["actuators"] = {}
            d["actuators"]["knee"] = {}
            knee_scalar = ScalarActuator.safeDownCast(self.actuatorSet.get('knee_actuator'))
            knee_dyn = ActivationCoordinateActuator.safeDownCast(self.actuatorSet.get('knee_actuator'))
            d["actuators"]["knee"]["speed"] = knee_scalar.getSpeed(self.state)
            d["actuators"]["knee"]["control"] = knee_scalar.getControl(self.state)
            d["actuators"]["knee"]["actuation"] = knee_scalar.getActuation(self.state)
            d["actuators"]["knee"]["power"] = knee_scalar.getPower(self.state)
            d["actuators"]["knee"]["activation"] = knee_dyn.getStateVariableValue(self.state,
                                                                                  '/forceset/knee_actuator/activation')

            d["actuators"]["ankle"] = {}
            ankle_scalar = ScalarActuator.safeDownCast(self.actuatorSet.get('ankle_actuator'))
            ankle_dyn = ActivationCoordinateActuator.safeDownCast(self.actuatorSet.get('ankle_actuator'))
            d["actuators"]["ankle"]["speed"] = ankle_scalar.getSpeed(self.state)
            d["actuators"]["ankle"]["control"] = ankle_scalar.getControl(self.state)
            d["actuators"]["ankle"]["actuation"] = ankle_scalar.getActuation(self.state)
            d["actuators"]["ankle"]["power"] = ankle_scalar.getPower(self.state)
            d["actuators"]["ankle"]["activation"] = ankle_dyn.getStateVariableValue(self.state,
                                                                                    '/forceset/ankle_actuator/activation')
        except Exception as e:
            print(e)
        # state_desc['joint_pos']
        # state_desc['joint_vel']
        # state_desc['joint_acc']
        # state_desc['body_pos']
        # state_desc['body_vel']
        # state_desc['body_acc']
        # state_desc['body_pos_rot']
        # state_desc['body_vel_rot']
        # state_desc['body_acc_rot']
        # state_desc['forces']
        # state_desc['muscles']
        # state_desc['markers']
        # state_desc['misc']

        return d

    def init_reward(self):
        return NotImplemented

    def init_reward_1(self):
        return NotImplemented

    def get_reward(self):
        if self.difficulty == 1:  # Could use difficulty for trying different modes of rewards
            return self.get_reward_1()
        return self.get_reward_2()

    def get_reward_1(self):  # For reward type 1  based on difficulty(1)
        im_file = self.im_file

        state_desc = self.get_observation_dict()
        t = round(self.t*200)
        #
        penalty = 0
        x_penalty = (state_desc["pelvis"]["body_pos"][0] - im_file['pelvis_tx'][t]) ** 2
        y_penalty = (state_desc["pelvis"]["body_pos"][1] - im_file['pelvis_ty'][t]) ** 2
        z_penalty = (state_desc["pelvis"]["body_pos"][2] - im_file['pelvis_tz'][t]) ** 2
        penalty += (x_penalty + y_penalty + z_penalty)
        penalty += np.sum(np.array(self.get_activations()) ** 2) * 0.001  # reduce penalty of energy used.

        goal_rew = np.exp(-8 * (x_penalty + y_penalty + z_penalty))

        ankle_loss = ((state_desc['l_leg']['joint']['ankle'] - im_file['ankle_angle_l'][t]) ** 2 +
                      (state_desc['r_leg']['joint']['ankle'] - im_file['ankle_angle_r'][t]) ** 2)
        knee_loss = ((state_desc['l_leg']['joint']['knee'] - im_file['knee_angle_l'][t]) ** 2 +
                     (state_desc['r_leg']['joint']['knee'] - im_file['knee_angle_r'][t]) ** 2)
        hip_loss = ((state_desc["l_leg"]['joint']['hip'] - im_file['hip_flexion_l'][t]) ** 2 +
                    (state_desc['r_leg']['joint']['hip'] - im_file['hip_flexion_r'][t]) ** 2 +
                    (state_desc["l_leg"]['joint']['hip_abd'] - im_file['hip_adduction_l'][t]) ** 2 +
                    (state_desc['r_leg']['joint']['hip_abd'] - im_file['hip_adduction_r'][t]) ** 2)
        pelvis_angle_loss = ((state_desc["pelvis"]["pitch"] - im_file['pelvis_tilt'][t]) ** 2 +
                             (state_desc["pelvis"]["roll"] - im_file['pelvis_list'][t]) ** 2 +
                             (state_desc["pelvis"]["joint_pos"][2] - im_file['pelvis_rotation'][t]) ** 2)
        pelvis_pos_loss = ((state_desc['pelvis']['body_pos'][0] - im_file['pelvis_tx'][t]) ** 2 +
                           (state_desc['pelvis']['body_pos'][1] - im_file['pelvis_ty'][t]) ** 2 +
                           (state_desc['pelvis']['body_pos'][2] - im_file['pelvis_tz'][t]) ** 2)

        total_position_loss = ankle_loss + knee_loss + hip_loss + pelvis_angle_loss
        pos_reward = np.exp(-4 * total_position_loss)

        ankle_loss_v = ((state_desc['l_leg']['d_joint']['ankle'] - im_file['ankle_angle_l_speed'][t]) ** 2 +
                        (state_desc['r_leg']['d_joint']['ankle'] - im_file['ankle_angle_r_speed'][t]) ** 2)
        knee_loss_v = ((state_desc['l_leg']['d_joint']['knee'] - im_file['knee_angle_l_speed'][t]) ** 2 +
                       (state_desc['r_leg']['d_joint']['knee'] - im_file['knee_angle_r_speed'][t]) ** 2)
        hip_loss_v = ((state_desc['l_leg']['d_joint']['hip'] - im_file['hip_flexion_l_speed'][t]) ** 2 +
                      (state_desc['r_leg']['d_joint']['hip'] - im_file['hip_flexion_r_speed'][t]) ** 2 +
                      (state_desc['l_leg']['d_joint']['hip_abd'] - im_file['hip_adduction_l_speed'][t]) ** 2 +
                      (state_desc['r_leg']['d_joint']['hip_abd'] - im_file['hip_adduction_r_speed'][t]) ** 2)

        total_velocity_loss = ankle_loss_v + knee_loss_v + hip_loss_v
        vel_reward = np.exp(-0.1 * total_velocity_loss)

        im_rew = 0.9 * pos_reward + 0.1 * vel_reward

        # print(f'im_rew: {im_rew},\t goal_rew: {goal_rew}')

        return 0.9 * im_rew + 0.1 * goal_rew, 10 - penalty, False

    def get_reward_2(self):  # For reward type 2  based on difficulty(2)
        state_desc = self.get_state_desc()
        return NotImplemented


# Environment to use Gait14dof22Musc Model
class Able1422Env(OsimModel):
    MASS = 75.16
    G = 9.80665

    LENGTH0 = 1  # leg length

    footstep = {}
    footstep['n'] = 0
    footstep['new'] = False
    footstep['r_contact'] = 1
    footstep['l_contact'] = 1

    dict_muscle = {'abd': 'HAB',
                   'add': 'HAD',
                   'hamstrings': 'HAM',
                   'bifemsh': 'BFSH',
                   'glut_max': 'GLU',
                   'iliopsoas': 'HFL',
                   'rect_fem': 'RF',
                   'vasti': 'VAS',
                   'gastroc': 'GAS',
                   'soleus': 'SOL',
                   'tib_ant': 'TA'}

    act2mus = [0, 1, 4, 7, 3, 2, 5, 6, 8, 9, 10, 11, 12, 15, 18, 14, 13, 16, 17, 19, 20, 21]

    # maps muscle order in action to muscle order in gait14dof22musc_20170320.osim
    # muscle order in gait14dof22musc_20170320.osim
    #    HAB, HAD, HAM, BFSH, GLU, HFL, RF, VAS, GAS, SOL, TA
    #    or abd, add, hamstrings, bifemsh, glut_max, iliopsoas, rect_fem, vasti, gastroc, soleus, tib_ant

    # Initial pose should be an array of 17 values as shown below:
    # NOTE: The angles should be in radians
    # NOTE: The coordinates should be similar to the coordinates as seen in the OpenSim GUI Coordinates tab
    INIT_POSE = np.array([
        0.0,  # pelvis_tilt
        0.0,  # pelvis_list
        0.0,  # pelvis_rotation
        0.0,  # pelvis_tx
        0.94,  # pelvis_ty
        0.0,  # pelvis_tz
        0.0,  # hip_flexion_r
        0.0,  # hip_adduction_r
        0.0,  # hip_rotation_r
        0.0,  # knee_angle_r
        0.0,  # ankle_angle_r
        0.0,  # hip_flexion_l
        0.0,  # hip_adduction_l
        0.0,  # hip_rotation_l
        0.0,  # knee_angle_l
        0.0,  # ankle_angle_l
        0.0])  # lumbar_extension

    def load_model(self, model_path=None):
        if model_path:
            self.model_path = model_path
        self.model = OsimModel(self.model_path, self.visualize, integrator_accuracy=5e-5, stepsize=self.stepsize)

    # Difficulty could be used for experimenting with different reward functions.
    def set_difficulty(self, difficulty):
        self.difficulty = difficulty
        if difficulty == 0:
            self.time_limit = 1000
            print("Difficulty 0, Choosing reward type 0")
        if difficulty == 1:
            self.time_limit = 1000
            print("Difficulty 1, Choosing reward type 1")
        if difficulty == 2:
            self.time_limit = 1000
            print("Difficulty 2, Choosing reward type 2")
        if difficulty == 3:
            self.time_limit = 2500  # 25 sec
            print("Difficulty 3, Choosing reward type 3")

    def __init__(self, model_name="", visualize=True, integrator_accuracy=5e-5, difficulty=1, stepsize=0.01, seed=None,
                 report=None):
        if difficulty not in [0, 1, 2, 3]:
            raise ValueError("difficulty level should be in [0, 1, 2, 3].")

        self.model_path = os.path.join(os.path.dirname(__file__), f"../models/{model_name}")
        super().__init__(visualize=visualize, model_path=self.model_path, integrator_accuracy=integrator_accuracy,
                         stepsize=stepsize)

        self.Fmax = {}
        self.lopt = {}
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            self.Fmax[leg] = {}
            self.lopt[leg] = {}
            for MUS, mus in zip(['HAB', 'HAD', 'HAM', 'BFSH', 'GLU', 'HFL', 'RF', 'VAS', 'GAS', 'SOL', 'TA'],
                                ['abd', 'add', 'hamstrings', 'bifemsh', 'glut_max', 'iliopsoas', 'rect_fem', 'vasti',
                                 'gastroc', 'soleus', 'tib_ant']):
                muscle = self.muscleSet.get('{}_{}'.format(mus, side))
                Fmax = muscle.getMaxIsometricForce()
                lopt = muscle.getOptimalFiberLength()
                self.Fmax[leg][MUS] = muscle.getMaxIsometricForce()
                self.lopt[leg][MUS] = muscle.getOptimalFiberLength()



        self.set_difficulty(difficulty)

    def reset(self, project=True, seed=None, init_pose=None, obs_as_dict=False):
        self.t = 0
        self.init_reward()

        self.footstep['n'] = 0
        self.footstep['new'] = False
        self.footstep['r_contact'] = 1
        self.footstep['l_contact'] = 1

        # initialize state
        self.state = self.model.initializeState()
        if init_pose is None:
            init_pose = self.INIT_POSE
        state = self.get_state()
        QQ = state.getQ()
        QQDot = state.getQDot()
        for i in range(17):
            QQDot[i] = 0


        state.setQ(QQ)
        state.setU(QQDot)
        self.set_state(state)
        self.model.equilibrateMuscles(self.state)

        self.state.setTime(0)
        self.istep = 0

        self.reset_manager()

        d = self.get_state_desc()
        pose = np.array([d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])

        if not project:
            return self.get_state_desc()
        if obs_as_dict:
            return self.get_observation_dict()
        return self.get_observation()

    def step(self, action, project=True, obs_as_dict=True):
        action_mapped = [action[i] for i in self.act2mus]

        obs = super().step(action)
        obs = self.get_observation_dict()
        self.t += self.stepsize
        self.update_footstep()

        d = super().get_state_desc()
        self.pose = np.array(
            [d['body_pos']['pelvis'][0], -d['body_pos']['pelvis'][2], d['joint_pos']['ground_pelvis'][2]])

        done = self.is_done()
        reward = self.get_reward()
        if round(self.t*200) >= len(self.im_file['pelvis_tilt']) - 10:
            # print("DONE!!!!!")
            done = True

        return obs, reward, done

    # Conditions to check if the simulation is done
    # Returns boolean value for the model state
    def is_done(self):
        state_desc = self.get_state_desc()
        # To check if the model has fallen down or not
        if state_desc['body_pos']['pelvis'][1] < 0.6:
            done = True  # the model has fallen
        else:
            done = False  # the model is standing tall
        return done

    def update_footstep(self):
        state_desc = self.get_state_desc()

        # update contact
        r_contact = True if state_desc['forces']['foot_r'][1] < -0.05 * (self.MASS * self.G) else False
        l_contact = True if state_desc['forces']['foot_l'][1] < -0.05 * (self.MASS * self.G) else False

        self.footstep['new'] = False
        if (not self.footstep['r_contact'] and r_contact) or (not self.footstep['l_contact'] and l_contact):
            self.footstep['new'] = True
            self.footstep['n'] += 1

        self.footstep['r_contact'] = r_contact
        self.footstep['l_contact'] = l_contact

    def get_observation_dict(self):
        state_desc = self.get_state_desc()

        obs_dict = {}

        # pelvis state (in local frame)
        obs_dict['pelvis'] = {}
        obs_dict['pelvis']['height'] = state_desc['body_pos']['pelvis'][1]
        obs_dict['pelvis']['pitch'] = state_desc['joint_pos']['ground_pelvis'][0]  # (+) pitching forward
        obs_dict['pelvis']['roll'] = state_desc['joint_pos']['ground_pelvis'][
            1]  # (+) rolling around the forward axis (to the right)
        # yaw = state_desc['joint_pos']['ground_pelvis'][2]
        # dx_local, dy_local = rotate_frame(  state_desc['body_vel']['pelvis'][0],
        #                                     state_desc['body_vel']['pelvis'][2],
        #                                     yaw)
        # dz_local = state_desc['body_vel']['pelvis'][1]
        obs_dict['pelvis']['body_pos'] = state_desc['body_pos']['pelvis'][:3]
        obs_dict['pelvis']['body_vel'] = state_desc['body_vel']['pelvis'][:3]  # pelvis body velocity: tx, ty, tz
        obs_dict['pelvis']['joint_pos'] = state_desc['joint_pos']['ground_pelvis'][
                                          :3]  # pelvis joint: list, rotatation, tilt
        obs_dict['pelvis']['joint_vel'] = state_desc['joint_vel']['ground_pelvis'][
                                          :3]  # Pelvis joint velocity: list, rotatation, tilt

        # leg state
        for leg, side in zip(['r_leg', 'l_leg'], ['r', 'l']):
            obs_dict[leg] = {}
            grf = [f / (self.MASS * self.G) for f in
                   state_desc['forces']['foot_{}'.format(side)][0:3]]  # forces normalized by bodyweight
            grm = [m / (self.MASS * self.G) for m in
                   state_desc['forces']['foot_{}'.format(side)][3:6]]  # forces normalized by bodyweight

            if leg == 'r_leg':
                obs_dict[leg]['ground_reaction_forces'] = grf
            if leg == 'l_leg':
                obs_dict[leg]['ground_reaction_forces'] = grf

            # joint angles
            obs_dict[leg]['joint'] = {}
            obs_dict[leg]['joint']['hip_abd'] = state_desc['joint_pos']['hip_{}'.format(side)][1]  # (+) hip abduction
            obs_dict[leg]['joint']['hip'] = state_desc['joint_pos']['hip_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['joint']['knee'] = state_desc['joint_pos']['knee_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['joint']['ankle'] = state_desc['joint_pos']['ankle_{}'.format(side)][0]  # (+) extension
            # joint angular velocities
            obs_dict[leg]['d_joint'] = {}
            obs_dict[leg]['d_joint']['hip_abd'] = state_desc['joint_vel']['hip_{}'.format(side)][1]  # (+) hip abduction
            obs_dict[leg]['d_joint']['hip'] = state_desc['joint_vel']['hip_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['d_joint']['knee'] = state_desc['joint_vel']['knee_{}'.format(side)][0]  # (+) extension
            obs_dict[leg]['d_joint']['ankle'] = state_desc['joint_vel']['ankle_{}'.format(side)][0]  # (+) extension

            # muscles
            for MUS, mus in zip(['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA'],
                                ['abd', 'add', 'iliopsoas', 'glut_max', 'hamstrings', 'rect_fem', 'vasti', 'bifemsh',
                                 'gastroc', 'soleus', 'tib_ant']):
                obs_dict[leg][MUS] = {}
                obs_dict[leg][MUS]['f'] = state_desc['muscles']['{}_{}'.format(mus, side)]['fiber_force'] / \
                                          self.Fmax[leg][MUS]
                obs_dict[leg][MUS]['l'] = state_desc['muscles']['{}_{}'.format(mus, side)]['fiber_length'] / \
                                          self.lopt[leg][MUS]
                obs_dict[leg][MUS]['v'] = state_desc['muscles']['{}_{}'.format(mus, side)]['fiber_velocity'] / \
                                          self.lopt[leg][MUS]

        return obs_dict

    ## Values in the observation vector
    # 'pelvis': height, pitch, roll, 3 body pos, 3 body vel, 3 joint pos, 3 joint vel (15 values)
    # for each 'r_leg' and 'l_leg' (*2)
    #   'ground_reaction_forces' (3 values)
    #   'joint' (4 values)
    #   'd_joint' (4 values)
    #   for each of the eleven muscles (*11)
    #       normalized 'f', 'l', 'v' (3 values)
    #  TOTAL = 103

    def get_observation(self):
        obs_dict = self.get_observation_dict()

        res = []
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
            for MUS in ['HAB', 'HAD', 'HFL', 'GLU', 'HAM', 'RF', 'VAS', 'BFSH', 'GAS', 'SOL', 'TA']:
                res.append(obs_dict[leg][MUS]['f'])
                res.append(obs_dict[leg][MUS]['l'])
                res.append(obs_dict[leg][MUS]['v'])
        return res

    def get_observation_clipped(self):
        obs = self.get_observation()
        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def get_observation_space_size(self):
        return 103

    def get_state_desc(self):
        d = super().get_state_desc()
        self.model.realizeAcceleration(self.state)

        return d

    def init_reward(self):
        return NotImplemented

    def init_reward_1(self):
        return NotImplemented

    def get_reward(self):
        if self.difficulty == 1:  # Could use difficulty for trying different modes of rewards
            return self.get_reward_2()
        return self.get_reward_1()

    def get_reward_1(self):  # For reward type 1  based on difficulty(1)
        # Do something with the state description
        state_desc = self.get_state_desc()
        dt = self.model.stepsize
        reward_1 = 0
        return NotImplemented

    def get_reward_2(self):  # For reward type 2  based on difficulty(2)
        state_desc = self.get_state_desc()
        return NotImplemented


# Extras

def rotate_frame(x, y, theta):
    x_rot = np.cos(theta) * x - np.sin(theta) * y
    y_rot = np.sin(theta) * x + np.cos(theta) * y
    return x_rot, y_rot


def rotate_frame_3D(x, y, z, axis, theta):
    if axis == 'x':
        coord_axis = opensim.CoordinateAxis(0)
    elif axis == 'y':
        coord_axis = opensim.CoordinateAxis(1)
    elif axis == 'z':
        coord_axis = opensim.CoordinateAxis(2)
    else:
        raise Exception("Coordinate axis should be either x,y or z")

    # Rotation matrix
    rot_matrix = opensim.Rotation(np.deg2rad(theta), coord_axis)
    v = opensim.Vec3(x, y, z)
    rotated_frame = rot_matrix.multiply(v)
    x_rot = rotated_frame[0]
    y_rot = rotated_frame[1]
    z_rot = rotated_frame[2]

    return x_rot, y_rot, z_rot
