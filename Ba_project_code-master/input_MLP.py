import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.activations import tanh, sigmoid
import pdb
from dataReader import fileRead
import matplotlib.pyplot as plt
from osim.env import RUGTFPEnv
import random
#from Plotting import DynamicPlotting
import os
import joblib
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K

loss_object = keras.losses.MeanSquaredError()

#optimizer = tf.keras.optimizers.SGD(lr_schedule, clipnorm=1.0)

lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.01,decay_steps=100,decay_rate=0.96)
optimizer = keras.optimizers.Adam(0.01)
train_loss = keras.metrics.MeanSquaredError(name = "train_loss")
mae_tracker = keras.metrics.MeanAbsoluteError(name='mae')


test_loss = tf.keras.metrics.MeanSquaredError(name='test_loss')
# class Dropout(keras.layers.Dropout):
# 	'''https://indico.cern.ch/event/917074/contributions/3855332/attachments/2035180/3407260/MCDropoutMLCoffee8thMay.pdf'''
# 	def __init__(self, rate, training= None, noise_shape=None, seed=None, **kwargs):
# 		super(Dropout, self).__init__(rate, noise_shape=None, seed=None, **kwargs)
# 		self.training = training
#
# 	def __call__(self, inputs, training=None):
# 		if 0. < self.rate <1.0:
# 			noise_shape = self._get_noise_shape(inputs)
#
# 			def dropped_inputs():
# 				return K.dropout(inputs, self.rate, noise_shape, seed=self.seed)
#
# 			if not training:
# 				return K.in_train_phase(dropped_inputs, inputs, training=self.training)
#
# 			return K.in_train_phase(dropped_inputs, inputs, training = training)
# 		return inputs
class MonteCarloDropout(keras.layers.Dropout):
  def call(self, inputs):
    return super().call(inputs, training=True)


class MLP(keras.Model):

  def __init__(self, model_name = "model", monitor_param='val_loss',mode='auto',verbose=1,save_best_only=True,save_weights_only=True, **kwargs):
	  super().__init__(**kwargs)
	  self.mc = ModelCheckpoint('best_model_{}'.format(str(model_name)), monitor='val_loss', mode='min', verbose=1,
                         save_best_only=True,
                         save_weights_only=True)

  @tf.function
  def train_step(self, data):
    x, y = data
    with tf.GradientTape() as tape:
      # training=True is only needed if there are layers with different
      # behavior during training versus inference (e.g. Dropout).
      predictions = self(x)
      loss = self.loss(y, predictions)
    #compute gradients
    gradients = tape.gradient(loss, self.trainable_variables)
    #update weights
    self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
   
    self.compiled_metrics.update_state(y, predictions)
    #train_loss.update_state(loss)
    #mae_tracker.update_state(y,predictions)
    return {m.name: m.result() for m in self.metrics}

  @tf.function
  def test_step(self, data):
    x,y = data
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = self(x)
    loss = self.loss(y, predictions)
    self.compiled_metrics.update_state(y, predictions)
    return {m.name: m.result() for m in self.metrics}
    
  @property
  def metrics(self):
    # We list our `Metric` objects here so that `reset_states()` can be
    # called automatically at the start of each epoch
    # or at the start of `evaluate()`.
    # If you don't implement this property, you have to call
    # `reset_states()` yourself at the time of your choosing.
    return [train_loss, mae_tracker]

def MultiModelCreate(input_dim, out_dim, activation_d, activation_op, networkUnits, nlayers = 5, model_name = 0):
	initializer = tf.compat.v1.initializers.he_normal(seed=340)
	regl2 = tf.keras.regularizers.l2(0.001)

	inputState = keras.Input(shape=(input_dim,))
	#layers.Normalization(axis=-1, mean=None, variance=None)
	prev_lay = Dropout(0.2, training=True)(inputState)
	#prev_lay = inputState
	for layer_num in range(1,nlayers):
		#denseLayer = layers.Dense(networkUnits, kernel_regularizer = regl2, bias_regularizer = regl2, kernel_initializer=initializer)(prev_lay)
		denseLayer = layers.Dense(networkUnits, kernel_regularizer = regl2, kernel_initializer=initializer, bias_initializer=initializer)(prev_lay)
			#denseLayer = layers.LayerNormalization(axis=1)(denseLayer)
		if activation_d == "lrelu":
			denseLayer = LeakyReLU(alpha=0.01)(denseLayer)
		elif activation_d == 'tanh':
			denseLayer = tanh(denseLayer)
		elif activation_d == "sigmoid":
			denseLayer = sigmoid(denseLayer)
		elif activation_d == "relu":
			denseLayer = ReLU()(denseLayer)
		if layer_num %1 == 0:
			denseLayer = Dropout(0.2, training=True)(denseLayer)
		prev_lay = denseLayer
	outputs = layers.Dense(out_dim, activation = "linear",kernel_regularizer = regl2)(prev_lay)

	model = MLP(inputs=inputState, outputs=outputs,model_name=model_name)

	model.compile(optimizer=optimizer, loss = loss_object , metrics=["mae"])
	return model

def doneModel(input_dim, out_dim = 1, activation = "relu", networkUnits = 200, nlayers = 2, model_name = "doneModel"):
	model = keras.Sequential()
	model.add(layers.Input(shape=(input_dim,)))
	regl2 = tf.keras.regularizers.l2(0.001)
	initializer = tf.compat.v1.initializers.he_normal(seed=340)
	for layer_num in range(1, nlayers):
		model.add(layers.Dense(networkUnits, activation="relu", kernel_regularizer=regl2, kernel_initializer=initializer))
		if layer_num == 1:
			model.add(tf.keras.layers.BatchNormalization())
	model.add(layers.Dense(out_dim, activation='sigmoid'))
	model.compile(optimizer=optimizer, loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
	return model


def MultiModelSeqCreate(input_dim, out_dim, activation_d, activation_op, networkUnits, nlayers = 5, model_name = 0):
	model = keras.Sequential()
	regl2 = tf.keras.regularizers.l2(0.0)
	initializer = tf.compat.v1.initializers.he_normal(seed=340)
	model.add(layers.Input(shape=(input_dim,)))
	for layer_num in range(1, nlayers):
		model.add(MonteCarloDropout(0.2))
		model.add(layers.Dense(networkUnits, kernel_regularizer=regl2, kernel_initializer=initializer))
		model.add(LeakyReLU(alpha=0.05))
		if layer_num == 1:
			model.add(tf.keras.layers.BatchNormalization())
	model.add(layers.Dense(out_dim))
	model.add(LeakyReLU(alpha=0.05))
	model.compile(optimizer=optimizer, loss=loss_object, metrics=[test_loss,])
	return model
'''Module Testing Code'''

def rand_act(env, actionDim):
  action = np.random.rand(15)
  knee_act = random.uniform(-1.0, 1.0) # Negative values for: knee flexion and +ve for extension
  ankle_act = random.uniform(-1.0, 1.0) # Negative values for: ankle plantarflexion +ve for dorsiflexion
  act = np.append(action, [knee_act, ankle_act]) # 17 control signal values: 15 muscles + 2 prosthetic actuators
  return list(act)

                
                
def rand_experience(env, state, actionDim):
  act = rand_act(env,actionDim)
  observation, reward, done = env.step(act)
  return env, state, act,reward,observation,done


def main():
	try:
		path = "SavedModels/ModelOnData2"
		os.mkdir(path)
	except OSError as error:
		print(error)
	print("-----reading input-----")
	x, y = fileRead()
	scaler = MinMaxScaler()
	x[:,:94] = scaler.fit_transform(x[:,:94])
	pdb.set_trace()
	y = scaler.transform(y)
	print("read")

	indices = np.random.permutation(x.shape[0])
	training_idx, test_idx = indices[:80], indices[80:]
	xtrainS, xtestS = x[training_idx,:94], x[test_idx,:94]
	xtrainA, xtestA = x[training_idx,94:], x[test_idx,94:]
	ytrain,ytest = y[training_idx]-xtrainS, y[test_idx]-xtestS
	xtrainS = tf.convert_to_tensor(xtrainS) 
	xtrainA = tf.convert_to_tensor(xtrainA)
	ytrain = tf.convert_to_tensor(ytrain)

	print(tf.shape(xtrainS), tf.shape(xtestS))
	print(tf.shape(ytrain),tf.shape(ytest))
	EPOCHS = 20

	# Create an instance of the model
	model = MultiModelCreate(94, 17, 'lrelu', 'relu', 2, 64)
	#model.compile(optimizer=optimizer, loss = loss_object)
	history = model.fit([xtrainS, xtrainA], ytrain, epochs = 100, batch_size = 128, steps_per_epoch = 1, validation_data=([xtestS, xtestA], ytest), validation_steps = 1)

	modelpath = os.path.join(path, "modelWeights")
	model.save_weights(modelpath)
	scalerpath = os.path.join(path, "Scaler.gz")
	joblib.dump(scaler, scalerpath)

	plotpath = os.path.join(path, "Losses.png")
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.savefig(plotpath)
	plt.show(block = True)

	print("--------------Further testing----------------")
	env = RUGTFPEnv(model_name="OS4_gait14dof15musc_2act_LTFP_VR_DynAct.osim", visualize=False)
	restore_model_from_file = False
	stateDim=env.get_observation_space_size()
	actionDim=env.get_action_space_size()
	trajLength = 10
	num_traj = 5
	State = []
	Action = []
	label=[]
	for traj in range(num_traj):
		state = env.reset()
	for step in range(trajLength):
		env, state, action, reward, next_state, done = rand_experience(env, state, actionDim)
		State.append(state)
		Action.append(action)
		label.append(next_state)
		state = next_state
		if done:
			break
	label = scaler.transform(label)
	State = np.array(scaler.transform(State))
	Action = tf.convert_to_tensor(Action)
	StateT = tf.convert_to_tensor(State)
	label = tf.convert_to_tensor(label)
	label = label - StateT
	pdb.set_trace()
	predictions = model.predict_on_batch([StateT, Action])
	final_states = predictions + State
	final_states = scaler.inverse_transform(final_states)
	State = scaler.inverse_transform(State)
	ploting = DynamicPlotting()
	pdb.set_trace()
	ploting.PCA_visual(State, final_states,"")
	pdb.set_trace()
	ploting.plotDifferences(State, final_states, "")


if __name__ == "__main__":
        main()


