import numpy as np
from input_MLP import MultiModelCreate, MultiModelSeqCreate, doneModel
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from numpy import dstack
from input_BNN import create_probablistic_bnn_model

class Ensemble:

    def __init__(self, replay_buffer_Drand, replay_buffer_Drl, eval_buffer,  mb_rms, op_mb_rms,scaler,
                 input_dim, output_dim,
                 activation_d="lrelu", activation_op="linear", networkUnits=64, nmembers=3,
                 nlayers=5, mb_ensemble = False):
        self.input_dim = input_dim
        self.out_dim = output_dim
        self.nmembers = nmembers
        self.ensemble = mb_ensemble
        self.make_bnn = False
        if self.ensemble:
            self.members, self.meta_model = self.create_ensemble(self.input_dim,self.out_dim,activation_d, activation_op, networkUnits, nlayers=nlayers)
        else:
            self.members = self.create_ensemble(self.input_dim,self.out_dim,activation_d, activation_op, networkUnits, nlayers=nlayers)
        #self.done_model = self.create_doneModel(self.out_dim,1,activation_d,networkUnits_done, nlayers_done, load_model)
        self.done_model = self.create_doneModel(self.out_dim, 1, activation_d, 200, 4)
        self.lr = 0.01
        self.meta_lr = 0.01
        self.scale = scaler
        self.replay_buffer_Drand = replay_buffer_Drand
        self.replay_buffer_Drl = replay_buffer_Drl
        self.eval_buffer = eval_buffer
        self.mb_rms = mb_rms
        self.op_mb_rms = op_mb_rms
        self.log_counter = 0

    def create_doneModel(self,input_dim = 94,out_dim = 1, activation_d='lrelu', networkUnits=64, nlayers=2):
        # model = None
        model = doneModel(input_dim,out_dim,activation_d,networkUnits,nlayers,model_name="Done_Model")
        return model


    def create_ensemble(self,input_dim = 111,out_dim = 94, activation_d='lrelu', activation_op='lrelu', networkUnits=64, nlayers=5):
        members = []
        for i in range(self.nmembers):
            if not self.make_bnn:
                model = MultiModelSeqCreate(input_dim, out_dim, activation_d, activation_op, networkUnits,
                                     nlayers=nlayers, model_name=str(i))
            else:
                model = create_probablistic_bnn_model(input_dim, out_dim)
            members.append(model)
        head_input_dim = out_dim*self.nmembers
        head_out_dim = out_dim
        if self.ensemble:
            meta_model = MultiModelCreate(head_input_dim, head_out_dim, activation_d, activation_op, 200,
                                 nlayers=2, model_name="meta")
        if self.ensemble:
            return members,meta_model  #if ensemble
        return (members)


    def get_data(self,sample_size = -1,sample_ratio = 0, seed = 22, replace=True):
        sample_Drand = int(sample_size*(1-sample_ratio))
        sample_Drl = sample_size - sample_Drand
        _, _, _, Labels_rl, Inputs_rl, _, Done_rl = map(np.array, self.replay_buffer_Drl.sample(sample_size=sample_Drl,replace=replace))
        _,_,_,Labels_rand,Inputs_rand,_, Done_rand = map(np.array, self.replay_buffer_Drand.sample(sample_size=sample_Drand,replace = replace))

        if sample_Drand == 0:
            return [], [], [], np.array(Labels_rl), np.array(Inputs_rl), [], np.array(Done_rl)
        elif sample_Drl == 0:
            return [], [], [], np.array(Labels_rand), np.array(Inputs_rand), [], np.array(Done_rand)
        else:
            return [],[],[],np.concatenate((Labels_rand,Labels_rl)),np.concatenate((Inputs_rand,Inputs_rl)),[], \
                   np.concatenate((Done_rand,Done_rl))


    def add_noise(self,x,noiseToSignal = 0.001):
        mean_data = np.mean(x, axis=0)
        std_of_noise = mean_data * noiseToSignal
        for j in range(mean_data.shape[0]):
            if (std_of_noise[j] > 0):
                x[:, j] = x[:, j] + np.random.normal(0, np.absolute(std_of_noise[j]), (x.shape[0],))
        return x

    def scale_features(self,x,y):
        if self.scale == "RuningMean":
            x = (x - self.mb_rms.mean) / self.mb_rms.std
            y = (y - self.op_mb_rms.mean) / self.op_mb_rms.std     #to scale output
        elif self.scale == "RuningMinMax":
            x = tf.convert_to_tensor(self.mb_rms.process(x))
            y = tf.convert_to_tensor(self.op_mb_rms.process(y))
        elif self.scale == "StandardMinMax":
            x = tf.convert_to_tensor(self.mb_rms.process(x))
            y = tf.convert_to_tensor(self.op_mb_rms.process(y))
        return x,y


    def load_saved_model(self, load_iters_mb):
        for index,member in enumerate(self.members):
            member.load_weights('best_model_{}'.format(str(index)))
        self.log_counter += load_iters_mb
        self.lr = self.lr * np.power(0.99, load_iters_mb)
        return


    def train(self, epochs=100, batch_size=128, path="", prop_rl=0.5,sample_size = 10000, logger=None):
        # if load_iters_mb is not None:
        #     self.load_saved_model(load_iters_mb)
        #     self.log_counter += load_iters_mb
        #     self.lr = self.lr * np.power(0.99, load_iters_mb)
        #     return
        history_tracker = []
        '''prepare evaluation set'''
        _, _, _, Labels_eval, Inputs_eval, _,_ = map(np.array, self.eval_buffer.sample(
            sample_size=500, replace = False))  # diff_observation-->Labels, inputs_scaled-->Inputs
        Inputs_eval, Labels_eval = self.scale_features(Inputs_eval, Labels_eval)
        #_, _, _, Labels_all, Inputs_all, _ = self.get_data(sample_size=sample_size,sample_ratio=prop_rl, replace=False) #change for ensemble
        # Training each model of ensemble
        print("----------------------Training model---------------------")
        for index,member in enumerate(self.members):
            _, _, _, Labels, Inputs, _,_ = self.get_data(sample_size=sample_size, sample_ratio=prop_rl, replace=False)
            Inputs, Labels = self.scale_features(Inputs, Labels)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint('best_model_{}'.format(str(index)), monitor='val_loss', mode='min', verbose=1,
                            save_best_only=True,
                            save_weights_only=True)
            K.set_value(member.optimizer.learning_rate, self.lr)
            if self.log_counter ==0:
                history = member.fit(Inputs, Labels, epochs=epochs, batch_size=batch_size, steps_per_epoch=200,
                                         validation_data=(Inputs_eval, Labels_eval), validation_steps=2,
                                         callbacks=[es,mc])
            else:
                history = member.fit(Inputs, Labels, epochs=epochs, batch_size=batch_size, steps_per_epoch=200,
                                         validation_data=(Inputs_eval, Labels_eval), validation_steps=2,
                                         callbacks=[es,mc])
            #member.save_weights("best_model_{}".format(str(index)))
            member.load_weights('best_model_{}'.format(str(index)))
            history_tracker.append(history)
        fig, axes = plt.subplots(nrows=self.nmembers, ncols=1, figsize=(16, 8))
        for history, ax in zip(history_tracker, np.array([axes])):  #switch to np.array([axes]) for non ensemble, axes for ensemble
            ax.plot(history.history['loss'])
            ax.plot(history.history['val_loss'])
        if logger:
            logger["train/loss_{}".format(str(self.log_counter))].upload(fig)
        plt.close(fig)
        self.log_counter += 1
        self.lr = self.lr * 0.98
        if not self.ensemble:
            return
        _, _, _, Labels, Inputs, _ = self.get_data(sample_size=5000,
                                                   sample_ratio=prop_rl,replace = True)
        Inputs = self.add_noise(Inputs)
        Labels = self.add_noise(Labels)
        Inputs, Labels = self.scale_features(Inputs, Labels)
        stackX = None
        stackX_val = None
        for member in self.members:
            pred_labels = member.predict(Inputs, steps = 1)
            pred_labels_val = member.predict(Inputs_eval, steps = 1)
            if stackX is None:
                stackX = pred_labels
                stackX_val = pred_labels_val
            else:
                stackX = dstack((stackX, pred_labels))
                stackX_val = dstack((stackX_val, pred_labels_val))

        # flatten predictions to [rows, members x probabilities]
        stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
        stackX_val = stackX_val.reshape((stackX_val.shape[0], stackX_val.shape[1] * stackX_val.shape[2]))
        K.set_value(self.meta_model.optimizer.learning_rate, self.meta_lr)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc_meta = ModelCheckpoint('best_model_meta', monitor='val_loss', mode='min', verbose=1,
                        save_best_only=True,
                        save_weights_only=True)
        history = self.meta_model.fit(stackX,Labels, epochs = 200, batch_size = 512, steps_per_epoch = 2,validation_data=(stackX_val,Labels_eval),validation_steps=1,
                                         callbacks=[es, mc_meta])
        self.meta_model.load_weights('best_model_meta')
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        self.meta_lr = self.meta_lr

    def train_doneModel(self, sample_size, prop_rl, replace, epochs = 200, batch_size = 128, load_iters_mb = None, logger = None):
        if load_iters_mb is not None:
            self.done_model.load_weights("best_termination_model")
            return
        print("----------------------Training Termination model---------------------")
        _, _, _, Labels_eval, Inputs_eval, _,Dones_eval = map(np.array, self.eval_buffer.sample(
            sample_size=700, replace = False))
        Inputs_eval, Labels_eval = self.scale_features(Inputs_eval, Labels_eval)
        _, _, _, Labels, Inputs, _, Dones = self.get_data(sample_size=sample_size, sample_ratio=prop_rl, replace=False)
        Inputs, Labels = self.scale_features(Inputs, Labels)
        Label_predict = self.members[0].predict(Inputs, steps=1)
        Label_predict_eval = self.members[0].predict(Inputs_eval, steps=1)
        State_predict = Label_predict + Inputs[:,:-17]
        State_predict_eval = Label_predict_eval + Inputs_eval[:,:-17]
        K.set_value(self.done_model.optimizer.learning_rate, 0.001)
        es = EarlyStopping(monitor='val_binary_accuracy', mode='max', verbose=1, patience=10)
        mc = ModelCheckpoint('best_termination_model', monitor='val_binary_accuracy', mode='max', verbose=1,
                        save_best_only=True,
                        save_weights_only=True)

        history_done = self.done_model.fit(State_predict, Dones, epochs=epochs, batch_size=batch_size, steps_per_epoch=50,
                                  validation_data=(State_predict_eval, Dones_eval), validation_steps=2,
                                  callbacks=[es, mc])
        self.done_model.load_weights("best_termination_model")
        #Calculate f1 score
        val_prec = np.array(history_done.history['val_precision'])
        val_rec = np.array(history_done.history['val_recall'])
        f1 = 2*val_prec*val_rec/(val_rec+val_prec)
        fig = plt.figure(figsize=(7, 9))
        plt.plot(history_done.history['loss'], label = "training loss")
        plt.plot(history_done.history['val_loss'], label = "test loss")
        plt.plot(f1, label = "f1 score")
        plt.xlabel("iterations")
        plt.ylabel("performance metric")
        if logger:
            logger["train_done/loss_{}".format(str(self.log_counter))].upload(fig)
        plt.close(fig)

    def predict_done(self, state):
        done = int(self.done_model.predict(state))
        return done

    def predict(self, x, calc_mean = False, for_eval = False):
        if self.ensemble:
            stackX = None
            for member in self.members:
                x_pred = member.predict(x, steps = 1)
                if stackX is None:
                    stackX = x_pred
                else:
                    stackX = dstack((stackX, x_pred))
            stackX = stackX.reshape((stackX.shape[0], stackX.shape[1] * stackX.shape[2]))
            final_pred = self.meta_model.predict(stackX, steps = 1)
            return final_pred
        else:
            if calc_mean or for_eval:
                x_pred,x_pred_std = self.prediction_mean(x, for_eval=for_eval, calc_mean = calc_mean)
                return x_pred, x_pred_std
            for member in self.members:
                x_pred = member.predict(x, steps=1)
            return x_pred,[]

    def prediction_mean (self,x, number_of_iterations = 50, for_eval = False, calc_mean = False) :
        '''Evaluates model performance on the test data using dropout during testing.
        Args :
        model (tensorflow/keras model): model test_data
        (numpy array): data we can call model.predict on,
        expected shape (#ofUnits, length, #features) test_data_target (numpy_array): target for the test_data,
        expected shape (#ofUnits, 1) number_of_iterations (int) : how many iterations of model.predict
        it is not the same each time because of
        using dropout during inference Returns:
        predictions (numpy array) : expected shape (#ofUnits, 1) RMSE (float)
        • RMSE of predicted RUL VS target RUL IF return prediction_list == True: predictions_list (list of numpy arrays)'''
        predictions_list = []
        for i in range(number_of_iterations):
            predictions_list.append(self.members[0].predict(x, steps = 1))

        predictions_mean = np.mean(predictions_list, axis=0).reshape(-1)
        predictions_std = np.std(predictions_list, axis=0).reshape(-1)

        predictions_mean = predictions_mean.reshape(1, predictions_mean.shape[0])
        predictions_std = predictions_std.reshape(1, predictions_std.shape[0])

        if for_eval:
            return predictions_mean, predictions_std  #return mean and std of prediction
        return predictions_mean, [] #return the mean prediction


    def predict_bnn(self):
        predicted = []
        iterations = 10
        for _ in range(iterations):
            predicted.append(self.members[0](x).numpy())
        predicted = np.concatenate(predicted, axis=1)

        prediction_mean = np.mean(predicted, axis=1).tolist()
        prediction_min = np.min(predicted, axis=1).tolist()
        prediction_max = np.max(predicted, axis=1).tolist()
        prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

        for idx in range(sample):
            print(
                f"Predictions mean: {round(prediction_mean[idx], 2)}, "
                f"min: {round(prediction_min[idx], 2)}, "
                f"max: {round(prediction_max[idx], 2)}, "
                f"range: {round(prediction_range[idx], 2)} - "
                f"Actual: {targets[idx]}"
            )

