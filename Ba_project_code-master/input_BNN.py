from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import tensorflow.compat.v1 as tf
# tf.enable_v2_behavior()
from utils import ManualScaler
import tensorflow_probability as tfp
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.WARN)

#sns.set_style('whitegrid')
#sns.set_context('talk')



tfd = tfp.distributions
learning_rate = 0.01

def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model

def random_gaussian_initializer(shape, dtype):
    n = int(shape / 2)
    loc_norm = tf.random_normal_initializer(mean=0., stddev=0.1)
    loc = tf.Variable(
        initial_value=loc_norm(shape=(n,), dtype=dtype)
    )
    scale_norm = tf.random_normal_initializer(mean=-3., stddev=0.1)
    scale = tf.Variable(
        initial_value=scale_norm(shape=(n,), dtype=dtype)
    )
    return tf.concat([loc, scale], 0)


hidden_units = [10,10]
def create_probablistic_bnn_model(input_dim , out_dim, train_size = 1000):
    inputs = tf.keras.Input(shape=(input_dim,))
    # features = keras.layers.concatenate(list(inputs.values()))
    features = tf.keras.layers.BatchNormalization()(inputs)
    #features = inputs

    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)

    # Create a probabilisticå output (Normal distribution), and use the `Dense` layer
    # to produce the parameters of the distribution.
    # We set units=2 to learn both the mean and the variance of the Normal distribution.
    #outputs = tf.keras.layers.Dense(units=49)(features)
    distribution_params = tf.keras.layers.Dense(units=tfp.layers.IndependentNormal.params_size(event_shape=out_dim))(features)
    outputs = tfp.layers.IndependentNormal(event_shape=out_dim)(distribution_params)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanSquaredError(name='test_loss')],
    )
    return model


def main():

    x = np.random.rand(1000,66)
    y = x[:,:49] + np.random.random([1000,49]) * np.random.random([1000,49])
    print("-----reading input-----")
    #scaler = ManualScaler(obs_dim=66)
    #x = scaler.process(x)

    indices = np.random.permutation(x.shape[0])
    training_idx, test_idx = indices[:80], indices[80:]
    xtrain, xtest = x[training_idx,:], x[test_idx,:]
    ytrain,ytest = y[training_idx], y[test_idx]
    #xtrain = tf.convert_to_tensor(xtrain)
    #ytrain = tf.convert_to_tensor(ytrain)

    # print(tf.shape(xtrain), tf.shape(xtest))
    # print(tf.shape(ytrain),tf.shape(ytest))
    EPOCHS = 20

    # Create an instance of the model
    model = create_probablistic_bnn_model(66, 49)
    #model.compile(optimizer=optimizer, loss = loss_object)
    history = model.fit(xtrain, ytrain, epochs = 10, steps_per_epoch = 1, batch_size = 64)
    yhat = model(xtest)

    plt.figure(figsize=[6, 1.5])  # inches
    plt.plot(xtrain[:,1], ytrain[:,1], 'b.', label='observed')

    yhats = [model(xtest) for _ in range(10)]
    avgm = np.zeros_like(xtest[..., 0])
    for i, yhat in enumerate(yhats):
      m = np.squeeze(yhat.mean())
      s = np.squeeze(yhat.stddev())
      if i < 15:
        plt.plot(xtest[:,1], m, 'r', label='ensemble means' if i == 0 else None, linewidth=1.)
        plt.plot(xtest[:,1], m + 2 * s, 'g', linewidth=0.5, label='ensemble means + 2 ensemble stdev' if i == 0 else None);
        plt.plot(xtest[:,1], m[1] - 2 * s[1], 'g', linewidth=0.5, label='ensemble means - 2 ensemble stdev' if i == 0 else None);
      avgm += m
    plt.plot(xtest[:,1], avgm[1]/len(yhats[:,1]), 'r', label='overall mean', linewidth=4)

    plt.ylim(-0.,17);
    plt.yticks(np.linspace(0, 1, 9));
    plt.xticks(np.linspace(0,1, num=9));

    ax=plt.gca();
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    #ax.spines['left'].set_smart_bounds(True)
    #ax.spines['bottom'].set_smart_bounds(True)
    plt.legend(loc='center left', fancybox=True, framealpha=0., bbox_to_anchor=(1.05, 0.5))
    plt.show()


if __name__ == "__main__":
        main()

