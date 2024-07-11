import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import tensorflow as tf
import os
import optax
from flax.training import train_state
import flax.serialization
import pickle


#change once data is acquired
open_hand_path = ""
closed_hand_path = ""

#Consider how data will be processed once data is grabbed 
#
#
#
#
#
#
#

class CNN(nn.Module):
    @nn.compact
    def __call__(self, x, training: bool=True):

        x = nn.Conv(features=16, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        x = nn.Conv(features=32, kernel_size=(3,3))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        x = nn.Conv(features=64, kernel_size=(2,2))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        x = nn.Conv(features=128, kernel_size=(2,2))(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not training)(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(features=128)(x)
        x = nn.relu(x)

        x = nn.Dropout(0.3)(x)

        x = nn.Dense(features=1)(x)
        x = nn.sigmoid(x)

        return x
    
    #height, width, channels
model = CNN()
rng = jax.random.PRNGKey(0)

#Variables
params = None #dummy variable for now
#batch_stats?

initial_learning_rate = 0.0001
optimizer = optax.adam(learning_rate=initial_learning_rate)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

def compute_loss(params, batch, batch_stats, rng):
    inputs, targets = batch
    variables = {'params':params, 'batch_stats':batch_stats}
    outputs, updated_state = state.apply_fn(variables, inputs, rngs={'dropout':rng}, mutable=['batch_stats'])
    outputs=jnp.squeeze(outputs, axis=-1)
    loss = optax.sigmoid_binary_cross_entropy(outputs, targets).mean()
    return loss, updated_state['batch_stats']

def compute_accuracy(params, batch, batch_stats):
    inputs, targets = batch
    variables = {'params':params, 'batch_stats':batch_stats}
    outputs, _ = 

    
