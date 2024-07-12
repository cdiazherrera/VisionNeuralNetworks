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


"""JAX Convolutional model for image classification"""


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
    

    
