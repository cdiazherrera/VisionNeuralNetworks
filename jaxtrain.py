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
from JaxConvNet import CNN



"""JAX training functions and training loop"""


def compute_loss(state, params, batch, batch_stats, rng):
    inputs, targets = batch
    variables = {'params':params, 'batch_stats':batch_stats}
    outputs, updated_state = state.apply_fn(variables, inputs, rngs={'dropout':rng}, mutable=['batch_stats'])
    outputs=jnp.squeeze(outputs, axis=-1)
    loss = optax.sigmoid_binary_cross_entropy(outputs, targets).mean()
    return loss, updated_state['batch_stats']

def compute_accuracy(state, params, batch, batch_stats):
    inputs, targets = batch
    variables = {'params':params, 'batch_stats':batch_stats}
    outputs, _ = state.apply_fn(variables, inputs, mutable=True)
    predictions = jnp.squeeze(outputs, axis=-1)
    accuracy = (predictions.round() == targets).mean()
    return accuracy

@jax.jit

def train_step(state, batch, batch_stats, rng):
    (loss, new_batch_stats), grads = jax.value_and_grad(compute_loss, has_aux=True)(state.params, batch, batch_stats, rng)
    state = state.apply_gradients(grads=grads)
    return state, loss, new_batch_stats

#Training loop

num_epochs = 10

accuracy = []

loss = []


def train_loop(train_data, val_data, state, batch_stats, rng):
    for epoch in range(num_epochs):
        rng, input_rng = jax.random.split(rng)
        for batch in train_data:
            state, loss, batch_stats = train_step(state, batch, batch_stats, input_rng)
        val_accuracy = jnp.mean(jnp.array([compute_accuracy(state.params, batch, batch_stats) for batch in val_data]))
        accuracy.append(val_accuracy)
        loss.append(loss)
        print(f"Epoch {epoch + 1}, validation accuracy: {val_accuracy}, loss: {loss}")

    state_dict = {'params': state.params, 'batch_stats': batch_stats}

    state_bytes = flax.serialization.to_bytes(state_dict)
    with open('jax_conv_model.pkl', 'wb') as f:
        pickle.dump(state_bytes, f)
    print('Model saved!')