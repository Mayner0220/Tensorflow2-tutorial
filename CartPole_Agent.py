"""
Cartpole Agent with Tensorflow 2

Reference: https://github.com/awjuliani/DeepRL-Agents/blob/master/Vanilla-Policy.ipynb
"""

import tensorflow as tf
import numpy as np
import gym

# Load cartpole environment
env = gym.make("CartPole-v0")

GAMMA = 0.99
learning_rate = 0.01
state_size = 4
num_actions = 2
hidden_size = 8
T_episodes = 5000
max_ep = 999
update_frequency = 5
is_visualize = False

def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0

    for i in reversed(range(0, r.size)):
        running_add = running_add * GAMMA + r[i]
        discounted_r[i] = running_add

    return discounted_r

class PolicyNetworks(tf.keras.Model):
    def __init__(self):
        super(PolicyNetworks, self).__init__()
        self.hidden_layer_1 = tf.keras.layers.Dense(hidden_size, activation='relu')
        self.output_layer = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, x):
        H1_output = self.hidden_layer_1(x)
        outputs = self.output_layer(H1_output)

        return outputs

def pg_loss(outputs, actions, rewards):
    indexes = tf.range(0, tf.shape(outputs)[0]) * tf.shape(outputs)[1] + actions
    responsible_outputs = tf.gather(tf.reshape(outputs, [-1]), indexes)

    loss = -tf.reduce_mean(tf.math.log(responsible_outputs) * rewards)

    return loss

optimizer = tf.optimizers.Adam(learning_rate)

def train_step(model, states, actions, rewards):
    with tf.GradientTape() as tape:
        outputs = model(states)
        loss = pg_loss(outputs, actions, rewards)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

PG_model = PolicyNetworks()

i = 0
total_reward = []
total_length = []

while i < T_episodes:
    s = env.reset()
    running_reward = 0
    ep_history = []

    for j in range(max_ep):
        if is_visualize == True:
            env.render()

        s = np.expand_dims(s, 0)
        a_dist = PG_model(s).numpy()
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)

        s1, r, d, _ = env.step(a)  # Get reward and next state
        ep_history.append([s, a, r, s1])
        s = s1
        running_reward += r

        if d == True:
            ep_history = np.array(ep_history)
            ep_history[:, 2] = discount_rewards(ep_history[:, 2])

            np_states = np.array(ep_history[0, 0])

            for idx in range(1, ep_history[:, 0].size):
                np_states = np.append(np_states, ep_history[idx, 0], axis=0)

            if i % update_frequency == 0 and i != 0:
                train_step(PG_model, np_states, ep_history[:, 1], ep_history[:, 2])

            total_reward.append(running_reward)
            total_length.append(j)
            break

    if i % 100 == 0:
        print(np.mean(total_reward[-100:]))
    i += 1