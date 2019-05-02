import tensorflow as tf
import numpy as np
import sonnet as snt
import gym
import itertools


learning_rate = 0.01
gamma = 0.95
RENDER_TH = 300
CUT_LENGTH = 2000
render = False

def build_actor(inputs, action_size):
  h = snt.Linear(32)(inputs)
  h = tf.nn.relu(h)
  h = snt.Linear(32)(h)
  h = tf.nn.relu(h)
  logits = snt.Linear(action_size)(h)
  new_action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
  return new_action, logits   

def calculate_returns(returns):
  output = np.zeros_like(returns)
  acc = 0
  for t in reversed(range(len(returns))):
    acc = acc * gamma + returns[t]
    output[t] = acc
  #output = output - np.mean(output)
  #output = output / np.std(output)
  return output

class Storage(object):
  def __init__(self):
    self.observtions = []
    self.actions = []
    self.rewards = []

  def add(self, obs, act, rew):
    self.observtions.append(obs)
    self.actions.append(act)
    self.rewards.append(rew)

  def empty(self):
    self.observtions = []
    self.actions = []
    self.rewards = []

env = gym.make('LunarLander-v2')
storage = Storage()

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

states_ph = tf.placeholder(dtype=tf.float32, shape=[None, state_size])
returns_ph = tf.placeholder(dtype=tf.float32, shape=[None, ])

actions, logits = build_actor(states_ph, action_size)

log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=actions,
                logits=logits,
                name="log_policy")

loss = tf.reduce_mean(log_policy * returns_ph)
optimizer = tf.train.AdamOptimizer(learning_rate)
global_steps = tf.Variable(0, trainable=False)
train_op = optimizer.minimize(loss, global_step=global_steps)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
  sess.run(init_op)

  for i in itertools.count():
    next_state = env.reset()
    storage.empty()

    episode_length = 1
    while True:
      action_sample = sess.run(actions, feed_dict={states_ph: [next_state]})[0]
      obs, rew, done, info = env.step(action_sample)   
      if render:
        env.render()  
      storage.add(next_state, action_sample, rew)
      next_state = obs

      if done or episode_length > CUT_LENGTH:
        ep_reward = sum(storage.rewards)
        print("episode {}, reward = {}, length = {}".format(i, ep_reward, episode_length))
        returns = calculate_returns(storage.rewards)
        sess.run(train_op, 
                 feed_dict={states_ph: storage.observtions,
                            actions: storage.actions,
                            returns_ph: returns})
        if ep_reward > RENDER_TH: 
          render = True
        break
      episode_length += 1