import tensorflow as tf
import numpy as np
import sonnet as snt
import scipy.signal
import gym
import itertools
from collections import namedtuple
from logx import EpochLogger

learning_rate = 0.01
decay = 0.99
gae_decay = 0.99
num_epochs = 50
steps_per_epoch = 4000
clip_ratio=0.2

pi_lr=3e-4
vf_lr=1e-3

train_pi_iters=80
train_v_iters=80

RENDER_TH = 100
render = False

AgentOutput = namedtuple("AgentOutput", 
                         "actions, log_policy, baselines")

PlaceHolders = namedtuple("PlaceHolders", 
                         "states, returns, log_policy, advs")

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Storage(object):
  def __init__(self, size, action_size, state_size, 
               lam=0.99, gamma=0.99):
    self.obs_buf = np.zeros((size, state_size), dtype=np.float32)
    self.act_buf = np.zeros(size, dtype=np.float32)
    self.adv_buf = np.zeros(size, dtype=np.float32)
    self.rew_buf = np.zeros(size, dtype=np.float32)
    self.ret_buf = np.zeros(size, dtype=np.float32)
    self.val_buf = np.zeros(size, dtype=np.float32)
    self.logp_buf = np.zeros(size, dtype=np.float32)
    self.gamma, self.lam = gamma, lam
    self.ptr, self.path_start_idx, self.max_size = 0, 0, size

  def add(self, obs, act, rew, val, logp):
    """
    Append one timestep of agent-environment interaction to the buffer.
    """
    assert self.ptr < self.max_size  
    self.obs_buf[self.ptr] = obs
    self.act_buf[self.ptr] = act
    self.rew_buf[self.ptr] = rew
    self.val_buf[self.ptr] = val
    self.logp_buf[self.ptr] = logp
    self.ptr += 1

  def finish_path(self, last_val=0):
    path_slice = slice(self.path_start_idx, self.ptr)
    rews = np.append(self.rew_buf[path_slice], last_val)
    vals = np.append(self.val_buf[path_slice], last_val)
    
    # the next two lines implement GAE-Lambda advantage calculation
    deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
    self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
    
    # the next line computes rewards-to-go, to be targets for the value function
    self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
    
    self.path_start_idx = self.ptr

  def get(self):
    assert self.ptr == self.max_size    # buffer has to be full before you can get
    self.ptr, self.path_start_idx = 0, 0

    self.adv_buf = (self.adv_buf - np.mean(self.adv_buf)) / np.std(self.adv_buf)
    return [self.obs_buf, self.act_buf, self.adv_buf, 
            self.ret_buf, self.logp_buf]


def build_nets(inputs, action_size):
  with tf.variable_scope("actor"):
    h = tf.layers.dense(inputs, 64, activation='relu')
    h = tf.layers.dense(h, 64, activation='relu')
    logits = tf.layers.dense(h, action_size, activation=None)
    new_action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
    logp_all = tf.nn.log_softmax(logits)
    log_policy = tf.reduce_sum(tf.one_hot(new_action, depth=action_size) * logp_all, axis=1)

  with tf.variable_scope("critic"):
    h = tf.layers.dense(inputs, 64, activation='relu')
    h = tf.layers.dense(h, 64, activation='relu')
    baseline = tf.squeeze(tf.layers.dense(h, 1, activation=None), axis=-1)

  return AgentOutput(actions=new_action, 
                     log_policy=log_policy, 
                     baselines=baseline)

def create_placeholders(state_size):
  states  = tf.placeholder(dtype=tf.float32, 
                           shape=[None, state_size])
  returns = tf.placeholder(dtype=tf.float32, shape=[None, ])
  advs = tf.placeholder(dtype=tf.float32, shape=[None, ])
  log_policy = tf.placeholder(dtype=tf.float32, shape=[None, ])
  return PlaceHolders(states=states,
                      returns=returns,
                      log_policy=log_policy,
                      advs=advs)


logger = EpochLogger()

env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

storage = Storage(steps_per_epoch, action_size, state_size)
ph = create_placeholders(state_size)
actor = build_nets(ph.states, action_size)

ratio = tf.exp(actor.log_policy - ph.log_policy)
min_adv = tf.where(ph.advs>0, 
                   (1+clip_ratio)*ph.advs, 
                   (1-clip_ratio)*ph.advs)
pi_loss = -tf.reduce_mean(tf.minimum(ratio * ph.advs, min_adv))
v_loss = tf.reduce_mean((ph.returns - actor.baselines)**2)

# Optimizers
train_pi = tf.train.AdamOptimizer(learning_rate=pi_lr).minimize(pi_loss)
train_v = tf.train.AdamOptimizer(learning_rate=vf_lr).minimize(v_loss)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)

  curr_state = env.reset()
  ep_ret, ep_len = 0, 0
  for epoch in range(num_epochs):
    for iteration in range(steps_per_epoch):
      act, logp, val = sess.run([actor.actions, 
                            actor.log_policy, 
                            actor.baselines], 
                        feed_dict={ph.states: [curr_state]})
      next_state, rew, done, info = env.step(act[0])   
      storage.add(curr_state, act, rew, val, logp)
      curr_state = next_state
      ep_ret += rew
      ep_len += 1

      if done or (iteration == steps_per_epoch-1):
        # print("episode return {}".format(ep_ret) )
        logger.store(EpRet=ep_ret, EpLen=ep_len)
        storage.finish_path()
        curr_state = env.reset()
        ep_ret, ep_len = 0, 0


    obs, act, adv, ret, logp_old = storage.get()
    for _ in range(train_pi_iters):
      _, pi_l = sess.run([train_pi, pi_loss],
                         feed_dict={
                          ph.states: obs,
                          actor.actions: act,
                          ph.log_policy: logp_old,
                          ph.advs: adv
                         })
    for _ in range(train_v_iters):
      _, v_l = sess.run([train_v, v_loss],
                         feed_dict={
                          ph.states: obs,
                          ph.returns: ret
                         })

    logger.store(LossPi=pi_l, LossV=v_l)

    # Log info about epoch
    logger.log_tabular('Epoch', epoch)
    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
    logger.log_tabular('LossPi', average_only=True)
    logger.log_tabular('LossV', average_only=True)
    logger.dump_tabular()
