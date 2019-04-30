import tensorflow as tf
import sonnet as snt
import numpy as np
import gym
import py_process
import environments

import collections
import contextlib

nest = tf.contrib.framework.nest

flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS


flags.DEFINE_string('logdir', './tmp/agent', 'TensorFlow log directory.')
flags.DEFINE_enum('mode', 'train', ['train', 'test'], 'Training or test mode.')

flags.DEFINE_string('game', 'LunarLander-v2', 'game code')


# Flags used for distributed training.
flags.DEFINE_integer('task', -1, 'Task id. Use -1 for local training.')
flags.DEFINE_enum('job_name', 'learner', ['learner', 'actor'],
                  'Job name. Ignored when task is set to -1.')

# Training
flags.DEFINE_integer('total_environment_frames', int(1e9),
                     'Total environment frames to train for.')
flags.DEFINE_integer('unroll_length', 1000, 'Unroll length in agent steps.')
flags.DEFINE_integer('seed', 1, 'Random seed.')


# Loss settings.
flags.DEFINE_float('entropy_cost', 0.00025, 'Entropy cost/multiplier.')
flags.DEFINE_float('baseline_cost', .5, 'Baseline cost/multiplier.')
flags.DEFINE_float('discounting', .99, 'Discounting factor.')
flags.DEFINE_float('GAE_discounting', 0.99, 'GAE decay')
flags.DEFINE_float('PPO_clip_ratio', .2, 'PPO clipping ratio')


flags.DEFINE_enum('reward_clipping', 'abs_one', ['abs_one', 'soft_asymmetric'],
                  'Reward clipping.')
# Optimizer settings.
flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')
flags.DEFINE_float('decay', .99, 'RMSProp optimizer decay.')
flags.DEFINE_float('momentum', 0., 'RMSProp momentum.')
flags.DEFINE_float('epsilon', .1, 'RMSProp epsilon.')

# Testing
flags.DEFINE_integer('test_num_episodes', 1, 'Number of episodes per level.')

# Structure to be sent from actors to learner.
ActorOutput = collections.namedtuple(
    'ActorOutput', 'level_name env_outputs agent_outputs')
AgentOutput = collections.namedtuple('AgentOutput',
                                     'action logits baseline')


def is_single_machine():
  return FLAGS.task == -1

@contextlib.contextmanager
def pin_global_variables(device):
  """Pins global variables to the specified device."""
  def getter(getter, *args, **kwargs):
    var_collections = kwargs.get('collections', None)
    if var_collections is None:
      var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
      with tf.device(device):
        return getter(*args, **kwargs)
    else:
      return getter(*args, **kwargs)

  with tf.variable_scope('', custom_getter=getter) as vs:
    yield vs

class Agent(snt.AbstractModule):
  def __init__(self, action_size, name="agent"):
    super(Agent, self).__init__(name=name)
    self._action_size = action_size

  def _torso(self, input_):
    last_action, env_output = input_
    reward, _, _, frame = env_output
    frame = frame[0]

    with tf.variable_scope('mlp'):
      mlp_out = frame
      mlp_out = snt.Linear(32)(mlp_out)
      mlp_out = tf.nn.relu(mlp_out)
      mlp_out = snt.Linear(32)(mlp_out)
      mlp_out = tf.nn.relu(mlp_out)

    mlp_out = snt.BatchFlatten()(mlp_out)
    return mlp_out
   
  def _head(self, _input):
    core_output, actions = _input
    logits = snt.Linear(self._action_size, name='logits')(
        core_output)
    action_sample = tf.random.categorical(logits, 1)
    action_sample = tf.squeeze(action_sample, 1, name='new_action')
    baseline = tf.squeeze(snt.Linear(1, name='baseline')(
        core_output), axis=-1)
    return AgentOutput(action_sample, 
                       logits, 
                       baseline)

  def _build(self, input_):
    action, env_output = input_
    actions, env_outputs = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                              (action, env_output))
    outputs = self.unroll(actions, env_outputs)
    return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs)

  @snt.reuse_variables
  def unroll(self, actions, env_outputs):
    _, _, done, _ = env_outputs
    shape = tf.shape(actions) # [T, B, d]
    torso_outputs = snt.BatchApply(self._torso)((actions, env_outputs))
    # Note, in this implementation we can't use CuDNN RNN to speed things up due
    # to the state reset. This can be XLA-compiled (LSTMBlockCell needs to be
    # changed to implement snt.LSTMCell).
    core_output_list = []
    for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
      # If the episode ended, the core state should be reset before the next.
      core_output_list.append(input_)
    return snt.BatchApply(self._head)((tf.stack(core_output_list), actions))


def create_environment(game_name, state_size):
  """Creates an environment wrapped in a `FlowEnvironment`."""
  config = {
    'observation_size': state_size
  }

  p = py_process.PyProcess(environments.PyProcessGym, game_name, config)
  return environments.FlowEnvironment(p.proxy)


def discount_returns(env_outputs):
  # Use last baseline value (from the value function) to bootstrap.
  rewards, infos, done, _ = nest.map_structure(
      lambda t: t[1:], env_outputs)

  discounts = tf.to_float(~done) * FLAGS.discounting
  sequences = (
    tf.reverse(discounts, axis=[0]),
    tf.reverse(rewards, axis=[0]),
  )
    # GAE
  def scanfunc(acc, sequence_item):
    discount_t, r_t = sequence_item
    return r_t + discount_t * acc
  initial_values = tf.zeros_like(rewards[-1])
  returns = tf.scan(
      fn=scanfunc,
      elems=sequences,
      initializer=initial_values,
      parallel_iterations=1,
      back_prop=False,
      name='scan')
  # Reverse the results back to original order.
  returns = tf.reverse(returns, [0], name='returns')
  return tf.stop_gradient(returns)


def compute_baseline_loss(advantages):
  # Loss for the baseline, summed over the time dimension.
  # Multiply by 0.5 to match the standard update rule:
  # d(loss) / d(baseline) = advantage
  return .5 * tf.reduce_sum(tf.square(advantages))

def compute_entropy_loss(logits):
  policy = tf.nn.softmax(logits)
  log_policy = tf.nn.log_softmax(logits)
  entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
  return -tf.reduce_sum(entropy_per_timestep)


def compute_policy_gradient_loss(logits, actions, advantages):
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=actions, logits=logits)
  policy_gradient_loss_per_timestep = cross_entropy * advantages
  return tf.reduce_sum(policy_gradient_loss_per_timestep)

def build_actor(agent, env, game_name, action_size):
  """Builds the actor loop."""
  # Initial values.
  initial_env_output, initial_env_state = env.initial()
  initial_action = tf.random.uniform([1], minval=0, maxval=1, dtype=tf.int32)
  dummy_agent_output = agent(
      (initial_action,
       nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output)))
  initial_agent_output = nest.map_structure(
      lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output)

  # All state that needs to persist across training iterations. This includes
  # the last environment output, agent state and last agent output. These
  # variables should never go on the parameter servers.
  def create_state(t):
    # Creates a unique variable scope to ensure the variable name is unique.
    with tf.variable_scope(None, default_name='state'):
      return tf.get_local_variable(t.op.name, initializer=t, use_resource=True)

  persistent_state = nest.map_structure(
      create_state, (initial_env_state, initial_env_output, 
                     initial_agent_output))

  def step(input_, unused_i):
    """Steps through the agent and the environment."""
    env_state, env_output, agent_output = input_

    # Run agent.
    action = agent_output[0]
    batched_env_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            env_output)
    agent_output = agent((action, batched_env_output))

    # Convert action index to the native action.
    raw_action = agent_output[0][0]
    env_output, env_state = env.step(raw_action, env_state)
    return env_state, env_output, agent_output

  # Run the unroll. `read_value()` is needed to make sure later usage will
  # return the first values and not a new snapshot of the variables.
  first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
  first_env_state, first_env_output, first_agent_output = first_values

  # Use scan to apply `step` multiple times, therefore unrolling the agent
  # and environment interaction for `FLAGS.unroll_length`. `tf.scan` forwards
  # the output of each call of `step` as input of the subsequent call of `step`.
  # The unroll sequence is initialized with the agent and environment states
  # and outputs as stored at the end of the previous unroll.
  # `output` stores lists of all states and outputs stacked along the entire
  # unroll. Note that the initial states and outputs (fed through `initializer`)
  # are not in `output` and will need to be added manually later.
  output = tf.scan(step, tf.range(FLAGS.unroll_length), first_values)
  _, env_outputs, agent_outputs = output

  # Update persistent state with the last output from the loop.
  assign_ops = nest.map_structure(lambda v, t: v.assign(t[-1]),
                                  persistent_state, output)

  # The control dependency ensures that the final agent and environment states
  # and outputs are stored in `persistent_state` (to initialize next unroll).
  with tf.control_dependencies(nest.flatten(assign_ops)):
    # Remove the batch dimension from the agent state/output.
    first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
    agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

    # Concatenate first output and the unroll along the time dimension.
    full_agent_outputs, full_env_outputs = nest.map_structure(
        lambda first, rest: tf.concat([[first], rest], 0),
        (first_agent_output, first_env_output), (agent_outputs, env_outputs))

    output = ActorOutput(
        level_name=game_name, 
        env_outputs=full_env_outputs, agent_outputs=full_agent_outputs)

    # No backpropagation should be done here.
    return nest.map_structure(tf.stop_gradient, output)

def build_learner(agent, env_outputs, agent_outputs):
  """Builds the learner loop.

  Args:
    agent: A snt.RNNCore module outputting `AgentOutput` named tuples, with an
      `unroll` call for computing the outputs for a whole trajectory.
    agent_state: The initial agent state for each sequence in the batch.
    env_outputs: A `StepOutput` namedtuple where each field is of shape
      [T+1, ...].
    agent_outputs: An `AgentOutput` namedtuple where each field is of shape
      [T+1, ...].

  Returns:
    A tuple of (done, infos, and environment frames) where
    the environment frames tensor causes an update.
  """
  learner_outputs = agent.unroll(agent_outputs.action, env_outputs)
  agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
  rewards, infos, done, _ = nest.map_structure(
      lambda t: t[1:], env_outputs)

  learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)
  returns = discount_returns(env_outputs)
  
  # Compute loss as a weighted sum of the baseline loss, the policy gradient
  # loss and an entropy regularization term.
  total_loss = compute_policy_gradient_loss(
                  learner_outputs.logits, 
                  agent_outputs.action,
                  returns)
  #total_loss += FLAGS.baseline_cost * compute_baseline_loss(
  #    returns - learner_outputs.baseline)
  
   # Optimization
  num_env_frames = tf.train.get_global_step()
  learning_rate = tf.train.polynomial_decay(FLAGS.learning_rate, num_env_frames,
                                            FLAGS.total_environment_frames, 0)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  #train_op = optimizer.minimize(total_loss)
  train_op = tf.contrib.training.create_train_op(total_loss, optimizer,summarize_gradients=True)


  # Merge updating the network and environment frames into a single tensor.
  with tf.control_dependencies([train_op]):
    num_env_frames_and_train = num_env_frames.assign_add(FLAGS.unroll_length)

  # Adding a few summaries.
  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('total_loss', total_loss)
  tf.summary.histogram('action', agent_outputs.action)
  return done, infos, num_env_frames_and_train

def find_size(game):
  env = gym.make(game)
  action_size = env.action_space.n
  state_size = env.observation_space.shape[0]
  return action_size, state_size

def train(game_name):
  action_size, state_size = find_size(game_name)
  """Train."""
  if is_single_machine():
    local_job_device = ''
    shared_job_device = ''
    is_actor_fn = lambda i: True
    is_learner = True
    global_variable_device = '/gpu'
    server = tf.train.Server.create_local_server()
    filters = []
  else:
    pass

  # Only used to find the actor output structure.
  with tf.Graph().as_default():
    agent = Agent(action_size)
    env = create_environment(game_name, state_size)
    structure = build_actor(agent, env, game_name, action_size)
    flattened_structure = nest.flatten(structure)
    dtypes = [t.dtype for t in flattened_structure]
    shapes = [t.shape.as_list() for t in flattened_structure]

  with tf.Graph().as_default(), \
       tf.device(local_job_device + '/cpu'), \
       pin_global_variables(global_variable_device):
    tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.
  
    with tf.device(shared_job_device):
      agent = Agent(action_size)

    tf.logging.info('Creating actor with game %s', game_name)
    env = create_environment(game_name, state_size)
    actor_output = build_actor(agent, env, game_name, action_size)
    # Create global step, which is the number of environment frames processed.
    tf.get_variable(
      'num_environment_frames',
      initializer=tf.zeros_initializer(),
      shape=[],
      dtype=tf.int64,
      trainable=False,
      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])

    actor_output = nest.map_structure(lambda t: tf.expand_dims(t, 0),
                                            actor_output)

    def make_time_major(s):
      return nest.map_structure(
          lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]), s)

    actor_output = actor_output._replace(
          env_outputs=make_time_major(actor_output.env_outputs),
          agent_outputs=make_time_major(actor_output.agent_outputs))

    with tf.device('/gpu'):
      # Using StagingArea allows us to prepare the next batch and send it to
      # the GPU while we're performing a training step. This adds up to 1 step
      # policy lag.
      flattened_output = nest.flatten(actor_output)
      area = tf.contrib.staging.StagingArea(
          [t.dtype for t in flattened_output],
          [t.shape for t in flattened_output])
      stage_op = area.put(flattened_output)

      data_from_actors = nest.pack_sequence_as(structure, area.get())

    output = build_learner(agent, 
                           data_from_actors.env_outputs,
                           data_from_actors.agent_outputs)

     # Create MonitoredSession (to run the graph, checkpoint and log).
    tf.logging.info('Creating MonitoredSession, is_chief %s', is_learner)
    config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
    with tf.train.MonitoredTrainingSession(
        server.target,
        is_chief=is_learner,
        checkpoint_dir=FLAGS.logdir,
        save_checkpoint_secs=600,
        save_summaries_secs=30,
        log_step_count_steps=50000,
        config=config,
        hooks=[py_process.PyProcessHook()]) as session:
      
      # Logging.
      level_returns = {game_name: []}
      summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

      # Prepare data for first run.
      session.run_step_fn(
          lambda step_context: step_context.session.run(stage_op))

      # Execute learning and track performance.
      num_env_frames_v = 0
      while num_env_frames_v < FLAGS.total_environment_frames:
        level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
            (actor_output.level_name,) + output + (stage_op,))
        level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)
        for level_name, episode_return, episode_step in zip(
            level_names_v[done_v],
            infos_v.episode_return[done_v],
            infos_v.episode_step[done_v]):
          level_name = level_name.decode() 
          episode_frames = episode_step 

          tf.logging.info('Level: %s Episode return: %f',
                          level_name, episode_return)
          #tf.logging.info('Level: %s Episode frames: %f',
          #                level_name, episode_frames)

          summary = tf.summary.Summary()
          summary.value.add(tag=level_name + '/episode_return',
                            simple_value=episode_return)
          summary.value.add(tag=level_name + '/episode_frames',
                            simple_value=episode_frames)
          summary_writer.add_summary(summary, num_env_frames_v)

          level_returns[level_name].append(episode_return)


def test(game_name):
  all_returns = {game_name: []}
  action_size = 4
  with tf.Graph().as_default():
    agent = Agent(action_size)
    env = create_environment(game_name)
    output = build_actor(agent, env, game_name, action_size)

    with tf.train.SingularMonitoredSession(
          checkpoint_dir=FLAGS.logdir,
          hooks=[py_process.PyProcessHook()]) as session:
      while True:
        done_v, infos_v = session.run((
          output.env_outputs.done,
          output.env_outputs.info
        ))
        returns = all_returns[game_name]
        returns.extend(infos_v.episode_return[1:][done_v[1:]])

        if len(returns) >= FLAGS.test_num_episodes:
          tf.logging.info('Mean episode return: %f', np.mean(returns))
          break

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  game_name = FLAGS.game


  if FLAGS.mode == 'train':
    train(game_name)
  else:
    test(game_name)


if __name__ == '__main__':
  tf.app.run()

