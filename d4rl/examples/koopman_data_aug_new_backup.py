import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import gym
import numpy as np
import random
import tensorflow as tf
import time
from d3rlpy.dataset import MDPDataset

def koop_mixup_data_aug(env,
                        dataset,
                        dvk_model_train = False,
                        n_aug = 0,
                        net_load = False,
                        args = [],
                        net = [],
                        epoch = 0,
                        use_all=False,
                        logging = False,
                        check_sweep = False,
                        dvk_model_dir='',
                        trial_len=512+256,
                        random_seed=1):

    from replay_memory import ReplayMemory
    from variational_koopman.train_variational_koopman import training
    
    np.random.seed(random_seed)

    # augmented data amount. If not set, augmented data amount = original dataset amount

    if n_aug == 0:
        n_aug = int(len(dataset.rewards))

    ## load data
    observations = dataset.observations
    actions = dataset.actions
    rewards = dataset.rewards
    terminals = dataset.terminals

    ## prepare dataset for dvk learning x : (datapoints, seq_length, x_dim), u : (datapoints, seq_length, u_dim)
    if use_all:
        x,u,config = data_generate_all(env, dataset)

    else:
        x,u,config = data_generate(env, dataset, trial_len, random_seed)

    state_dim = x.shape[2] # Observation dim + 1 for reward
    act_dim = u.shape[2]

    ## linear embedding learning
    if net_load:
        pass
    else:
        args, net, epoch = training(x,
                                    u,
                                    state_dim,
                                    act_dim,
                                    dvk_model_train = dvk_model_train,
                                    config = config,
                                    logging = logging,
                                    check_sweep=check_sweep,
                                    dvk_model_dir=dvk_model_dir,
                                    random_seed=random_seed)

    ## linear mapping
    with tf.Session() as sess:

        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if dvk_model_train:
            saver.restore(sess, os.path.join(args.save_dir, dvk_model_dir + 'dvk.ckpt-{}'.format(epoch-1))) # load learned dvk model when skip = False
        else:
            if check_sweep:
                saver.restore(sess, os.path.join(args.save_dir, '../variational_koopman/checkpoints/astral/dvk.ckpt-{}'.format(59))) # load specific learned dvk model when skip = True
            else:
                saver.restore(sess, os.path.join(args.save_dir, dvk_model_dir + "dvk.ckpt-{}".format(80))) # load specific learned dvk model when skip = True

        shift_x = net.shift.eval(session = sess)
        scale_x = net.scale.eval(session = sess)
        shift_u = net.shift_u.eval(session = sess)
        scale_u = net.scale_u.eval(session = sess)

        # normalize data before going through the network
        x = (x - shift_x)/scale_x
        u = (u - shift_u)/scale_u

        n_data = len(x)
        x = np.reshape(x, (2*n_data*args.seq_length, args.state_dim))
        u = np.reshape(u, (n_data*(2*args.seq_length-1), args.action_dim))
        z = np.zeros((0,args.latent_dim), dtype = np.float64)
        for i in range(int(n_data/args.batch_size)):
            x_temp = x[i*2*args.seq_length*args.batch_size:(i+1)*2*args.seq_length*args.batch_size, :]
            u_temp = np.reshape(u[i*(2*args.seq_length-1)*args.batch_size:(i+1)*(2*args.seq_length-1)*args.batch_size,:], (args.batch_size, 2*args.seq_length-1,args.action_dim))
            z_temp = x_to_z(sess, net, x_temp, u_temp) ## embedding network. data points are embedded with batch size
            z = np.concatenate((z, z_temp), axis = 0)    

        alpha = 0.1 ## mixup alpha value for beta distribution
        lam = np.random.beta(alpha, alpha, size = (n_aug,1))
        lam = np.float64(lam)

        # pick mix idx.  if use pick_mix_idx_traj, data pair is selected in same episode
        mix_1_idx_z, mix_2_idx_z, mix_1_idx_u, mix_2_idx_u, mix_1_idx_z_n, mix_2_idx_z_n, mix_1_idx_u_n, mix_2_idx_u_n = pick_mix_idx(int(n_data/args.batch_size)*args.batch_size, n_aug, args)
        mix_z = lam * z[mix_1_idx_z,:] + (1-lam)*z[mix_2_idx_z,:]
        mix_z_n = lam * z[mix_1_idx_z_n,:] + (1-lam)*z[mix_2_idx_z_n,:] ## _n means next. next latent state
        u = u*scale_u + shift_u
        mix_u = lam * u[mix_1_idx_u,:] + (1-lam)*u[mix_2_idx_u,:]
        mix_u_n = lam * u[mix_1_idx_u_n,:] + (1-lam)*u[mix_2_idx_u_n,:]
        mix_x = net._get_decoder_output(args, mix_z)*net.scale + net.shift  ## reconstruct linear state z to x
        mix_x_n = net._get_decoder_output(args, mix_z_n)*net.scale + net.shift ## 

        mix_x = mix_x.eval(session = sess) 
        mix_x_n = mix_x_n.eval(session = sess)


    obs, acts, rews, epi_tems, tems = to_trans(mix_x, mix_x_n, mix_u, mix_u_n)

    ## add to original dataset
    observations = np.concatenate((observations, obs), axis = 0)
    actions = np.concatenate((actions, acts), axis = 0)
    rewards = np.concatenate((rewards, rews))
    epi_terminals = np.concatenate((terminals, epi_tems))
    terminals = np.concatenate((terminals, tems))

    # genearate MDPDataset
    new_dataset = MDPDataset(observations, actions, rewards, terminals, epi_terminals)

    # episode.transitions
    print('MDP dataset generated')
    print(n_data, n_aug, len(observations))
    return new_dataset, args, net, epoch

def to_trans(mix_x, mix_x_n, mix_u, mix_u_n):
    n_aug = len(mix_x)
    
    obs = np.zeros((n_aug*2,mix_x[0].size-1)) ## s,a,r,tem, epi_tem for one timestep. two data points will make one transition (s, a, r, s')
    acts = np.zeros((n_aug*2,mix_u[0].size))
    rews = np.zeros((n_aug*2))
    tems = np.zeros(n_aug*2)
    epi_tems = np.zeros(n_aug*2)
    prev_list = np.linspace(0,n_aug*2-2,n_aug, dtype = int)
    next_list = prev_list +1


    temp = mix_x[:,:-1]
    obs[prev_list,:] = mix_x[:,:-1]
    obs[next_list,:] = mix_x_n[:,:-1]
    acts[prev_list,:] = mix_u
    acts[next_list,:] = mix_u_n
    rews[prev_list] = mix_x[:,-1]
    rews[next_list] = mix_x_n[:,-1]
    tems[prev_list] = 0
    tems[next_list] = 0 ## environment terminal is False
    epi_tems[prev_list] = 0
    epi_tems[next_list] = 1 ## episode terminal is True to make the episode has one transition.

    return obs, acts, rews, epi_tems, tems

def x_to_z(sess, net, x, u):
    # Construct inputs for network
    feed_in = {}
    feed_in[net.x] = x
    feed_in[net.u] = u
    feed_out = net.z_vals
    out = sess.run(feed_out, feed_in)

    z1 = out

    return z1

def mixup_data_aug(env, dataset, n_aug = 0):  
    ## it is for normal mixup augmentation

    # data amount
    if n_aug == 0:
        n_aug = len(dataset.rewards)
    # n_data = len(dataset.rewards)
    
    ## load data
    observations = dataset.observations
    actions = dataset.actions
    rewards = dataset.rewards
    terminals = dataset.terminals

    # how to mix
    alpha = 0.4
    lam = np.random.beta(alpha, alpha, size = (n_aug,1))

    # pick mix_idx
    x,u= data_generate(env, dataset)

    n_data = len(x)
    state_dim = x.shape[2]
    act_dim = u.shape[2]
    seq_length = int(x.shape[1]/2)
    class args():
        def __init__(self, seq_length, state_dim, act_dim):        
            self.seq_length = seq_length
            self.state_dim = state_dim
            self.action_dim = act_dim
            self.batch_size = 32
    args = args(seq_length, state_dim, act_dim)
    x = np.reshape(x, (2*n_data*args.seq_length, args.state_dim))
    u = np.reshape(u, (n_data*(2*args.seq_length-1), args.action_dim))    
    mix_1_idx_z, mix_2_idx_z, mix_1_idx_u, mix_2_idx_u, mix_1_idx_z_n, mix_2_idx_z_n, mix_1_idx_u_n, mix_2_idx_u_n = pick_mix_idx(int(n_data/args.batch_size)*args.batch_size, n_aug, args)

    mix_x = lam * x[mix_1_idx_z,:] + (1-lam)*x[mix_2_idx_z,:]
    mix_x_n = lam * x[mix_1_idx_z_n,:] + (1-lam)*x[mix_2_idx_z_n,:]
    mix_u = lam * u[mix_1_idx_u,:] + (1-lam)*u[mix_2_idx_u,:]
    mix_u_n = lam * u[mix_1_idx_u_n,:] + (1-lam)*u[mix_2_idx_u_n,:]

    obs, acts, rews, epi_tems, tems = to_trans(mix_x, mix_x_n, mix_u, mix_u_n)

    ## add to original dataset
    observations = np.concatenate((observations, obs), axis = 0)
    actions = np.concatenate((actions, acts), axis = 0)
    rewards = np.concatenate((rewards, rews))
    epi_terminals = np.concatenate((terminals, epi_tems))
    terminals = np.concatenate((terminals, tems))

    # genearate MDPDataset
    new_dataset = MDPDataset(observations, actions, rewards, terminals, epi_terminals)

    # episode.transitions
    print('MDP dataset generated')
    print(n_data, n_aug, len(observations))
    return new_dataset

def pick_mix_idx(n_data, n_aug, args):
    mix_1_idx = np.random.randint(0, n_data-2, size = n_aug)
    mix_2_idx = np.random.randint(0, n_data-2, size = n_aug)

    mix_1_idx_z = mix_1_idx * args.seq_length
    mix_2_idx_z = mix_2_idx * args.seq_length

    mix_1_idx_z_n = mix_1_idx_z +1
    mix_2_idx_z_n = mix_2_idx_z +1

    mix_1_idx_u = mix_1_idx * (args.seq_length*2 -1)
    mix_2_idx_u = mix_2_idx * (args.seq_length*2 -1)

    mix_1_idx_u_n = mix_1_idx_u +1
    mix_2_idx_u_n = mix_2_idx_u +1

    return mix_1_idx_z, mix_2_idx_z, mix_1_idx_u, mix_2_idx_u, mix_1_idx_z_n, mix_2_idx_z_n, mix_1_idx_u_n, mix_2_idx_u_n

def pick_mix_idx_in_traj(n_data, n_aug, args):
    seq = np.random.randint(0, n_data-1, size = n_aug)
    mix_1_idx = np.random.randint(0, args.seq_length-2, size = n_aug)
    mix_2_idx = np.random.randint(0, 1, size = n_aug)
    for i in range(n_aug):
        mix_2_idx[i] = np.random.randint(1, args.seq_length-1-mix_1_idx[i], size = 1)
    # mix_2_idx = np.random.randint(1, args.seq_length-2, size = n_aug)

    mix_1_idx_z = seq * args.seq_length + mix_1_idx
    mix_2_idx_z = mix_1_idx_z + mix_2_idx

    mix_1_idx_z_n = mix_1_idx_z +1
    mix_2_idx_z_n = mix_2_idx_z +1

    mix_1_idx_u = seq * (args.seq_length*2 -1) + mix_1_idx
    mix_2_idx_u = seq + mix_2_idx

    mix_1_idx_u_n = mix_1_idx_u +1
    mix_2_idx_u_n = mix_2_idx_u +1

    return mix_1_idx_z, mix_2_idx_z, mix_1_idx_u, mix_2_idx_u, mix_1_idx_z_n, mix_2_idx_z_n, mix_1_idx_u_n, mix_2_idx_u_n

def data_generate(env, dataset, trial_len, random_seed):

    np.random.seed(random_seed)

    batch_size = 32 ## for DVK learning
    seq_length = 64*2 ## for DVK learning

    n_subseq = trial_len - seq_length - 1
    state_dim = env.observation_space.shape[0] + 1 #  add reward, dim +1
    action_dim = env.action_space.shape[0]

    # Counting # of available episodes that is longer than trial_len
    len_traj = np.zeros((len(dataset.episodes)))
    for i in range(len(dataset.episodes)):
        len_traj[i] = len(dataset.episodes[i])

    count = 0
    traj_index = []
    for index, item in enumerate(len_traj):
        if item >= trial_len:
            traj_index.append(index)
            count = count + 1

    print('n_trials : {}'.format(count))

    n_trials = count

    ## For logging
    config = dict(
        env_name = str(env).split()[1].lstrip("<"),
        seq_length = seq_length,
        trial_len = trial_len,
        n_subseq = n_subseq,
        active_traj = n_trials,
    )

    # Initialize array to hold states and actions
    x = np.zeros((n_trials, n_subseq, seq_length, state_dim), dtype=np.float64)
    u = np.zeros((n_trials, n_subseq, seq_length-1, action_dim), dtype=np.float64)

    # Define array for dividing trials into subsequences
    stagger = (trial_len - seq_length)/n_subseq
    start_idxs = np.linspace(0, stagger*n_subseq, n_subseq)

    sample_num = 0
    idx = 0
    np.random.shuffle(traj_index)

    # Loop through episodes
    for i in range(n_trials):
        # Define arrays to hold observed states and actions in each trial
        x_trial = np.zeros((trial_len, state_dim), dtype=np.float64)
        u_trial = np.zeros((trial_len-1, action_dim), dtype=np.float64)

        # Reset environment and simulate with random actions
        # x_trial[0] =  dataset.episodes[traj_index[i]].observations[0]
        # when reward adding
        x_trial[0,0:-1] = dataset.episodes[traj_index[i]].observations[0]
        x_trial[0,-1] = dataset.episodes[traj_index[i]].rewards[1] ## hopper rewards start with 0?
        for t in range(1, trial_len):
            action = dataset.episodes[traj_index[i]].actions[t-1]
            u_trial[t-1] = action
            # x_trial[t] = dataset.episodes[traj_index[i]].observations[t]

            x_trial[t, 0:-1] = dataset.episodes[traj_index[i]].observations[t]
            x_trial[t, -1] = dataset.episodes[traj_index[i]].rewards[t]

        # Divide into subsequences
        for j in range(n_subseq):
            x[i, j] = x_trial[int(start_idxs[j]):(int(start_idxs[j])+seq_length)]
            u[i, j] = u_trial[int(start_idxs[j]):(int(start_idxs[j])+seq_length-1)]

        sample_num += n_subseq
        idx += 1
        if sample_num > 50000: # You can change this number to whatever you want
            break

    x = x[:idx]
    u = u[:idx]

    # Reshape and trim data sets
    x = x.reshape(-1, seq_length, state_dim)
    u = u.reshape(-1, seq_length-1, action_dim)

    print("================================")
    print("Total Number of Sub Sequences: ", sample_num)
    print("================================")

    return x, u, config

def data_generate_all(env, dataset, trial_len, random_seed):
    np.random.seed(random_seed)

    seq_length = 64*2 ## for DVK learning
    # criterion = seq_length*2

    state_dim = env.observation_space.shape[0] + 1 #  add reward, dim +1
    action_dim = env.action_space.shape[0]

    ## For logging
    config = dict(
        env_name = str(env).split()[1].lstrip("<"),
        seq_length = seq_length,
    )

    x = np.zeros((1, seq_length, state_dim), dtype=np.float64)
    u = np.zeros((1, seq_length-1, action_dim), dtype=np.float64)

    # Counting # of available episodes that is longer than trial_len
    len_traj = np.zeros((len(dataset.episodes)))
    for i in range(len(dataset.episodes)):
        len_traj[i] = len(dataset.episodes[i])

    count = 0
    traj_index = []
    for index, item in enumerate(len_traj):
        if item >= trial_len:
            traj_index.append(index)
            count = count + 1

    print('n_trials : {}'.format(count))

    n_trials = count
    np.random.shuffle(traj_index)

    for i in range(n_trials):
        target_episode = dataset.episodes[traj_index[i]]

        trial_len = len(target_episode.observations)

        # if len(target_episode.observations) <= criterion:
        #     continue

        n_subseq = trial_len - seq_length - 1
        stagger = (trial_len - seq_length)/n_subseq
        start_idxs = np.linspace(0, stagger*n_subseq, n_subseq)

        x_trial = np.zeros((len(target_episode.observations), state_dim), dtype=np.float64)
        u_trial = np.zeros((len(target_episode.actions)-1, action_dim), dtype=np.float64)

        x_trial[0, 0:-1] = target_episode.observations[0]
        x_trial[0, -1] = target_episode.rewards[1]

        for t in range(1, x_trial.shape[0]):
            action = target_episode.actions[t-1]
            u_trial[t-1] = action

            x_trial[t, 0:-1] = target_episode.observations[t]

            if t == x_trial.shape[0]-1:
                x_trial[t, -1] = 0
            else:
                x_trial[t, -1] = target_episode.rewards[t+1]

        for j in range(n_subseq):
            if i == 0 and j == 0:
                x[i] = x_trial[int(start_idxs[j]):(int(start_idxs[j])+seq_length)]
                u[i] = u_trial[int(start_idxs[j]):(int(start_idxs[j])+seq_length)-1]
                continue
            temp_x = x_trial[int(start_idxs[j]):(int(start_idxs[j])+seq_length)]
            temp_u = u_trial[int(start_idxs[j]):(int(start_idxs[j])+seq_length)-1]

            x = np.concatenate((x, temp_x[None,:,:]), axis = 0)
            u = np.concatenate((u, temp_u[None,:,:]), axis = 0)

    print("================================")
    print(n_trials, " Trajectories selected")
    print("Total Number of Sub Sequences: ", len(x))
    print("================================")

    return x, u, config

def check_dataset(dataset, traj_len = 0):
    len_traj = np.zeros((len(dataset.episodes)))
    for i in range(len(dataset.episodes)):
        len_traj[i] = len(dataset.episodes[i])
    print('min traj length :', min(len_traj), 'max traj length :', max(len_traj), 'mean traj length :', np.mean(len_traj))

    if traj_len == 0:
        traj_len = np.mean(len_traj)

    count = 0
    traj_index = []
    for index, item in enumerate(len_traj):
        if item >= traj_len:
            count = count +1
            traj_index.append(index)

    print('count :', count)
    return count
