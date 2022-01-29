import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import gym
import numpy as np
import progressbar
import random
import tensorflow as tf
import time

import wandb
import d3rlpy

from variational_koopman.replay_memory import ReplayMemory
from variational_koopman.variational_koopman_model import VariationalKoopman
from koopman_data_aug_new import data_generate
# from utils import visualize_predictions, perform_rollouts

def training(config = None):

    with wandb.init(config=config):
        ######################################
        #          General Params            #
        ######################################
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_dir',       type=str,   default='./checkpoints', help='directory to store checkpointed models')
        parser.add_argument('--val_frac',       type=float, default=0.1,        help='fraction of data to be witheld in validation set')
        parser.add_argument('--ckpt_name',      type= str,  default='',         help='name of checkpoint file to load (blank means none)')
        parser.add_argument('--save_name',      type= str,  default='dvk', help='name of checkpoint files for saving')
        parser.add_argument('--domain_name',    type= str,  default='custom', help='environment name')

        parser.add_argument('--seq_length',     type=int,   default= 32,        help='sequence length for training')
        parser.add_argument('--mpc_horizon',    type=int,   default= 16,        help='horizon to consider for MPC')
        parser.add_argument('--batch_size',     type=int,   default= 32,        help='minibatch size')
        parser.add_argument('--latent_dim',     type=int,   default= 40,         help='dimensionality of code')

        parser.add_argument('--num_epochs',     type=int,   default= 60,       help='number of epochs')
        parser.add_argument('--learning_rate',  type=float, default= 0.0005,    help='learning rate')
        parser.add_argument('--decay_rate',     type=float, default= 0.5,       help='decay rate for learning rate')
        parser.add_argument('--l2_regularizer', type=float, default= 1.0,       help='regularization for least squares')
        parser.add_argument('--grad_clip',      type=float, default= 5.0,       help='clip gradients at this value')

        parser.add_argument('--n_trials',       type=int,   default= 1000,       help='number of data sequences to collect in each episode')
        parser.add_argument('--trial_len',      type=int,   default= 256,       help='number of steps in each trial')
        parser.add_argument('--n_subseq',       type=int,   default= 12,        help='number of subsequences to divide each sequence into')
        parser.add_argument('--kl_weight',      type=float, default= 0,       help='weight applied to kl-divergence loss')

        ######################################
        #          Network Params            #
        ######################################
        parser.add_argument('--extractor_size', nargs='+', type=int, default=[512, 256, 256],    help='hidden layer sizes in feature extractor/decoder')
        parser.add_argument('--inference_size', nargs='+', type=int, default=[512, 256, 256],    help='hidden layer sizes in feature inference network')
        parser.add_argument('--prior_size',     nargs='+', type=int, default=[512, 256],    help='hidden layer sizes in prior network')
        parser.add_argument('--rnn_size',       type=int,   default= 512,        help='size of RNN layers')
        parser.add_argument('--transform_size', type=int,   default= 512,        help='size of transform layers')
        parser.add_argument('--reg_weight',     type=float, default= 1e-4,      help='weight applied to regularization losses')
        parser.add_argument('--seed',           type=int,   default= 1,         help='random seed for sampling operations')

        #####################################
        #       Addtitional Options         #
        #####################################
        parser.add_argument('--ilqr',           type=bool,  default= False,     help='whether to perform ilqr with the trained model')
        parser.add_argument('--evaluate',       type=bool,  default= False,     help='whether to evaluate trained network')
        parser.add_argument('--perform_mpc',    type=bool,  default= False,     help='whether to perform MPC instead of training')
        parser.add_argument('--worst_case',     type=bool,  default= False,     help='whether to optimize for worst-case cost')
        parser.add_argument('--gamma',          type=float, default= 1.0,       help='discount factor')
        parser.add_argument('--num_models',     type=int,   default= 5,         help='number of models to use in MPC')

        args = parser.parse_args()

        # Set random seed
        random.seed(1)
        # Test
        # Create environment
        # env = gym.make(args.domain_name)

        env_mode = "hopper-medium-v0"
        dataset, env = d3rlpy.datasets.get_dataset(env_mode)

        x,u,config = data_generate(env, dataset)

        # Find state and action dimensionality from environment
        args.state_dim = env.observation_space.shape[0] + 1 #  add reward, dim +1

        # if args.domain_name == 'CartPole-v1': args.state_dim += 1 # taking sine and cosine of theta
        args.action_dim = env.action_space.shape[0]
        # args.action_max = env.action_space.high[0]
        config = wandb.config

        args.learning_rate = config.learning_rate
        args.latent_dim = int(config.latent_dim)
        # args.extractor_size = config.extractor_size
        # args.inference_size = config.inference_size
        # args.prior_size = config.prior_size
        args.rnn_size = config.rnn_size
        args.transform_size = config.transform_size
        args.reg_weight = config.reg_weight

        net = VariationalKoopman(args)
        epoch = train(args, net, x, u) ## we get epoch value because train will automatically end when the loss doesnot further decrease

# Train network
def train(args, net, x, u):

    # Begin tf session
    with tf.Session() as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Generate data
        shift = sess.run(net.shift)
        scale = sess.run(net.scale)
        shift_u = sess.run(net.shift_u)
        scale_u = sess.run(net.scale_u)

        # Generate training data
        replay_memory = ReplayMemory(args, shift, scale, shift_u, scale_u,  net, sess, x, u, predict_evolution=True)

        #Function to evaluate loss on validation set
        def val_loss(kl_weight):
            replay_memory.reset_batchptr_val()
            loss = 0.0
            for b in range(replay_memory.n_batches_val):
                # Get inputs
                batch_dict = replay_memory.next_batch_val()
                x = batch_dict['states']
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
                feed_in[net.u] = u
                feed_in[net.kl_weight] = kl_weight

                # Find loss
                feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss/replay_memory.n_batches_val

        # Specify number of times to loop through training procedure
        n_loops = 10 if args.ilqr else 1
        for n in range(n_loops):
            # Re-initialize network parameters if after the first loop (in place of fine-tuning since training is cheap)
            if n >= 1:
                tf.global_variables_initializer().run()

            # Store normalization parameters
            sess.run(tf.assign(net.shift, replay_memory.shift_x))
            sess.run(tf.assign(net.scale, replay_memory.scale_x))
            sess.run(tf.assign(net.shift_u, replay_memory.shift_u))
            sess.run(tf.assign(net.scale_u, replay_memory.scale_u))

            # Initialize variable to track validation score over time
            old_score = 1e20

            # Set initial learning rate and weight on kl divergence
            print('setting learning rate to ', args.learning_rate)
            sess.run(tf.assign(net.learning_rate, args.learning_rate))
            lr = args.learning_rate

            # Define temperature for annealing kl_weight
            anneal_time = 5
            T = anneal_time*replay_memory.n_batches_train

            # Define counting variables
            count = 0
            count_decay = 0
            decay_epochs = []

            # Loop over epochs
            for e in range(args.num_epochs):
                # visualize_predictions(args, sess, net, replay_memory, env, e)

                # Initialize loss
                loss = 0.0
                kl_loss = 0.0
                pred_loss = 0.0
                loss_count = 0
                b = 0
                replay_memory.reset_batchptr_train()

                # Loop over batches
                while b < replay_memory.n_batches_train:
                    start = time.time()

                    # Update kl_weight
                    if e < 3:
                        kl_weight = 1e-6
                    else:
                        count += 1
                        kl_weight = min(args.kl_weight, 1e-6 + args.kl_weight*count/float(T))

                    # Get inputs
                    batch_dict = replay_memory.next_batch_train()
                    x = batch_dict['states']
                    u = batch_dict['inputs']

                    # Construct inputs for network
                    feed_in = {}
                    feed_in[net.x] = np.reshape(x, (2*args.batch_size*args.seq_length, args.state_dim))
                    feed_in[net.u] = u
                    feed_in[net.kl_weight] = kl_weight

                    # Find loss and perform training operation
                    feed_out = [net.cost, net.kl_loss, net.pred_loss, net.train]
                    out = sess.run(feed_out, feed_in)

                    # Update and display cumulative losses
                    loss += out[0]
                    kl_loss += out[1]
                    pred_loss += out[2]
                    loss_count += 1

                    end = time.time()
                    b += 1

                    # Logging
                    log_dict = dict()
                    log_dict['training_loss'] = out[0]
                    log_dict['kl_loss'] = out[1]
                    log_dict['pred_loss'] = out[2]
                    wandb.log(log_dict)

                    # Print loss
                    if (e * replay_memory.n_batches_train + b) % 1000 == 0 and b > 0:
                        print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, loss/loss_count, end - start))
                        print("{}/{} (epoch {}), pred_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, pred_loss/loss_count, end - start))
                        print("{}/{} (epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}" \
                          .format(e * replay_memory.n_batches_train + b, args.num_epochs * replay_memory.n_batches_train,
                                  e, kl_loss/loss_count, end - start))

                        print('')
                        loss = 0.0
                        kl_loss = 0.0
                        pred_loss = 0.0
                        loss_count = 0

                # Evaluate loss on validation set
                score = val_loss(kl_weight)
                print('Validation Loss: {0:f}'.format(score))

                # Logging
                log_dict = dict()
                log_dict['val_loss'] = score
                wandb.log(log_dict)

                b = 0

                # Set learning rate
                if (old_score - score) < -0.01 and e >= 8:
                    count_decay += 1
                    decay_epochs.append(e)
                    if len(decay_epochs) >= 3 and np.sum(np.diff(decay_epochs)[-2:]) == 2: break
                    lr = args.learning_rate * (args.decay_rate ** count_decay)
                    if args.ilqr and lr < 5e-5: break
                    print('setting learning rate to ', lr)
                    sess.run(tf.assign(net.learning_rate, lr))

                # Save model every epoch
                checkpoint_path = os.path.join(args.save_dir, str(args.learning_rate) + '/' + args.save_name + '.ckpt')
                saver.save(sess, checkpoint_path, global_step = e)
                print("model saved to {}".format(checkpoint_path))
            
                old_score = score
            
    return e

if __name__ == '__main__':

    # For Hyperparameter Sweeping
    sweep_config = {
        'method': 'bayes',
        'metric': {
            "name": "kl_loss",
            'goal': "maximize"
            },
        'early_terminate': {
        	"type": "hyperband",
        	"min_iter": 10000, # around 20 epoch
        	"eta": 1.5
        	}
        }
    parameters_dict = {
        "learning_rate": {
            "min": 0.00001,
            "max": 0.001
        },
        "latent_dim": {
            "min": 30,
            "max": 50
        },
        # "extractor_size": {
        #     "values": [[128, 128, 128], [256, 128, 128], [256, 256, 128], [256, 256, 256], [64, 128, 256, 128, 64], [512, 512], [512, 256], [256, 128, 64], [512, 256, 256]]
        # },
        # "inference_size": {
        #     "values": [[128, 128, 128], [256, 128, 128], [256, 256, 128], [256, 256, 256], [64, 128, 256, 128, 64], [512, 512], [512, 256], [256, 128, 64], [512, 256, 256]]
        # },
        # "prior_size": {
        #     "values": [[128, 128, 128], [256, 128, 128], [256, 256, 128], [256, 256, 256], [64, 128, 256, 128, 64], [512, 512], [512, 256], [256, 128, 64], [512, 256, 256]]
        # },
        "rnn_size": {
            "values": [128, 256, 512]
        },
        "transform_size": {
            "values": [128, 256, 512]
        },
        "reg_weight": {
            "min": 0.00001,
            "max": 0.001
        }
    }

    sweep_config['parameters'] = parameters_dict

    sweep_id = wandb.sweep(sweep_config, project = 'DVK_sweep')
    wandb.agent(sweep_id, training)


