# generate supervised learning/burn-in data using A star
from copy import copy, deepcopy
import numpy as np
import tensorflow as tf
import random

from astar import Astar
from CREnv_v0 import CREnv
from config import Params

from nn_architecures import network_builder
from models import CategoricalModel, GaussianModel
from wrapper_env import env_wrapper

def generate_data(Env, revise_prehead=False):

    env = deepcopy(Env)
    env.reset()

    data = []
    for net_idx in range(2, int(max(env.start))+1):

        env.reset()
        env.head = env.start[net_idx]

        board = copy(env.board)
        board[board==net_idx]=0
        board[board>=1]=1

        start = env.start[net_idx]
        end = env.finish[net_idx]

        astar = Astar(board)
        path = astar.run(start, end)

        path.pop(0)
        
        for node in path:
            sample = list(env.board_embedding())
            act = env.path_node_to_act_idx(node)
            sample.append(act)
            data.append(sample)
            env.step(act)

    # np.savetxt("samples.csv", data, delimiter=",")        
    return data


def pretrain(model, data, train_iters, testing=False):

    from sklearn.metrics import accuracy_score

    random.shuffle(data)

    features, labels = separate_feature_label(data)

    if testing:
        num_train = int(features.shape[0]*0.9)
        test_features = features[num_train:]
        test_labels = labels[num_train:]

        features = features[:num_train]
        labels = labels[:num_train]

    for i in range(train_iters):

        loss = train_one_step(model, features, labels)
        train_y = predict(model, features)
        train_accuracy = accuracy_score(labels, train_y)
        print("epoch is:{}, training loss is: {}, training accuracy is: {}".format(i, loss, train_accuracy))

        if testing:
            test_y = predict(model, test_features)
            test_accuracy = accuracy_score(test_labels, test_y)
            print("epoch is:{}, testing accuracy is: {}".format(i, test_accuracy))            

    return loss

def separate_feature_label(data):

    data_tem = np.array(data)
    features = data_tem[:, 0:-1]
    labels = data_tem[:, -1]

    return features, labels

def loss(model, vec_obs, labels):

    vis_obs = None

    logits, _ = model.predict({"vec_obs": vec_obs, "vis_obs": vis_obs}) 

    p_all = tf.nn.softmax(logits)
    onehot_depth = p_all.numpy().shape[1]
    labels = tf.one_hot(labels, onehot_depth)
    # print(labels)

    loss = tf.math.reduce_sum(tf.keras.losses.mean_squared_error(labels, p_all))

    return loss

# @tf.function
def train_one_step(model, vec_obs, labels):

    clip_grads = 0.5
    lr = 0.001
    optimizer = tf.keras.optimizers.Adam(learning_rate= lr)

    with tf.GradientTape() as tape:
        losses = loss(model, vec_obs, labels)
    
    # just train policy network of model
    trainable_variables = model.all_networks[0].trainable_variables
    
    grads = tape.gradient(losses, trainable_variables)

    grads, grad_norm = tf.clip_by_global_norm(grads, clip_grads)

    optimizer.apply_gradients(zip(grads, trainable_variables))

    return losses

def predict(model, vec_obs):

    vis_obs = None

    p_all = model.p_all({"vec_obs": vec_obs, "vis_obs": vis_obs})
    return np.argmax(p_all, axis=1)

if __name__ == "__main__":

    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    import sys
    import os
    boards_directory = sys.argv[1]
    board_idx = sys.argv[2]

    params = Params()          # Get Configuration | HORIZON = Steps per epoch
    board_name = "board_{}.csv".format(board_idx)
    
    tf.random.set_seed(params.env.seed)                                     # Set Random Seeds for np and tf
    np.random.seed(params.env.seed)

    env = CREnv(board_path=os.path.join(boards_directory, board_name))

    pretrain_data = generate_data(env)
    env = env_wrapper(env, params.env)

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)    # Build Neural Network with Forward Pass

    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                   # Build Model for Discrete or Continuous Spaces

    # print(separate_feature_label(pretrain_data))

    pretrain(model, pretrain_data, 50, testing=False)

    model.save_weights( 'saved_model/pretrain_model', save_format='tf')
    # print(predict(model, pretrain_data))
