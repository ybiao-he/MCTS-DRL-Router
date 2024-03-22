from config import (BATCH_SIZE, CLIP_REWARD, DISCOUNT_FACTOR,
                    EVAL_LENGTH, FRAMES_BETWEEN_EVAL, INPUT_SHAPE,
                    LEARNING_RATE, LOAD_FROM, LOAD_REPLAY_BUFFER, MAX_NOOP_STEPS, MEM_SIZE,
                    MIN_REPLAY_BUFFER_SIZE, PRIORITY_SCALE, SAVE_PATH,
                    TENSORBOARD_DIR, TOTAL_FRAMES, UPDATE_Q, UPDATE_T, USE_PER,
                    WRITE_TENSORBOARD)

import numpy as np

import random
import os
import json
import time

import gym

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop

from memory import ReplayBuffer
from agent import Agent


def build_q_network(n_actions, learning_rate=0.00001, input_shape=(5,), history_length=1):
    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=input_shape)
    # CNN model
    x = Lambda(lambda layer: layer / 30)(model_input)  # normalize by 255

    x = Dense(32, kernel_initializer=VarianceScaling(scale=2.), activation='tanh')(model_input)
    x = Dense(32, kernel_initializer=VarianceScaling(scale=2.), activation='tanh')(x)

    # Split into value and advantage streams
    # val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 1))(x)  # custom splitting layer

    # val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    # adv_stream = Flatten()(adv_stream)
    # adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    # reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean

    # q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    q_vals = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(x)

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model

# Create environment
if __name__ == "__main__":

    from CREnv import CREnv

    env = CREnv()
    print("The environment has the following {} actions: {}".format(env.action_space.n, env.directions))

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # Build main and target networks
    MAIN_DQN = build_q_network(env.action_space.n, LEARNING_RATE, input_shape=INPUT_SHAPE)
    TARGET_DQN = build_q_network(env.action_space.n, input_shape=INPUT_SHAPE)

    replay_buffer = ReplayBuffer(size=MEM_SIZE, input_shape=INPUT_SHAPE, use_per=USE_PER)
    agent = Agent(MAIN_DQN, TARGET_DQN, replay_buffer, env.action_space.n, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, use_per=USE_PER)

    # Training and evaluation
    if LOAD_FROM is None:
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

        # Apply information loaded from meta
        frame_number = meta['frame_number']
        rewards = meta['rewards']
        loss_list = meta['loss_list']

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                # Training

                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    start_time = time.time()
                    obs = env.reset()
                    done = False
                    episode_reward_sum = 0
                    steps = 0
                    while not done:

                        action = agent.get_action(frame_number, obs, env.ill_action)

                        new_obs, reward, done, info = env.step(action)
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        steps += 1
                        # Add experience to replay memory
                        agent.add_experience(action=action,
                                            frame=obs,
                                            reward=reward, clip_reward=CLIP_REWARD,
                                            terminal=done)
                        obs = new_obs
                        # print(done)
                        # Update agent
                        if frame_number % UPDATE_Q == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            # print("test learning----")
                            loss, _ = agent.learn(BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number, priority_scale=PRIORITY_SCALE)
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % UPDATE_T == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()

                    print(steps)

                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print(f'Game number: {str(len(rewards)).zfill(6)}  Frame number: {str(frame_number).zfill(8)}  Average reward: {np.mean(rewards[-10:]):0.1f}  Time taken: {(time.time() - start_time):.1f}s')

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                terminal = False
                eval_rewards = []
                evaluate_frame_number = 0
                obs_eval = env.reset()

                for _ in range(EVAL_LENGTH):
                    obs_eval = env.reset()
                    episode_reward_sum = 0
                    terminal = False
                    while not terminal:

                        # Breakout requires a "fire" action (action #1) to start the
                        # game each time a life is lost.
                        # Otherwise, the agent would sit around doing nothing.
                        action = agent.get_action(frame_number, obs_eval, env.ill_action, evaluation=True)

                        # Step action
                        obs_eval, reward, terminal, info = env.step(action)
                        evaluate_frame_number += 1
                        episode_reward_sum += reward

                    # On game-over
                    eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum
                # Print score and write to tensorboard
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # # Save model
                # if len(rewards) > 300 and SAVE_PATH is not None:
                #     agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
    except KeyboardInterrupt:
        print('\nTraining exited early.')
        writer.close()

        # if SAVE_PATH is None:
        #     try:
        #         SAVE_PATH = input('Would you like to save the trained model? If so, type in a save path, otherwise, interrupt with ctrl+c. ')
        #     except KeyboardInterrupt:
        #         print('\nExiting...')

        # if SAVE_PATH is not None:
        #     print('Saving...')
        #     agent.save(f'{SAVE_PATH}/save-{str(frame_number).zfill(8)}', frame_number=frame_number, rewards=rewards, loss_list=loss_list)
        #     print('Saved.')
