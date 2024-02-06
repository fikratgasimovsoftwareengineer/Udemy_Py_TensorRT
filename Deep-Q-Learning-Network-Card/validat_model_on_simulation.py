"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

This file creates a movie that shows the performance of the trained model

"""
# you will also need to install MoviePy, and you do not need to import it explicitly
# pip install moviepy

# import Keras
import keras

# import the class
from deep_qlearning import DeepQLearning

# import gym
import gym

# numpy
import numpy as np

# load the model
loaded_model = keras.models.load_model("/tensorfl_vision/Deep-Q-Learning-Network-from-Scratch-in-Python-TensorFlow-and-OpenAI-Gym/trained_model_temp.h5",custom_objects={'my_loss_fn':DeepQLearning.my_loss_fn})


sumObtainedRewards=0
# simulate the learned policy for verification


# create the environment, here you need to keep render_mode='rgb_array' since otherwise it will not generate the movie
env = gym.make("CartPole-v1",render_mode='rgb_array')
# reset the environment
(currentState,prob)=env.reset()

# Wrapper for recording the video
# https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RenderCollection
# the name of the folder in which the video is stored is "stored_video"
# length of the video in the number of simulation steps
# if we do not specify the length, the video will be recorded until the end of the episode 
# that is, when terminalState becomes TRUE
# just make sure that this parameter is smaller than the expected number of 
# time steps within an episode
# for some reason this parameter does not produce the expected results, for smaller than 450 it gives OK results
video_length=200
# the step_trigger parameter is set to 1 in order to ensure that we record the video every step
#env = gym.wrappers.RecordVideo(env, 'stored_video',step_trigger = lambda x: x == 1, video_length=video_length)
env = gym.wrappers.RecordVideo(env, 'stored_video', video_length=video_length)


# since the initial state is not a terminal state, set this flag to false
terminalState=False
while not terminalState:
    # get the Q-value (1 by 2 vector)
    Qvalues=loaded_model.predict(currentState.reshape(1,4))
    # select the action that gives the max Qvalue
    action=np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
    # if you want random actions for comparison
    #action = env.action_space.sample()
    # apply the action
    (currentState, currentReward, terminalState,_,_) = env.step(action)
    # sum the rewards
    sumObtainedRewards+=currentReward

env.reset()
env.close()




