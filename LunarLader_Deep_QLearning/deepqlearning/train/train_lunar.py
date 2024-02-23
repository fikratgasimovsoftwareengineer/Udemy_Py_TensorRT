"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial
 
 *****LUDAR LANDING PROBLEM :TRAINING SESSION*****
The class DeepQLearning implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library
"""

import sys
from pathlib import Path

base_dir = Path('/tensorfl_vision/LunarLader_Deep_QLearning')

sys.path.append(str(base_dir))
# import the class
from deepqlearning.algorithms.Deep_Lunar_Q import DeepQLearning

# classical gym 
import gym
# instead of gym, import gymnasium 
#import gymnasium as gym

# create environment
env=gym.make("LunarLander-v2")



# select the parameters
gamma=0.995
# probability parameter for the epsilon-greedy approach
epsilon=0.1 # initial ε value for ε-greedy policy
# number of training episodes

numberEpisodes=100

# create an object
LearningQDeep=DeepQLearning(env,gamma,epsilon,numberEpisodes)

# run the learning process
LearningQDeep.trainingEpisodes()
# get the obtained rewards in every episode
LearningQDeep.sumRewardsEpisode

#  summarize the model
LearningQDeep.mainNetwork.summary()
# save the model, this is important, since it takes long time to train the model 
# and we will need model in another file to visualize the trained model performance
LearningQDeep.mainNetwork.save(base_dir  / "data/input/pretrained_lunar.h5")



