"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial
 
The class DeepQLearning implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library
"""
# import the class
from deep_qlearning import DeepQLearning

# classical gym 
import gym
# instead of gym, import gymnasium 
#import gymnasium as gym

# create environment
env=gym.make('CartPole-v1')

# select the parameters
gamma=1
# probability parameter for the epsilon-greedy approach
epsilon=0.1
# number of training episodes
# NOTE HERE THAT AFTER CERTAIN NUMBERS OF EPISODES, WHEN THE PARAMTERS ARE LEARNED
# THE EPISODE WILL BE LONG, AT THAT POINT YOU CAN STOP THE TRAINING PROCESS BY PRESSING CTRL+C
# DO NOT WORRY, THE PARAMETERS WILL BE MEMORIZED
numberEpisodes=400

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
LearningQDeep.mainNetwork.save("trained_model_temp.h5")



