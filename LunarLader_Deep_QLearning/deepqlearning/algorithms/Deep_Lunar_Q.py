 # -*- coding: utf-8 -*-
"""
Deep Learning Reinforcement Tutorial: Deep Q Network (DQN) = Combination of Deep Learning and Q-Learning Tutorial

The class developed in this file implements the Deep Q Network (DQN) Reinforcement Learning Algorithm.
The implementation is based on the OpenAI Gym Cart Pole environment and TensorFlow (Keras) machine learning library

The webpage explaining the codes and the main idea of the DQN is given here:



"""
# import the necessary libraries
import numpy as np
import random
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from collections import deque 
from tensorflow import gather_nd
from tensorflow.keras.losses import mean_squared_error 
import tensorflow as tf


class DeepQLearning:
    
    ###########################################################################
    #   START - __init__ function
    ###########################################################################
    # INPUTS: 
    # env - LUNAR LANDING PROBLEM
    # gamma - discount rate
    # epsilon - parameter for epsilon-greedy approach
    # numberEpisodes - total number of simulation episodes
    
            
    def __init__(self,env,gamma,epsilon,numberEpisodes):
    
        self.env=env
        self.gamma=gamma
        self.epsilon=epsilon
        self.numberEpisodes=numberEpisodes
        
        # state dimension
        self.stateDimension= self.env.observation_space.shape[0]
        
        
        # action dimension
        self.actionDimension=self.env.action_space.n
        
        print(f"state size {self.stateDimension} and action space {self.actionDimension}")        
        # this is the maximum size of the replay buffer
        self.replayBufferSize=300 # 300
        # this is the size of the training batch that is randomly sampled from the replay buffer
        self.batchReplayBufferSize=100 # 100
        # number of training episodes it takes to update the target network parameters
        # that is, every updateTargetNetworkPeriod we update the target network parameters
        self.updateTargetNetworkPeriod=20 #100
        
        # this is the counter for updating the target network 
        # if this counter exceeds (updateTargetNetworkPeriod-1) we update the network 
        # parameters and reset the counter to zero, this process is repeated until the end of the training process
        self.counterUpdateTargetNetwork=0
        
        # this sum is used to store the sum of rewards obtained during each training episode
        self.sumRewardsEpisode=[]
        
        # replay buffer
        self.replayBuffer=deque(maxlen=self.replayBufferSize)
        
        # this is the main network
        # create network
        self.mainNetwork=self.onlineNetwork()
        
        # this is the target network
        # create network
        self.targetNetwork=self.targetNetworkUpdate()
        
        # copy the initial weights to targetNetwork
        self.targetNetwork.set_weights(self.mainNetwork.get_weights())
        
        # this list is used in the cost function to select certain entries of the 
        # predicted and true sample matrices in order to form the loss
        self.actionsAppend=[]
  
    
    '''
        CALCULATE LOSS PREDICTED Q VALUES FROM MAIN AND TARGET NETWORJ
    '''
    def custom_loss(self, y_true, y_pred):
        # Get the shape of the prediction tensor
        batch_size = tf.shape(y_true)[0]
        
        # Create an index for each sample in the batch
        batch_indices = tf.range(0, batch_size)
        
        # Pair each batch index with the corresponding action index
        indices = tf.stack([batch_indices, self.actionsAppend], axis=1)
        
        # Use gather_nd to select the predicted Q-value for each action taken
        pred_q_values_for_actions = tf.gather_nd(y_pred, indices)
        
        # Similarly, select the target Q-value for each action taken
        target_q_values_for_actions = tf.gather_nd(y_true, indices)
        
        # Compute the loss as the mean squared error of the selected Q-values
        loss = mean_squared_error(target_q_values_for_actions, pred_q_values_for_actions)
        return loss   

    
    # create a neural network
    def targetNetworkUpdate(self):
        model=Sequential()
        model.add(Dense(128,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(56,activation='relu'))
        model.add(Dense(self.actionDimension,activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer = Adam(learning_rate=1e-3), loss = self.custom_loss, metrics = ['accuracy'])
        return model
    
    
    
    # create a neural network
    def onlineNetwork(self):
        model=Sequential()
        model.add(Dense(128,input_dim=self.stateDimension,activation='relu'))
        model.add(Dense(56,activation='relu'))
        model.add(Dense(self.actionDimension,activation='linear'))
        # compile the network with the custom loss defined in my_loss_fn
        model.compile(optimizer = Adam(learning_rate=1e-5), loss = self.custom_loss, metrics = ['accuracy'])
        return model
  

    def trainingEpisodes(self):
   
        
        # here we loop through the episodes
        for indexEpisode in range(self.numberEpisodes):
            
            # list that stores rewards per episode - this is necessary for keeping track of convergence 
            rewardsEpisode=[]
                       
            print("Simulating episode {}".format(indexEpisode))
            
            # reset the environment at the beginning of every episode
            (currentState,_)=self.env.reset()
                      
            # here we step from one state to another
            # this will loop until a terminal state is reached
            terminalState=False
            
            while not terminalState:
                                      
                # select an action on the basis of the current state, denoted by currentState
                action = self.selectAction(currentState,indexEpisode)
                
                # here we step and return the state, reward, and boolean denoting if the state is a terminal state
                (nextState, reward, terminalState,_,_) = self.env.step(action)          
                rewardsEpisode.append(reward)
         
                # add current state, action, reward, next state, and terminal flag to the replay buffer
                self.replayBuffer.append((currentState,action,reward,nextState,terminalState))
                
                # train network
                self.trainNetwork()
                
                # set the current state for the next step
                currentState=nextState
            
            print("Sum of rewards {}".format(np.sum(rewardsEpisode)))        
            self.sumRewardsEpisode.append(np.sum(rewardsEpisode))
    ###########################################################################
    #   END - function trainingEpisodes()
    ###########################################################################
            
   
    def selectAction(self,state,index):
        import numpy as np
        
        # first index episodes we select completely random actions to have enough exploration
        # change this
        if index<1:
            return np.random.choice(self.actionDimension)   
            
        # Returns a random real number in the half-open interval [0.0, 1.0)
        # this number is used for the epsilon greedy approach
        randomNumber=np.random.random()
        
        # after index episodes, we slowly start to decrease the epsilon parameter
        if index>200:
            self.epsilon=0.999*self.epsilon
        
        # if this condition is satisfied, we are exploring, that is, we select random actions
        if randomNumber < self.epsilon:
            # returns a random action selected from: 0,1,...,actionNumber-1
            return np.random.choice(self.actionDimension)            
        
        # otherwise, we are selecting greedy actions
        else:
            # we return the index where Qvalues[state,:] has the max value
            # that is, since the index denotes an action, we select greedy actions
                       
            Qvalues=self.mainNetwork.predict(state.reshape(1,self.stateDimension))
          
            return np.random.choice(np.where(Qvalues[0,:]==np.max(Qvalues[0,:]))[0])
         
  
    '''
    when replay buffer is full or fullfilled with requirements, then 
    train network is implementing, to train neural network
    '''
    def trainNetwork(self):

        # if the replay buffer has at least batchReplayBufferSize elements,
        # then train the model 
        # otherwise wait until the size of the elements exceeds batchReplayBufferSize
        if (len(self.replayBuffer)>self.batchReplayBufferSize):
            

            # sample a batch from the replay buffer
            '''
            once random.sample returns list of items based on given 
            length . :
            heree: length is batchReplayBufferSize.
            buffer is replayBuffer
            
            '''
            randomSampleBatch=random.sample(self.replayBuffer, self.batchReplayBufferSize)
            
            # here we form current state batch 
            # and next state batch
            # they are used as inputs for prediction
            currentStateBatch=np.zeros(shape=(self.batchReplayBufferSize,self.stateDimension))
            
            nextStateBatch=np.zeros(shape=(self.batchReplayBufferSize,self.stateDimension))           
             
            # this will enumerate the tuple entries of the randomSampleBatch
            # index will loop through the number of tuples
            
            for index,tupleS in enumerate(randomSampleBatch):
                # first entry of the tuple is the current state
                '''
                because we already created tuple based buffer
                on which currentState,action,reward,nextState,terminalState
                are inserted!
                '''
                currentStateBatch[index,:]=tupleS[0]
                
                # fourth entry of the tuple is the next state
                nextStateBatch[index,:]=tupleS[3]
            
            # here, use the target network to predict Q-values 
            QnextStateTargetNetwork = self.targetNetwork.predict(nextStateBatch)
            # here, use the main network to predict Q-values 
            QcurrentStateMainNetwork = self.mainNetwork.predict(currentStateBatch)
            
            # now, we form batches for training
            # input for training
            inputNetwork=currentStateBatch
            
            # output for training
            outputNetwork=np.zeros(shape=(self.batchReplayBufferSize, self.actionDimension))
            
            # this list will contain the actions that are selected from the batch 
            # this list is used in my_loss_fn to define the loss-function
            self.actionsAppend=[]            
            for index,(currentState,action,reward,nextState,terminated) in enumerate(randomSampleBatch):
                
                # if the next state is the terminal state
                if terminated:
                    y=reward                  
                # if the next state if not the terminal state    
                else:
                    y=reward+self.gamma*np.max(QnextStateTargetNetwork[index])
                
                # this is necessary for defining the cost function
                self.actionsAppend.append(action)
                
                # this actually does not matter since we do not use all the entries in the cost function
                outputNetwork[index]=QcurrentStateMainNetwork[index]
                # this is what matters
                outputNetwork[index,action]=y
            
            # here, we train the network
            history = self.mainNetwork.fit(inputNetwork, outputNetwork, batch_size = self.batchReplayBufferSize, verbose=0, epochs=100)     
            
            print(f"Training Loss: {history.history['loss'][0]}")
          
            self.counterUpdateTargetNetwork+=1  
            if (self.counterUpdateTargetNetwork>(self.updateTargetNetworkPeriod-1)):
                # copy the weights to targetNetwork
                self.targetNetwork.set_weights(self.mainNetwork.get_weights())        
                print("Target network updated!")
                print("Counter value {}".format(self.counterUpdateTargetNetwork))
                # reset the counter
                self.counterUpdateTargetNetwork=0

                  
                
                
            
            
            
            
            
            
            
            
            
            
        
        
        
        
        
        
    