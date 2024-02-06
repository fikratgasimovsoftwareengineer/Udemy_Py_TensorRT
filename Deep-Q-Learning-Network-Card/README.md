### Deep-Q-Learning-Network-from-Scratch-in-Python-TensorFlow-and-OpenAI-Gym

These code files implement the Deep Q-learning Network (DQN) algorithm from scratch by using Python, TensorFlow (Keras), and OpenAI Gym. The codes are tested in the OpenAI Gym Cart Pole (v1) environment. These code files are a part of the reinforcement learning tutorial I am developing. The tutorial webpage explaining the codes is given here: 

#### Attention...

#### Please install requirements file within docker container which is initialized within udemy course!


```
Repository is described by Followign files :

- "train_card.py" - this is the driver code for training the model. This code import a class definition that implements the Deep Q Network from "functions_final.py". You should start from here.

- "deep_qlearning.py" - this is the file that implements the class called "DeepQLearning" that implements the Deep Q Network.

- "validat_model_on_simulation.py" - this file loads the trained TensorFlow model stored in the TensorFlow model file "trained_model.h5"  and creates a video that shows the control performance. Note that the video is saved in the folder "stored_video" (if such a folder does not exist it will be created)


- "trained_model_temp.h5" - this is just a temporary model obtained after several training episodes to illustrate the performance of untrained model, and to use it as a baseline

- In case , you can also download trained_model_temp.h5 from google drive :
(LINK for model )["https://drive.google.com/file/d/1KyS_CK-KIYdTNwrOyFrWP4yeblJWyWj3/view?usp=drive_link]
```
