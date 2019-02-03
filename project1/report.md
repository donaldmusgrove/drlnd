# Banana Navigation Report

## Introduction
This report document details the approach I used for the Banana Collection environment. All code is available in my [Jupyter Notebook](https://github.com/donaldmusgrove/drlnd/blob/master/project1/Navigation.ipynb). I begin this report by describing the environment, including the state space, available actions, and rewards. Next, I show the algorithm I used to solve the task, including the Q-learning framework, hyperparameters used, and training stopping criteria. Finally, I give a few examples of futher improvement of the model performance.


## Banana Collector Environment
From the course project notes, the state space of the Banana Collector environment has 37 dimensions, which includes the agent's velocity and a ray-based perception around the agent's forward direction. Alternatively, the state could be the actual 2D images, but that is a bonus problem for this project that I do not attempt.

Given the current state of the environment, there are 4 actions that an agent can take:
- `0` - move forward 
- `1` - move backward
- `2` - turn left
- `3` - turn right

Since the goal is to collect yellow bananas and avoid purple bananas, the agent receives a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a purple banana. 

The Banana Collection problem is presented as an episodic task. It's not clear from the project instructions, but an episode is apparently 300 time steps. In order to consider the problem solved, the agent must get an average score of +13 over 100 consecutive episodes. 


## Learning Algorithm
To solve the Banana Collection task, I implemented Q-learning via a deep Q network (DQN), with a target network and an experience replay buffer. I used a target network variant called [Double DQN learning](https://arxiv.org/pdf/1509.06461.pdf). Details of each of these methods, along with hyperparameters used, are given below.

### Deep Q Network
My DQN takes as input the 37-vector of states and outputs a 4-vector of expected rewards corresponding to each action. 

The DQN I used consists of a neural network with 3 dense layers. The first layer takes as input the states and has 64 nodes. The second layer takes as input the output of the first layer and has 64 nodes as well. The 3rd and final layer takes as input the output of the second layer and has 4 nodes. The first and second layers have ReLU activation functions while the final layer has a linear activation function. The DQN was implemented using [PyTorch](https://pytorch.org/).

### Target Network
Two networks were used during training, a local network and a target network. The local network is used to take actions and is updated after every time step. The target network is updated with the weights of the local network at a uniform interval. The model loss function is then the mean squared error (MSE) between the local and target network outputs. In my implementation, the target network is updated every 4 steps.

### Experience Replay
The observed state, action, reward, and next state based on the chosen action are a tuple referred to as an experience. During training, if the experiences are used to train the model in a sequential fashion, there is a good chance that correlation between consecutive experiences with be learned by the agent, thus reducing performance. Instead of learning sequentially, experiences are stored in a buffer, the replay buffer, and random batches of experiences are sampled during training. In my implementation, the replay buffer size is 100,000 and the batch size of randomly drawn experiences is 64.


### Double DQN Learning
The method of Double DQN learning is a special case of target networks. In a basic target network approach, the maximum Q-values of the target network are used. It has been seen that this approach can lead to overestimation of the Q-values, that is, the expected rewards for state/action combinations can be incorrectly large in magnitude. Instead, I implement the [Double DQN](https://arxiv.org/pdf/1509.06461.pdf) approach.

The double DQN computes the Q-value of the target network using two steps: 
1. From the target network, select the action value corresponding to the largest expected reward for the next state
2. Evaluate the target network at the next state using the selected action

This is a very simple modification that results in the Banana Collector solving the task in less than 700 episodes.

### Hyperparameters
Q-Network

* First layer input size: 37
* 3 hidden layers with nodes 64, 64, and 4
* Loss function: MSE
* Optimizer: Adam
* Learning rate: 0.0005
* Soft update parameter (tau): 0.001

Experience Replay Buffer

* Buffer size: 100,000
* Buffer sampling batch size: 64

Learning (model updating during training)

* Reward discount (gamma): 0.99
* Target network update steps: 4
* Epsilon-greedy parameters (start, end, decay): (1.0, 0.05, 0.995)
* Agent seed: 983275



## Plot of Rewards
The agent was set to train for up to 10,000 episodes. Early stopping criteria was set as maintaining an average score of >= 13 for at least 100 episodes. During the training as reflected in my [Jupyter Notebook](https://github.com/donaldmusgrove/drlnd/blob/master/project1/Navigation.ipynb), training met the early stopping criteria after 636 episodes. Thus, the agent was able to solve the problem in 536 episodes.

The following plot illustrates the evolution of the agent's score over the training episodes. We can see that by about episode 536, the central tendency of the score is above the desired value of 13.

<img src="https://github.com/donaldmusgrove/drlnd/blob/master/project1/episodes_vs_scores.PNG" width="300" >



## Ideas for Future Work
There are several methods for enhancing the agent's performance. Here are two methods that would likely lead to improvements:

1. Training using the pixels. Instead of relying on a 37-vector representation of the state space, we could instead use the actual 2D pixel images directly. The current state of the art would be then to replace the dense layers of my DQN with several and successive 2D [convolution layers](https://en.wikipedia.org/wiki/Convolutional_neural_network).

2. [Prioritized experience replay](https://arxiv.org/abs/1511.05952). Switching the uniform sampling of the replay buffer to some principled weighted sampling scheme could improve performance and efficiency, e.g., fewer training episodes. The sampling weight could be computed as a function of the training error; high error for certain experiences suggest that we're taking poor actions and thus ought to prioritize these experiences. When training error is low for other experiences, we're pretty good at making decisions and thus we can de-prioritize these experiences.


