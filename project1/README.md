# Banana Navigation Project for DRLND Udacity Nanodegree 
### Winter 2018/2019 course
---
## Project Details
This repo contains the Python code for training an agent to collect bananas in the [Unity Banana Collector](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#banana-collector) environment (note that my description below does not match this link, apparently things have been simplified for the course). The work here was completed as part of the first project from Udacity's [Deep Reinforcment Learning course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). In this project, I used [PyTorch](https://pytorch.org/) to train a deep neural network to carry out [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning).

This first project requires the students to develop and train a [Deep Q-Network (DQN)](https://deepmind.com/research/dqn/) model to collect yellow bananas, while avoiding purple bananas, in a simulated environment. I'm not great at creating animated graphics, so [here](https://www.youtube.com/watch?v=m7aOodyDlkk) is a Youtube video of an agent interacting with the environment.

The project report, including model details, hyperparameters, etc., is available [here](https://github.com/donaldmusgrove/drlnd/blob/master/project1/report.md).


### Banana Collector Environment Details
From the course project notes, the state space of the Banana Collector environment has 37 dimensions, which includes the agent's velocity and a ray-based perception around the agent's forward direction. I have no idea what *ray-based perception* means, but this is just someone's approach to representing the environment using 37 values. Alternatively, the state could be the actual 2D images, but that is a bonus problem for this project that I do not attempt.

Given the current state of the environment, there are 4 actions that an agent can take:
- `0` - move forward 
- `1` - move backward
- `2` - turn left
- `3` - turn right

Since the goal is to collect yellow bananas and avoid purple bananas, the agent receives a reward of +1 for collecting a yellow banana and a reward of -1 for collecting a purple banana. 

The project requires that in order to consider the problem solved, the agent must get an average score of +13 over 100 consecutive episodes.


## Getting Started
To run my code, you will need to set up the Unity environment. Here are steps to help you along.

### Step 1: Clone the Deep Reinforcement Learning Nanodegree (DRLND) Repository
First, follow the instructions [here](https://github.com/udacity/deep-reinforcement-learning#dependencies). The instructions have you set-up a new Python environment, install [OpenAI Gym](https://github.com/openai/gym), the [Classic Control](https://github.com/openai/gym#classic-control) environment, and box2d. The instructions for installing box2d are non-existent at the OpenAI repo, I suggest you build from source using the instructions [here](https://github.com/pybox2d/pybox2d/blob/master/INSTALL.md). 

Using the linked instructions, you will install PyTorch, the ML-Agents toolkit, and a few more Python packages required to use my Jupyter Notebook file.

### Step 2: Download the Unity Environment
You need the Udacity version of the environment. Download the Unity environment that matches your operating system:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

I completed this project using the Windows (64-bit) version, thus the contents of this repo reflect that operating system choice.


## Instructions
After you have followed the instructions above, open my [Navigation.ipynb](https://github.com/donaldmusgrove/drlnd/blob/master/project1/Navigation.ipynb) in Jupyter Notebook. I have included step-by-step instructions and some explanations with what is going on in each cell. Be sure that you select the drlnd kernel!

Section 4 of [Navigation.ipynb](https://github.com/donaldmusgrove/drlnd/blob/master/project1/Navigation.ipynb) includes the code for training the agent using the DQN (see [the report](https://github.com/donaldmusgrove/drlnd/blob/master/project1/report.md) for more info on the DQN used). Section 5 loads pre-saved weights for the agent's DQN and runs through a single episode.

