# Gridworld/Metalearning Exercises
When you run `train.py`, it initalizes a very simple grid world environment consisting of a single walled room containing a goal. The class `MLPAgent` is very similar to the agent you worked with this morning. It's `learn ` method implements A2C. This agent is trained to find the goal while avoiding the walls and in as few steps as possible. If the agent chooses an action that would take it into a wall, it will stay in the current location (and incur the cost of hitting a wall). The episode ends when the agent reaches the goal or when the maximum number of steps has been reached (default:8).

You can always use the flag `--render` to see what the environment looks like. 

## Exercise 1: Getting to know gym-simplegrid
At the bottom of train.py you'll see the definitions of `my_desc` which is a list of strings that species the layout of the environment using letters to indicate elements like walls (W), starting locations (S), and goal location (G). I've made two default layouts for two different tasks 'find goal' and 'collect'. You'll also see a `reward_map`, which maps each element in the environment to it's reward value. Let's stick with the default task ('find goal') task for now.

### Exercise 1.1 
Edit the layout of the environment by adding a pool of lava (using the letter 'L') to the environment and see if your MLPAgent can learn to find the goal while avoiding the pool of lava. You'll have to edit both `my_desc` and `reward_map`. How else can you make the task harder (or easier) to learn by editing these specifications? 

### Exercise 1.2
The second task ('collect') involves picking up purple balls. Each ball gives a reward of 5. The task is to collect all the purple balls in as few steps as possible while not bumping into any walls. An episode ends when all balls have been collected or when the maximum number of steps has been taken. Here I recommend a larger number of steps, which can be controlled with the `--n-steps` flag. Try running the following and you'll see that our MLPAgent cannot handle this task presently.
```bash
python train.py --task=collect --n_steps=16 
```
Why does our agent struggle so? 

There are at least two ways to modify our setup to handle this task. The first is to change the state reprsentation. By default, the input to our RL agent is a two-hot vector indicating the row and column of the agent's current position within the grid. Without any memory, this is a difficult state representation from which to perform the task. I've implemented a few other state representations. Try using an RGB rendering of the current state of the environment.
```bash
python train.py --task=collect --n_steps=16 --state_rep=rgb --n_episodes=3000
```
This should learn a bit better.

## Exercise 2: Adding LSTM agent
The second modification is to endow our agent with memory by using a recurrent instead of a feedforward architecture. Create a new class `RNNAgent` which uses LSTM versions of the Policy and Critic networks. These `PolicyRNN` and `CriticRNN` networks are already provided for you in networks.py. You should be able to more or less copy this RNNAgent from your work in this morning's session. 

Add a command line argument to specify which architecture to use so that you can run something like this to train our RNNAgent on the row-column state representation. With memory, the RNNagent should be able to remember which balls it has already collected and which are yet to be picked up and so should perform much better than the MLPAgent.
```bash
python train.py --task=collect --n_steps=16 --n_episodes=3000 --arch=lstm
```

## Exercise 3: Metalearning
### Exercise 3.1
Modeled after the class `train_A2C`, write a class `meta_train_A2C` which implements a meta-learning version of the find the goal task according to the following specifications. The solution you will be guided towards is in the style of the paper RL2: Fast Reinforcement Learning via Slow Reinforcement Learning by Yan Duan, John Schulman, Xi Chen, Peter L. Bartlett, Ilya Sutskever, Pieter Abbeel. You will need your newly written `RNNAgent` for this.

I've provided you with a function which generates a distribution of find the goal MDPs. These environments share a common structure. They consist of a single room with a fixed starting location and a single goal location, but the goal location varies from environment to environment. We are going to learn to learn to find the goal. 

1. Sample a new MDP every 5 episodes
2. Instead of restting the agent's hidden state after every episode, now you'll reset the hidden state after each 'block' of 5 episodes of interaction with a common. (`agent.reset()`)
3. Append to your state representation a value indicating whether this step is the first step of a new episode.
4. Add a command line argument to contol whether you use your new `meta_train_A2C` or the old `train_A2C`

Now if you run 
```bash
python train.py --n_episodes=3000 --meta
```
You should find that the agent eventually learns to find the goal quickly in any of the provided MDPs.

### Exercise 3.2
Reserve one (or more) environments to test generalization. If the model has meta-learned well, its recurrent dynamics over the course of 5 episodes of interaction with this held out environment should be able to discover the goal location (without any weight updates).

<!-- ### Exercise 3.3
Plot  -->

# 