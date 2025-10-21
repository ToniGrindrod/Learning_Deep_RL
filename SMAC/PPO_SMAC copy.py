"""
Implements independent PPO agents (each have their own policy network and train separately)".
Creates an agent for each in the total number in the environments -e.g. for 3m, 3 agents are created; for 2s3z, 5 agents
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import neptune

from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym

#from smaclite import SMACliteEnv
#from smacv2.env import StarCraft2Env 
from smac.env import StarCraft2Env 

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the BackboneNetwork class with fully connected layers
class actorNetwork(nn.Module):
    def __init__(self, input_size, hidden_dimensions, out_features):
        super(actorNetwork, self).__init__()

        # Fully connected layers (MLP)
        self.fc1 = nn.Linear(input_size, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, out_features)

    def forward(self, state, avail_action):
        x = F.relu(self.fc1(state))  # First fully connected layer with ReLU activation
        x = self.fc2(x)

        # # Reshape if necessary to match logits shape
        # if avail_action.shape != x.shape:
        #     avail_action = avail_action.view(x.shape)

        # Apply masking: Set logits of unavailable actions to -inf
        x[avail_action == 0] = float('-inf')
        
        return x

class criticNetwork(nn.Module):
    def __init__(self, input_size, hidden_dimensions, out_features):
        super(criticNetwork, self).__init__()

        # Fully connected layers (MLP)
        self.fc1 = nn.Linear(input_size, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, out_features)

    def forward(self, state):
        x = F.relu(self.fc1(state))  # First fully connected layer with ReLU activation
        x = self.fc2(x)
       
        return x

class actorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    def forward(self, state, avail_action):
        action_pred = self.actor(state, avail_action)
        value_pred = self.critic(state)
        return action_pred, value_pred

class PPO_agent:
    def __init__(self,
                 env,
                 device,
                 run,
                 hidden_dimensions,
                 dropout, discount_factor,
                 reward_threshold,
                 print_interval,
                 PPO_steps,
                 n_trials,
                 epsilon,
                 entropy_coefficient,
                 learning_rate,
                 batch_size,
                 optimizer_name,
                 n_actions, n_obs):
        self.env = env  # Store the environment as an attribute

        self.device = device
        self.run = run

        self.INPUT_FEATURES = n_obs
        self.ACTOR_OUTPUT_FEATURES = n_actions

        self.HIDDEN_DIMENSIONS = hidden_dimensions

        self.CRITIC_OUTPUT_FEATURES = 1
        self.DROPOUT = dropout

        ###still need to remove possibly redundant attributes of PPO_agent: n_trials, reward_threshold
        self.discount_factor = discount_factor
        #self.max_episodes = max_episodes
        self.reward_threshold= reward_threshold
        self.print_interval = print_interval
        self.PPO_steps=PPO_steps
        self.n_trials=n_trials
        self.epsilon=epsilon
        self.entropy_coefficient=entropy_coefficient
        self.learning_rate=learning_rate
        self.batch_size=batch_size

        #print(f"hidden_dimensions: {self.HIDDEN_DIMENSIONS}, out_features: {self.ACTOR_OUTPUT_FEATURES}")
 
        # Initialize actor and critic networks
        self.actor = actorNetwork(
            self.INPUT_FEATURES, self.HIDDEN_DIMENSIONS, self.ACTOR_OUTPUT_FEATURES
        ).to(self.device)
        self.critic = criticNetwork(
            self.INPUT_FEATURES, self.HIDDEN_DIMENSIONS, self.CRITIC_OUTPUT_FEATURES
        ).to(self.device)

        #Better move the .to(self.device) call separately for both self.actor and self.critic. This ensures the individual parts of the model are moved to the correct device before combined into the actorCritic class
        # Combine into a single actor-critic model
        self.model = actorCritic(self.actor, self.critic)

        try:
            # Try to get the optimizer from torch.optim based on the provided name
            self.optimizer = getattr(optim, optimizer_name)(self.model.parameters(), lr=self.learning_rate)
        except AttributeError:
            # Raise an error if the optimizer_name is not valid
            raise ValueError(f"Optimizer '{optimizer_name}' is not available in torch.optim.")
    

    #The standard policy gradient loss is calculated as the product of the policy action probabilities and the advantage function
    #The standard policy gradietn loss cannot make corrections for abrupt policy changes. The surrogate loss modifies the standard loss to restrict the amount the policy can change in each iteration.
    #The surrogate loss is the minimum of (policy ratio X advantage function) and (clipped value of policy ratio X advantage function) where the policy ratio is between the action probabilities according to the old versus new policies and clipping restricts the value to a region near 1.

    def calculate_surrogate_loss(self, actions_log_probability_old, actions_log_probability_new, advantages):
        advantages = advantages.detach()
        # creates a new tensor that shares the same underlying data as the original tensor but breaks the computation graph. This means:
        # The new tensor is treated as a constant with no gradients.
        # Any operations involving this tensor do not affect the gradients of earlier computations in the graph.

        #If the advantages are not detached, the backpropagation of the loss computed using the surrogate_loss would affect both the actor and the critic networks
        # The surrogate loss is meant to update only the policy (actor).
        # Allowing gradients to flow back through the advantages would inadvertently update the critic, potentially disrupting its learning process.

        policy_ratio  = (actions_log_probability_new - actions_log_probability_old).exp()
        surrogate_loss_1 = policy_ratio*advantages
        surrogate_loss_2 = torch.clamp(policy_ratio, min =1.0-self.epsilon, max = 1.0+self.epsilon)*advantages
        surrogate_loss=torch.min(surrogate_loss_1, surrogate_loss_2)
        return surrogate_loss

    #TRAINING THE AGENT
    #Policy loss is the sum of the surrogate loss and the entropy bonus. It is used to update the actor (policy network)
    #Value loss is based on the difference between the value predicted by the critic and the returns (cumulative reward) generated by the policy. This loss is used to update the critic (value network) to make predictions more accurate.

    def calculate_losses(self, surrogate_loss, entropy, returns, value_pred):
        entropy_bonus = self.entropy_coefficient*entropy
        policy_loss = -(surrogate_loss+entropy_bonus).sum()
        value_loss = torch.nn.functional.smooth_l1_loss(returns, value_pred).sum() #helps to smoothen the loss function and makes it less sensitive to outliers.
        return policy_loss, value_loss

    def init_training(self):
        #create a set of buffers as empty arrays. To be used during training to store information
        states = []
        avail_actions = []
        actions = []
        actions_log_probability = []
        values = []
        return states, avail_actions, actions, actions_log_probability, values
    
    def update_policy(self,
            states,
            avail_actions,
            actions,
            actions_log_probability_old,
            advantages,
            returns):
        #print(f"Returns NaNs: {torch.isnan(returns).any()}")
        total_policy_loss = 0
        total_value_loss = 0
        actions_log_probability_old = actions_log_probability_old.detach()
        actions=actions.detach()

        # print(f"Returns NaNs: {torch.isnan(returns).any()}")
        # print(f"advantages NaNs (after calculation): {torch.isnan(advantages).any()}")


        #detach() is used to remove the tensor from the computation graph, meaning no gradients will be calculated for that tensor when performing backpropagation.
        #In this context, it's used to ensure that the old actions and log probabilities do not participate in the gradient computation during the optimization of the policy, as we want to update the model based on the current policy rather than the old one.
        #print(type(states), type(actions),type(actions_log_probability_old), type(advantages), type(returns))
        training_results_dataset= TensorDataset(
                states,
                avail_actions,
                actions,
                actions_log_probability_old,
                advantages,
                returns) #TensorDataset class expects all the arguments passed to it to be tensors (or other compatible types like NumPy arrays, which will be automatically converted to tensor
        batch_dataset = DataLoader(
                training_results_dataset,
                batch_size=self.batch_size,
                shuffle=False)
        #creates a DataLoader instance in PyTorch, which is used to load the training_results_dataset in batches during training.
        #batch_size defines how many samples will be included in each batch. The dataset will be divided into batches of size BATCH_SIZE. The model will then process one batch at a time, rather than all of the data at once,
        #shuffle argument controls whether or not the data will be shuffled before being split into batches.
        #Because shuffle is false, dataloader will provide the batches in the order the data appears in training_results_dataset. In this case, the batches will be formed from consecutive entries in the dataset, and the observations will appear in the same sequence as they are stored in the dataset.
        for _ in range(self.PPO_steps):
            for batch_idx, (states,avail_actions,actions,actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
                #get new log prob of actions for all input states
                action_pred, value_pred = self.model(states,avail_actions)
                #print("action_pred shape:",action_pred.shape)
                value_pred = value_pred.squeeze(-1)
                #Value_pred is typically the value estimate for each state, which is often predicted by the critic network. Depending on the network architecture, the output might have an extra dimension due to how the network is structured. For instance if your network's final layer produces a shape like (batch_size, 1), which is often the convention when processing batches with NNs
                #print("Logits before softmax:", action_pred)
                action_prob = torch.softmax(action_pred, dim=-1)
                # print("Action probabilities shape:", action_prob.shape)
                # print("Action probabilities sum:", action_prob.sum(dim=-1))
                if torch.isnan(action_prob).any():
                    # Handle the NaN values (e.g., replace them with zeros or a small number)
                    # This brute forces any actions that were not allowed to have zero probability 
                    action_prob = torch.nan_to_num(action_prob, nan=0.0)
                probability_distribution_new=torch.distributions.Categorical(action_prob)
                entropy = probability_distribution_new.entropy()
                #print("entropy: ",entropy)
                #.entropy() is a method available for torch.distributions.Categorical objects
                #The Categorical distribution in PyTorch is used to model discrete probability distributions. When you define a Categorical object, you're providing it with a vector of probabilities (in this case, action_prob), which represent the likelihood of each possible action.
                #Entropy is a measure of uncertainty. For a given distribution, it quantifies the "spread" or unpredictability of the possible outcomes. In reinforcement learning, we often want to maximize the entropy to encourage exploration (i.e., the agent should not become too deterministic too quickly).
                #Higher the entropy, the more uncertain and therefore the more uniform the choices of action/ the more exploration
                #The formula for entropy of a probability distribution $p = (p_1,p_2,...,p_n)$ over $n$ possible outcomes (where $p_i$ is the prbability of the i-th outcome) is given by
                # H(p) = -\sum_{i=1}^n p_i \log(p_i)
                #p_i*log(p_i) is negative (since log(p_i) is negative for probabilities less than 1).
                #when probability p_i is large (close to 1), log(p_i) is close to zero, but still negative, and so p_i*log(p_i) will be a small negative number. This reduces the total entropy because the negative contribution is smaller.
                # when probability p_i is small (close to 0), log(p_i) becomes larger in magnitude (more negative), making p_i*log(p_i) a larger negative number. This increases the total entropy

                #estimate new log probabilities using old actions
                actions_log_probability_new = probability_distribution_new.log_prob(actions)
                #print(actions_log_probability_new - actions_log_probability_old)
                # # Check for NaN or Inf in log probabilities
                # if torch.isnan(actions_log_probability_old).any() or torch.isinf(actions_log_probability_old).any():
                #     print("NaN or Inf detected in actions_log_probability_old!")
                #     return  # You can return or handle this case as needed

                # if torch.isnan(actions_log_probability_new).any() or torch.isinf(actions_log_probability_new).any():
                #     print("NaN or Inf detected in actions_log_probability_new!")
                #     return  # You can return or handle this case as needed

                # print(f"actions_log_probability_old NaNs: {torch.isnan(actions_log_probability_old).any()}")
                # print(f"actions_log_probability_new NaNs: {torch.isnan(actions_log_probability_new).any()}")
                # print(f"advantages NaNs: {torch.isnan(advantages).any()}")

                surrogate_loss = self.calculate_surrogate_loss(
                    actions_log_probability_old,
                    actions_log_probability_new,
                    advantages
                )

                # print(f"Surrogate Loss NaNs: {torch.isnan(surrogate_loss).any()}")
                # print(f"Entropy NaNs: {torch.isnan(entropy).any()}")
                # print(f"Returns NaNs: {torch.isnan(returns).any()}")
                # print(f"Value Predictions NaNs: {torch.isnan(value_pred).any()}")

                policy_loss, value_loss = self.calculate_losses(
                    surrogate_loss,
                    entropy,
                    returns,
                    value_pred
                )
                self.optimizer.zero_grad() #clear existing gradietns in the optimizer (so that these don't propagate accross multiple .backward(). Ensures each optimization step uses only the gradients computed during the current batch.
                if torch.isnan(policy_loss).any():
                    print("NaN detected in policy_loss before backward pass!")
                policy_loss.backward() #computes gradients for policy_loss with respect to the agent's parameters
                # #Check for NaN gradients after policy_loss backward
                # for param in self.model.parameters():
                #     if param.grad is not None:  # Check if gradients exist for this parameter
                #         if torch.isnan(param.grad).any():
                #             print("NaN gradient detected in policy_loss!")
                # #             return
                value_loss.backward()
                # Check for NaN gradients after value_loss backwardor param in self.model.parameters():
                # for param in self.model.parameters():
                #     if param.grad is not None:  # Check if gradients exist for this parameter
                #         if torch.isnan(param.grad).any():
                #             print("NaN gradient detected in value_loss!")
                #             return
                
                self.optimizer.step()
                #The update step is based on the learning rate and other hyperparameters of the optimizer
                # The parameters of the agent are adjusted to reduce the policy and value losses.
                total_policy_loss += policy_loss.item() #accumulate the scalar value of the policy loss for logging/ analysis
                #policy_loss.item() extracts the numerical value of the loss tensor (detaching it from the computational graph).
                #This value is added to total_policy_loss to compute the cumulative loss over all batches in the current PPO step.
                #Result: tracks the total policy loss for the current training epoch
                # The loss over the whole dataset is the sum of the losses over all batches.
                #The training dataset is split into batches during the training process. Each batch represents a subset of the collected training data from one episode.
                # Loss calculation is performed for each batch (policy loss and value loss)
                # for each batch, gradients are calculated with respect to the total loss for that batch and the optimizer then updates the network parameters using these gradients.
                # this is because the surrogate loss is only calculated over a single batch of data
                #look at the formula for surrogate loss.
                # It is written in terms of an expectation ˆ Et[. . .] that indicates the empirical average over a finite batch of samples.
                # This means you have collected a set of data (time steps) from the environment, and you're averaging over these data points. The hat symbol implies you're approximating the true expectation with a finite sample of data from the environment. This empirical average can be computed as the mean of values from the sampled transitions
                # the expectation is taken over all the data you've collected
                #If you're training with multiple batches (i.e., collecting data in chunks), then you can think of the expectation as being computed over each batch.
                #The overall expectation can indeed be seen as the sum of expectations computed for each batch, but The expectation of the sum is generally not exactly equal to the sum of the expectations unless the samples are independent, but in practical reinforcement learning algorithms, it's typically a good enough approximation
                #For samples to be independent, the outcome of one sample must not provide any information about the outcome of another. Specifically, in the context of reinforcement learning, this means that the states, actions, rewards, and subsequent states observed in different time steps or different episodes should be independent of each other.
                total_value_loss += value_loss.item()
                #Notice that we are calculating an empirical average, which is already an approximation on the true value (the true expectation would be the average over an infinite amount of data, and the empirical average is the average over the finite amount of data that we have collected).
                #But furthermore, we are approximating even the empirical average istelf. The empirical average is the average over all our collected datal, but here we actually batch our data, calculate average over each batch and then sum these averages, which is not exaclty equal to the average of the sums (but is a decent approximation).
        return total_policy_loss / self.PPO_steps, total_value_loss / self.PPO_steps
    
class SMAC_agents: #could rename to SMAC execution
    def __init__(self,
                 env,
                 device,
                 run,
                 hidden_dimensions,
                 dropout, discount_factor,
                 max_episodes,
                 reward_threshold,
                 print_interval,
                 PPO_steps,
                 n_trials,
                 epsilon,
                 entropy_coefficient,
                 learning_rate,
                 batch_size,
                 optimizer_name):
        self.env = env  # Store the environment as an attribute

        self.device = device
        self.run = run

        self.n_trials=n_trials
        self.print_interval = print_interval

        self.reward_threshold= reward_threshold
        self.print_interval = print_interval
        self.PPO_steps=PPO_steps
        self.n_trials=n_trials
        self.epsilon=epsilon
        self.entropy_coefficient=entropy_coefficient
        self.learning_rate=learning_rate
        self.batch_size=batch_size

        self.num_agents = self.env.n_agents
        self.env.reset()
        agents_actions= self.env.get_avail_actions()
        num_agents_actions= [len(agents_actions[t]) for t in range(self.num_agents)]
        #print(num_agents_actions)
        agents_obs=self.env.get_obs()
        num_agents_obs = [len(agents_obs[t]) for t in range(self.num_agents)]
        #print(f"num agents obs: {num_agents_obs}")
        self.PPO_agents = [PPO_agent(
                 self.env,
                 self.device,
                 self.run,
                 hidden_dimensions,
                 dropout, discount_factor,
                 reward_threshold,
                 print_interval,
                 PPO_steps,
                 n_trials,
                 epsilon,
                 entropy_coefficient,
                 learning_rate,
                 batch_size,
                 optimizer_name,
                 num_agents_actions[j],
                 num_agents_obs[j]) for j in range(self.num_agents)]
        self.max_episodes=max_episodes
        self.discount_factor = discount_factor
        #will probably need to go through these parameters and decide which should be in PPO_agent function and which should be in SMAC_agents
        
    def calculate_returns(self, rewards):
        returns = []
        cumulative_reward = 0
        for r in reversed(rewards):
            cumulative_reward = r +cumulative_reward*self.discount_factor
            returns.insert(0, cumulative_reward)
        returns = torch.tensor(returns).to(self.device)
        #normalize the return
        epsilon = 1e-8  # Small constant to avoid division by zero
        returns = (returns - returns.mean()) / (returns.std() + epsilon)
        #returns = (returns-returns.mean())/returns.std()
        # print(f"*return norm NaNs: {torch.isnan(1/ (returns.std() + epsilon)).any()}")
        # print(f"*return NaNs: {torch.isnan(returns).any()}")
        #I had conceptual trouble with normalizing the reward by an average, because it seemed to me since we're adding more rewards for earlier timesteps, the cumulative reward for earlier times would be a lot larger. But need to consider dicount facotr.
        # Future rewards contribute significantly to the cumulative return, so earlier timesteps will likely have larger returns.
        #if gamma is close to 0, future rewards have little influence, and the return at each timestep will closely resemble the immediate reward, meaning the pattern might not be as clear.
        return returns
    
    #The advantage is calculated as the difference between the value predicted by the critic and the expected return from the actions chosen by the actor according to the policy.
    def calculate_advantages(self, returns, values):
        advantages = returns - values
        # Normalize the advantage
        advantages = (advantages - advantages.mean()) / advantages.std()
        return advantages
    
    def init_training_all(self):
        #create a set of buffers as empty arrays. To be used during training to store information
        states = [[] for _ in range(self.num_agents)]
        avail_actions = [[] for _ in range(self.num_agents)]
        actions = [[] for _ in range(self.num_agents)]
        actions_log_probability = [[] for _ in range(self.num_agents)]
        values = [[] for _ in range(self.num_agents)]
        rewards = []
        done = False
        episode_reward = 0
        return states, avail_actions, actions, actions_log_probability, values, rewards, done, episode_reward
    
    def forward_pass_all(self):
        states, avail_actions, actions, actions_log_probability, values, rewards, done, episode_reward = self.init_training_all()
        self.env.reset()
        state = self.env.get_obs()
        avail_action = self.env.get_avail_actions()
        for i in range(self.num_agents):
            states[i],avail_actions[i], actions[i], actions_log_probability[i], values[i] = self.PPO_agents[i].init_training()
            self.PPO_agents[i].model.eval()# change to eval mode
        while True:
            for i in range(self.num_agents):
                state[i] = torch.FloatTensor(state[i]).unsqueeze(0).to(self.device)#converts state (a numpy array) into a pytorch tensor of type FloatTensor. NN in pyTorch expect tensor inputs.
                #print(state[i])
                ###Note that in PyTorch, when you perform operations on tensors, the output tensor will automatically be on the same device as the input tensors, provided that all the tensors involved in the operation are on the same device.
                #unsqueeze(0) adds a new dimension at poisition zero (the batch dimension) as expected by NN in pytorch
                states[i].append(state[i])
                #print("state: ",np.shape(state[i]))
                avail_action[i] = torch.FloatTensor(avail_action[i]).unsqueeze(0).to(self.device)
                #print(avail_action[i])
                #print("avail_actions: ",np.shape(avail_action[i]))
                avail_actions[i].append(avail_action[i])
                action_pred, value_pred = self.PPO_agents[i].model(state[i],avail_action[i]) # automatically on correct device, because the model and the input tensor (state) will already be on the same device, and the forward pass in the model will operate on that device.
                action_prob = torch.softmax(action_pred,dim=-1)
                if torch.isnan(action_prob).any():
                    # Handle the NaN values (e.g., replace them with zeros or a small number)
                    action_prob = torch.nan_to_num(action_prob, nan=0.0)
                dist = torch.distributions.Categorical(action_prob)
                action=dist.sample()#note action is a PyTorch tensor
                log_prob_action = dist.log_prob(action)
                actions[i].append(action)
                actions_log_probability[i].append(log_prob_action)
                values[i].append(value_pred)

            action_items=[]#only necessary if you need to do .item() before passing the actions to the step function
            for j in range(self.num_agents):
                action_items.append(actions[j][-1].item())
            reward, done, _ = self.env.step(action_items) #the third object returned is info, don't need this extra info
            next_state = self.env.get_obs()
            next_avail_action=self.env.get_avail_actions()
            rewards.append(reward)
            episode_reward+=reward
            state = next_state # Move to the next state
            avail_action=next_avail_action

            if done: # if the episode is done or truncated
                break
        
        returns = self.calculate_returns(rewards)
        advantages =[]

        for i in range(self.num_agents):
            #print("avail_actions: ", avail_actions)
            #print("states: ", states)
            #print("states: ",type(states[i]))
            states[i]=torch.cat(states[i]).to(self.device)
            #converts the list of individual states into a sinlem tensor that is necessary for later processing
            #Creates a single tensor with dimensions like (N, state_dim), where: N is the number of states collected in the episode; state_dim is the dimensionality of each state.
            #torch.cat() expects a sequence (e.g. list or tuple) of PyTorch tensors as input.
            #print(avail_actions[i])
            avail_actions[i]=torch.cat(avail_actions[i]).to(self.device)
            #print("avail_actions: ",type(avail_actions[i]))
            actions[i]=torch.cat(actions[i]).to(self.device)
            #Note that, in the loop, both state and action are PyTorch tensors so that states and actions are both lists of PyTorch tensors
            actions_log_probability[i]=torch.cat(actions_log_probability[i]).to(self.device)
            values[i]=torch.cat(values[i]).squeeze(-1).to(self.device)# .squeeze removes a dimension of size 1 only from tensor at the specified position, in this case, -1, the last dimesion in the tensor. Note that .squeeze() does not do anything if the size of the dimension at the specified potision is not 1.
            # print(f"rewards NaNs: {torch.isnan(torch.tensor(rewards, dtype=torch.float32)).any()}")
            # print(f"values NaNs: {torch.isnan(torch.tensor(values, dtype=torch.float32)).any()}")
            advantages.append(self.calculate_advantages(returns, values[i]))
        return episode_reward, states, avail_actions, actions, actions_log_probability, advantages, returns
    
    def train_with_test(self):
        train_rewards = []
        policy_losses = [[] for _ in range(self.num_agents)]
        value_losses = [[] for _ in range(self.num_agents)]
        timestep_count = 0
        #lens = []

        #print(self.max_episodes+1)

        for episode in range(1, self.max_episodes + 1):
            #print("start of episode: ", episode)
            # Perform a forward pass and collect experience
            train_reward, states, avail_actions, actions, actions_log_probability, advantages, returns = self.forward_pass_all()
            # Update the policy using the experience collected
        
            for i in range(self.num_agents):
                policy_loss, value_loss = self.PPO_agents[i].update_policy(
                    states[i],
                    avail_actions[i],
                    actions[i],
                    actions_log_probability[i],
                    advantages[i],
                    returns)
            # test_reward = self.evaluate()

                # # Visualize the environment if it supports rendering (currently this is done once each episode - might want to change to once every multiple of episodes)
                # if hasattr(self.env, "render") and callable(getattr(self.env, "render", None)):
                #   self.env.render()

                # Log the results
                policy_losses[i].append(policy_loss)
                value_losses[i].append(value_loss)
            
                #print(train_reward)
                #lens.append(episode_len)
                # test_rewards.append(test_reward)
                self.run["policy_loss"+str(i)].log(policy_loss)
                self.run["value_loss"+str(i)].log(value_loss)
                
                # self.run["test_reward"].log(test_reward)

                
                # #mean_test_rewards = np.mean(test_rewards[-self.n_trials:])
                # mean_abs_policy_loss = np.mean(np.abs(policy_losses[i][-self.n_trials:]))
                # mean_abs_value_loss = np.mean(np.abs(value_losses[i][-self.n_trials:]))
            train_rewards.append(train_reward)
            print("episode",episode,": ", train_reward)
            self.run["train_reward"].log(train_reward)
            # Calculate the mean of recent rewards and losses for display
            mean_train_rewards = np.mean(train_rewards[-self.n_trials:])
            # Print results at specified intervals
            if episode % self.print_interval == 0:
                print(f'Episode: {episode:3} | \
                    Mean Train Rewards: {mean_train_rewards:3.1f} ')



                # # Check if reward threshold is reached
                # if mean_test_rewards >= self.reward_threshold:
                #     print(f'Reached reward threshold in {episode} episodes')
                #     break
            # Check if the environment has a close method before calling it
            # if hasattr(self.env, "close") and callable(getattr(self.env, "close", None)):
            #   self.env.close() #Close environment visualisation after training is done.
        return policy_losses, value_losses, train_rewards

#train_rewards measures the cumulative reward obtained in the training environment while the agent is learning
#test_rewards measures the cumulative reward obtained in the test environment with no learning updates- only evaluating the policy as it currently stands
#If you call agent.train_with_test(), test_rewards evaluates the performace of the policy in a test environment (env_test) after updates made during the training phase. It provides insight into how well the policy generalizes to unseen or separate scenarios.
#The inclusion of test_rewards in the same loop as train_rewards allows you to monitor how the policy generalizes immediately after each training update which provides a real-time understanding of the trade-off between overfitting to the training environment and achieving generalizable performnace.
#Keeping them in the same loop ensures both metrics correspond to the same stage of training. This helps in diagnozing potential issues, such as overfitting.
    

def plot_train_rewards(train_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(train_rewards, label='Training Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Training Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(train_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_test_rewards(test_rewards, reward_threshold):
    plt.figure(figsize=(12, 8))
    plt.plot(test_rewards, label='Testing Reward')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Testing Reward', fontsize=20)
    plt.hlines(reward_threshold, 0, len(test_rewards), color='y')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_losses(policy_losses, value_losses):
    plt.figure(figsize=(12, 8))
    plt.plot(value_losses, label='Value Losses')
    plt.plot(policy_losses, label='Policy Losses')
    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Loss', fontsize=20)
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def mainSMAC():
    my_api="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDg3ZjNlYi04MWI3LTQ1ODctOGIxNS1iNTY3ZjgzMGYzMzYifQ=="
    run = neptune.init_run(
    project="EnergyGridRL/SMAC-3m",
    api_token=my_api
    )  # your credentials
    # Create the environment
    params = {
        "optimizer_name": "Adam",
        "MAX_EPISODES":10000,
        "DISCOUNT_FACTOR":0.99,#0.8 to 0.9997. Common 0.99
        "REWARD_THRESHOLD": 20,
        "PRINT_INTERVAL": 10,
        "PPO_STEPS":3,
        "N_TRIALS": 100,
        "EPSILON": 0.1,
        "ENTROPY_COEFFICIENT":0.001,#0 to 0.01
        "HIDDEN_DIMENSIONS": 64,
        "DROPOUT": 0.2,
        "LEARNING_RATE":5e-4,#0.003 to 5e-6
        "BATCH_SIZE":32
    }

    run["parameters"] = params
    env = StarCraft2Env(map_name="3m")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agents =SMAC_agents(env= env,
                 run = run,
                 device= device,
                 hidden_dimensions = params["HIDDEN_DIMENSIONS"],
                 dropout= params["DROPOUT"],
                 discount_factor = params["DISCOUNT_FACTOR"],
                 optimizer_name=params["optimizer_name"],
                 max_episodes=params["MAX_EPISODES"],
                 reward_threshold= params["REWARD_THRESHOLD"],
                 print_interval =params["PRINT_INTERVAL"],
                 PPO_steps =params["PPO_STEPS"],
                 n_trials= params["N_TRIALS"],
                 epsilon =params["EPSILON"],
                 entropy_coefficient=params["ENTROPY_COEFFICIENT"],
                 learning_rate=params["LEARNING_RATE"],
                 batch_size=params["BATCH_SIZE"])



    policy_losses, value_losses, train_rewards = agents.train_with_test()
    # Plotting the rewards and losses
    run.stop()
    #print(train_rewards)

    train_rewards = np.array(train_rewards)

    plot_train_rewards(train_rewards, agents.reward_threshold)

#next: figure out which parameters aren't parameters of each agent but rather of the SMACagents class

mainSMAC()