# %%
import numpy as np
#from GridWorldDynamicTarget import GridWorldDynTargetEnv
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import neptune

from torch.utils.data import TensorDataset, DataLoader

import gymnasium as gym

from minatar import Environment

# %%
#PPO is implemented as an actor-critic model.
#Actor imlements the policy
#Critic predicts its estimated value.
#Both actor and critic NNs take the state at each timestep as input, so can share a common NN, the "backbone architecture"
#The actor and critic can extend the backbone architecture with additional layers.

import torch
import torch.nn as nn
import torch.nn.functional as F

class BackboneNetwork(nn.Module):
    def __init__(self, input_shape, hidden_dimensions, out_features, dropout):
        super(BackboneNetwork, self).__init__()

        # Unpack input shape (height, width, channels)
        h, w, c = input_shape  # h: height (10), w: width (10), c: number of channels

        # Convolutional layers
        self.conv1 = nn.Conv2d(c, 16, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1)

        # Compute the size of the feature maps after convolutions
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32  # 32 channels from last conv layer

        # Fully connected layers
        self.fc1 = nn.Linear(linear_input_size, hidden_dimensions)
        self.fc2 = nn.Linear(hidden_dimensions, out_features)

        #self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # Reorder dimensions to match PyTorch's Conv2d format: (batch_size, c, h, w)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)  # Flatten for fully connected layers
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = self.fc2(x)
        return x

#Define the actor-critic network
class actorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred
        #Returns both the action predictions and the value predictions.

# %%
#We’ll use the networks defined above to create an actor and a critic. Then, we will create an agent, including the actor and the critic.
#finish this step later
# def create_agent(hidden_dimensions, dropout):
#     INPUT_FEATURES =env_train.
class PPO_agent:
    def __init__(self,
                 env,
                 env_test,
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
        self.env_test = env_test

        self.device = device
        self.run = run

        n_actions = self.env.num_actions()
        # Reset and get the number of state observations
        ###THIS CHANGED FOR MINATAR
        self.env.reset()
        state=self.env.state()

        self.INPUT_FEATURES = state.shape
        self.ACTOR_OUTPUT_FEATURES = n_actions

        self.HIDDEN_DIMENSIONS = hidden_dimensions

        self.CRITIC_OUTPUT_FEATURES = 1
        self.DROPOUT = dropout

        self.discount_factor = discount_factor
        self.max_episodes = max_episodes
        self.reward_threshold= reward_threshold
        self.print_interval = print_interval
        self.PPO_steps=PPO_steps
        self.n_trials=n_trials
        self.epsilon=epsilon
        self.entropy_coefficient=entropy_coefficient
        self.learning_rate=learning_rate

        self.batch_size=batch_size

        # Initialize actor and critic networks
        self.actor = BackboneNetwork(
            self.INPUT_FEATURES, self.HIDDEN_DIMENSIONS, self.ACTOR_OUTPUT_FEATURES, self.DROPOUT
        ).to(self.device)
        self.critic = BackboneNetwork(
            self.INPUT_FEATURES, self.HIDDEN_DIMENSIONS, self.CRITIC_OUTPUT_FEATURES, self.DROPOUT
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
        actions = []
        actions_log_probability = []
        values = []
        rewards = []
        done = False
        episode_reward = 0
        return states, actions, actions_log_probability, values, rewards, done, episode_reward

    def forward_pass(self):#this is just the training function (might just want to rename it)
        states, actions, actions_log_probability, values, rewards, done, episode_reward = self.init_training()
        self.env.reset()
        # state() returns a NumPy array of shape (10, 10, c)
        state=self.env.state()
        self.model.train() # Recall model = ActorCritic(actor, critic). model.train() is a predefined method inherited from torch.nn.Module. It is used to set the module into training mode.
        while True:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)#converts state (a numpy array) into a pytorch tensor of type FloatTensor. NN in pyTorch expect tensor inputs.
            ###Note that in PyTorch, when you perform operations on tensors, the output tensor will automatically be on the same device as the input tensors, provided that all the tensors involved in the operation are on the same device.
            #unsqueeze(0) adds a new dimension at poisition zero (the batch dimension) as expected by NN in pytorch
            states.append(state)
            action_pred, value_pred = self.model(state) # automatically on correct device, because the model and the input tensor (state) will already be on the same device, and the forward pass in the model will operate on that device.
            action_prob = torch.softmax(action_pred,dim=-1)
            #print("Action probabilities shape:", action_prob.shape)
            #print("Action probabilities sum:", action_prob.sum(dim=-1))
            #applies the softmax function to the output of the actor network (action_pred) to compute a probability distribution over the possible actions
            #action_pred tensor contains the raw output of the actor network for the given state. These raw outputs, also called logits, are not constrained to lie in any specific range or sum to 1.
            #The dim=-1 argument specifies the dimension along which to apply the softmax function.
            #In this case, action_pred is a 1D tensor (a single batch of action logits), so dim=-1 ensures softmax is applied along the last dimension.
            dist = torch.distributions.Categorical(action_prob)
            action=dist.sample()#note action is a PyTorch tensor
            log_prob_action = dist.log_prob(action) #Looks up the probability of the selected action in the distribution dist. Takes the natural logarithm of that probability.
            #calculates the logarithm of the probability of a specific action
            reward, done= self.env.act(action.item())
            next_state = self.env.state()
            #dist.sample() from torch.distributions returns a PyTorch tensor, even if the sampled value is a scalar
            #Hence, must use .item() to access just the value of the action
            actions.append(action)
            actions_log_probability.append(log_prob_action)
            values.append(value_pred)

            # if episode_len > 100:
            #   reward -= 100
            #   done = True

            rewards.append(reward)
            episode_reward+=reward
            state = next_state # Move to the next state

            if done: # if the episode is done or truncated
                break

            #episode_len += 1
        
        states=torch.cat(states).to(self.device)#converts the list of individual states into a sinlem tensor that is necessary for later processing
        #Creates a single tensor with dimensions like (N, state_dim), where: N is the number of states collected in the episode; state_dim is the dimensionality of each state.
        #torch.cat() expects a sequence (e.g. list or tuple) of PyTorch tensors as input.
        actions=torch.cat(actions).to(self.device)
        #Note that, in the loop, both state and action are PyTorch tensors so that states and actions are both lists of PyTorch tensors
        actions_log_probability=torch.cat(actions_log_probability).to(self.device)
        values=torch.cat(values).squeeze(-1).to(self.device)# .squeeze removes a dimension of size 1 only from tensor at the specified position, in this case, -1, the last dimesion in the tensor. Note that .squeeze() does not do anything if the size of the dimension at the specified potision is not 1.
        # print(f"rewards NaNs: {torch.isnan(torch.tensor(rewards, dtype=torch.float32)).any()}")
        # print(f"values NaNs: {torch.isnan(torch.tensor(values, dtype=torch.float32)).any()}")
        returns = self.calculate_returns(rewards)
        advantages = self.calculate_advantages(returns, values)

        # print(f"Returns NaNs: {torch.isnan(returns).any()}")
        # print(f"advantages NaNs (after calculation): {torch.isnan(advantages).any()}")

        return episode_reward, states, actions, actions_log_probability, advantages, returns

    ##Updating the model parameters
    #Each training iteration runs the model through a complete episode consisting of many timesteps (until it reaches a terminal condition). In each timestep, we store the policy parameters, the agent’s action, the returns, and the advantages. After each iteration, we update the model based on the policy’s performance through all the timesteps in that iteration.
    #In complex environments, where there are very many timesteps, the training results dataset must be split into batches. The number of timesteps in each batch is called the optimization batch size.
    # Steps to update the model parameters:
    #1. Split the trianing results dataset into batches
    #2. For each batch: - get agen'ts action and predicted value for each state
    #-use these predicted values to estimate the new action probability distribution
    #-use this distribution to calculate the entropy
    #-use this distribution to get the log probability of the actions in the training results dataset. This is the new set of log probabilities of the actinos in the traiging results dataset. The old set of log probabilities of theses same actions was calculated in the training loop explaine in the previous section.
    #-calculate surrogate loss using the action's old and new probability distribution
    #-calculate the policy loss and the value loss using the surrogate loss, the entropy and advantages
    #run .backward() separately on the policy and value losses. This updates the gradients on the loss functions.
    #- Run .step() on the optimizer to update the policy parameters. In this case, we use the adam optimizer to balance speed and robustness.
    #-Accumulate the policy and value losses
    #3. Repeat the above operations (backward pass) on each abtch a few times, depending on the value of the parameter PPO_STEPS. Repeating the backward pass on each batch is computationally efficient because it increases the size of the training data set without having to run additional forward passes. The number of environment steps in each alternation between sampling and optimization is called the iteration batch size.
    #4. Return the average policy loss and value loss

    #Forward Pass: Calculate predictions (e.g., action probabilities, value estimates) and losses for a batch by inputting data through the NN to generate outputs.
    #Backward Pass: Compute gradients of the loss functions wrt the model's parameters and updating them.
    #Instead of performing a single backward pass for each batch and then moving on to the next, PPO repeats the backward pass multiple times on the same batch of data, recalculating gradients and updating the model.
    #Data from environment interactions (states, actions, rewards) is expensive to collect. In complex RL environments, generating new data (via simulations or interactions) can take significant time and resources.
    #Repeating the backward pass on the same batch allows the optimizer to refine the gradients and make multiple updates to the policy without requiring new data from additional forward passes.

    #Why is it more efficient doing multiple backward passes for one forward pass?
    #Gradients are calculated layer-by-layer starting from the output.
    #Every operation in the forward pass must be revisited to compute the derivatives (stored in the computation graph during the forward pass).
    #Gradients flow backward through the network, requiring nearly the same amount of computation as the forward pass. Although, in RL, generating training data often involves running an agent in an environment which is an added computation.
    #Forward pass must be performed once per batch to initialise outputs (e.g. action probabilities, entropy, losses). Repeating the backward pass allows the network to update its weights multiple times without having to recompute these outputs.
    #This avoids the redundant cost of redoing the forward pass for the same data.

    #How can different backward passes can result in different updates dfor the same forward pass data?
    #Each backward pass uses the same precomputed outputs to compute the gradients of the loss function and update weights incrementally, via gradient descent or similar optimization algorithms.
    #In stochastic gradient descent (SGD), the gradient is computed on a small batch or even a single data point. This introduces stochasticity, because the gradient is noisy and not the true gradient over the entire dataset.
    #Repeating the backward pass alone can lead to different results because of the stochasticity in gradient computation (due to small batches).
    #e.g. If you use the same data for the forward pass and then run a backward pass multiple times, the updates you get from the backward pass can differ slightly because of the inherent noise in stochastic updates, particularly when working with smaller batches or noisy environments.
    # e.g. Optimizers like Adam and SGD with momentum use momentum terms that accumulate past gradient information to smooth updates. When you repeat the backward pass, even if the gradients are the same, the accumulated momentum could change the way the optimizer updates the weights.

    def update_policy(self,
            states,
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
            for batch_idx, (states,actions,actions_log_probability_old, advantages, returns) in enumerate(batch_dataset):
                #get new log prob of actions for all input states
                action_pred, value_pred = self.model(states)
                #print("action_pred shape:",action_pred.shape)
                value_pred = value_pred.squeeze(-1)
                #Value_pred is typically the value estimate for each state, which is often predicted by the critic network. Depending on the network architecture, the output might have an extra dimension due to how the network is structured. For instance if your network's final layer produces a shape like (batch_size, 1), which is often the convention when processing batches with NNs
                #print("Logits before softmax:", action_pred)
                action_prob = torch.softmax(action_pred, dim=-1)
                # print("Action probabilities shape:", action_prob.shape)
                # print("Action probabilities sum:", action_prob.sum(dim=-1))
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

    def evaluate(self):
        self.model.eval()
        done = False
        episode_reward = 0
        state,_ = self.env_test.reset()
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                action_pred, _ = self.model(state)
                action_prob = torch.softmax(action_pred, dim=-1)
            action = torch.argmax(action_prob, dim=-1)
            state, reward, done, _,_ = self.env_test.step(action.item())
            episode_reward += reward
        return episode_reward

    def train_with_test(self):
        train_rewards = []
        test_rewards = []
        policy_losses = []
        value_losses = []
        #lens = []

        for episode in range(1, self.max_episodes + 1):
            # Perform a forward pass and collect experience
            train_reward, states, actions, actions_log_probability, advantages, returns = self.forward_pass()

            # Update the policy using the experience collected
            policy_loss, value_loss = self.update_policy(
                states,
                actions,
                actions_log_probability,
                advantages,
                returns)
            # test_reward = self.evaluate()

            # # Visualize the environment if it supports rendering (currently this is done once each episode - might want to change to once every multiple of episodes)
            # if hasattr(self.env, "render") and callable(getattr(self.env, "render", None)):
            #   self.env.render()

            # Log the results
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            train_rewards.append(train_reward)
            #print(train_reward)
            #lens.append(episode_len)
            # test_rewards.append(test_reward)
            self.run["policy_loss"].log(policy_loss)
            self.run["value_loss"].log(value_loss)
            self.run["train_reward"].log(train_reward)
            # self.run["test_reward"].log(test_reward)

            # Calculate the mean of recent rewards and losses for display
            mean_train_rewards = np.mean(train_rewards[-self.n_trials:])
            #mean_test_rewards = np.mean(test_rewards[-self.n_trials:])
            mean_abs_policy_loss = np.mean(np.abs(policy_losses[-self.n_trials:]))
            mean_abs_value_loss = np.mean(np.abs(value_losses[-self.n_trials:]))

            # Print results at specified intervals
            if episode % self.print_interval == 0:
                print(f'Episode: {episode:3} | \
                    Mean Train Rewards: {mean_train_rewards:3.1f} \
                    | Mean Abs Policy Loss: {mean_abs_policy_loss:2.2f} \
                    | Mean Abs Value Loss: {mean_abs_value_loss:2.2f} ')
                    


                                    # | Mean Test Rewards: {mean_test_rewards:3.1f} \
                                    #| "Episode Len: {np.mean(lens[-self.n_trials:])}



            # # Check if reward threshold is reached
            # if mean_test_rewards >= self.reward_threshold:
            #     print(f'Reached reward threshold in {episode} episodes')
            #     break
        # Check if the environment has a close method before calling it
        # if hasattr(self.env, "close") and callable(getattr(self.env, "close", None)):
        #   self.env.close() #Close environment visualisation after training is done.
        return policy_losses, value_losses, train_rewards, test_rewards

#train_rewards measures the cumulative reward obtained in the training environment while the agent is learning
#test_rewards measures the cumulative reward obtained in the test environment with no learning updates- only evaluating the policy as it currently stands
#If you call agent.train_with_test(), test_rewards evaluates the performace of the policy in a test environment (env_test) after updates made during the training phase. It provides insight into how well the policy generalizes to unseen or separate scenarios.
#The inclusion of test_rewards in the same loop as train_rewards allows you to monitor how the policy generalizes immediately after each training update which provides a real-time understanding of the trade-off between overfitting to the training environment and achieving generalizable performnace.
#Keeping them in the same loop ensures both metrics correspond to the same stage of training. This helps in diagnozing potential issues, such as overfitting.

# %%
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

# %%
# # def main():
# #     my_api="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDg3ZjNlYi04MWI3LTQ1ODctOGIxNS1iNTY3ZjgzMGYzMzYifQ=="
# #     run = neptune.init_run(
# #     project="EnergyGridRL/SA-Gridworld-PPO",
# #     api_token=my_api
# #     )  # your credentials
# #     # Create the environment
# #     params = {
# #         "grid_size": 5,
# #         "grid_size_test": 5,
# #         "optimizer_name": "Adam",
# #         "MAX_EPISODES":500,
# #         "DISCOUNT_FACTOR":0.99,
# #         "REWARD_THRESHOLD": -10,
# #         "PRINT_INTERVAL": 10,
# #         "PPO_STEPS":8,
# #         "N_TRIALS": 100,
# #         "EPSILON": 0.5,
# #         "ENTROPY_COEFFICIENT":0.1,
# #         "HIDDEN_DIMENSIONS": 64,
# #         "DROPOUT": 0.2,
# #         "LEARNING_RATE":0.001,
# #         "BATCH_SIZE":128
# #     }

#     run["parameters"] = params
#     env = GridWorldDynTargetEnv(size=params["grid_size"])
#     env_test =GridWorldDynTargetEnv(size=params["grid_size_test"])
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     agent = PPO_agent(env= env,
#                  env_test=env_test,
#                  run = run,
#                  device= device,
#                  hidden_dimensions = params["HIDDEN_DIMENSIONS"],
#                  dropout= params["DROPOUT"],
#                  discount_factor = params["DISCOUNT_FACTOR"],
#                  optimizer_name=params["optimizer_name"],
#                  max_episodes=params["MAX_EPISODES"],
#                  reward_threshold= params["REWARD_THRESHOLD"],
#                  print_interval =params["PRINT_INTERVAL"],
#                  PPO_steps =params["PPO_STEPS"],
#                  n_trials= params["N_TRIALS"],
#                  epsilon =params["EPSILON"],
#                  entropy_coefficient=params["ENTROPY_COEFFICIENT"],
#                  learning_rate=params["LEARNING_RATE"],
#                  batch_size=params["BATCH_SIZE"])



#     policy_losses, value_losses, train_rewards, test_rewards= agent.train_with_test()
#     # Plotting the rewards and losses
#     run.stop()
#     print(policy_losses)
#     print(value_losses)
#     print(train_rewards)
#     print(test_rewards)

#     policy_losses = np.array(policy_losses)
#     value_losses = np.array(value_losses)
#     train_rewards = np.array(train_rewards)
#     test_rewards = np.array(test_rewards)

#     plot_train_rewards(train_rewards, agent.reward_threshold)
#     plot_test_rewards(test_rewards, agent.reward_threshold)
#     plot_losses(policy_losses, value_losses)

# %%
# if __name__ == "__main__":
#     main()

# %%
def mainCartPole():
    my_api="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDg3ZjNlYi04MWI3LTQ1ODctOGIxNS1iNTY3ZjgzMGYzMzYifQ=="
    run = neptune.init_run(
    project="EnergyGridRL/PPO-Cartpole",
    api_token=my_api
    )  # your credentials
    # Create the environment
    params = {
        "optimizer_name": "Adam",
        "MAX_EPISODES":1000,
        "DISCOUNT_FACTOR":0.99,
        "REWARD_THRESHOLD": 475,
        "PRINT_INTERVAL": 10,
        "PPO_STEPS":8,
        "N_TRIALS": 100,
        "EPSILON": 0.2,
        "ENTROPY_COEFFICIENT":0.01,
        "HIDDEN_DIMENSIONS": 64,
        "DROPOUT": 0.2,
        "LEARNING_RATE":3e-4,
        "BATCH_SIZE":128
    }

    run["parameters"] = params
    env = gym.make('CartPole-v1')
    env_test = gym.make('CartPole-v1')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agent = PPO_agent(env= env,
                 env_test=env_test,
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



    policy_losses, value_losses, train_rewards, test_rewards= agent.train_with_test()
    # Plotting the rewards and losses
    run.stop()
    print(policy_losses)
    print(value_losses)
    print(train_rewards)
    print(test_rewards)

    policy_losses = np.array(policy_losses)
    value_losses = np.array(value_losses)
    train_rewards = np.array(train_rewards)
    test_rewards = np.array(test_rewards)

    plot_train_rewards(train_rewards, agent.reward_threshold)
    plot_test_rewards(test_rewards, agent.reward_threshold)
    plot_losses(policy_losses, value_losses)

# %%
#mainCartPole()

# %%
def mainMinatar():
    my_api="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZDg3ZjNlYi04MWI3LTQ1ODctOGIxNS1iNTY3ZjgzMGYzMzYifQ=="
    run = neptune.init_run(
    project="EnergyGridRL/minatar-PPO",
    api_token=my_api
    )  # your credentials
    # Create the environment
    params = {
        "optimizer_name": "Adam",
        "MAX_EPISODES":10000,
        "DISCOUNT_FACTOR":0.99,#0.8 to 0.9997. Common 0.99
        "REWARD_THRESHOLD": 100,
        "PRINT_INTERVAL": 10,
        "PPO_STEPS":5,
        "N_TRIALS": 100,
        "EPSILON": 0.1,
        "ENTROPY_COEFFICIENT":0.01,#0 to 0.01
        "HIDDEN_DIMENSIONS": 64,
        "DROPOUT": 0.2,
        "LEARNING_RATE":1e-4,#0.003 to 5e-6
        "BATCH_SIZE":32
    }

    run["parameters"] = params
    #env = gym.make("LunarLander-v3", render_mode="human")
    env = Environment(env_name="breakout")
    env_test = Environment(env_name="breakout")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    agent = PPO_agent(env= env,
                 env_test=env_test,
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



    policy_losses, value_losses, train_rewards, test_rewards= agent.train_with_test()
    # Plotting the rewards and losses
    run.stop()
    print(policy_losses)
    print(value_losses)
    print(train_rewards)
    print(test_rewards)

    policy_losses = np.array(policy_losses)
    value_losses = np.array(value_losses)
    train_rewards = np.array(train_rewards)
    test_rewards = np.array(test_rewards)

    plot_train_rewards(train_rewards, agent.reward_threshold)
    plot_test_rewards(test_rewards, agent.reward_threshold)
    plot_losses(policy_losses, value_losses)

# %%
mainMinatar()

# %%
# envTemp = gym.make("CartPole-v1")
# help(envTemp.step)
# help(envTemp.reset)
# obs, info = envTemp.reset()
# print(obs)
# print(info)
#When writing my PPO agent as applied to Gridworld, I assumed env.step returned (observation, reward, terminated, truncated, info) and env.reset returned (obs,info)
#Most gynmaisum environments follow this convention, though I should check when using new ones
#need to censure my other agents follow the same convention

# %%
# print(envTemp.observation_space)
# #print(envTemp.action_space.n)

# # %%
# env = GridWorldDynTargetEnv(size=5)
# print(env.observation_space)

# # %%
# env = GridWorldDynTargetEnv(size=5)
# print(env.action_space.n)

# # %%
# print(envTemp.action_space.n)

# # %%
# main()

# # %%
# #testing sync


