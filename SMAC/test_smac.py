from smacv2.env import StarCraft2Env  
import numpy as np  

# Initialize the environment with a sample map
env = StarCraft2Env(map_name="3m")  
obs, info = env.reset()  
print(env.n_agents)

env1 = StarCraft2Env(map_name="2s3z")  
print(env1.n_agents)

# Run 5 random steps in the environment  
for _ in range(5):  
    avail_actions = env.get_avail_actions()  # Get available actions for all agents
    actions = [np.random.choice(np.where(avail_actions[i])[0]) for i in range(env.n_agents)]  # Select valid actions  
    reward, terminated, info = env.step(actions)
    obs = env.get_obs()
    print(f"Step Reward: {reward}, Done: {terminated}, Info: {info}")
    #print(f"obs: {obs}")


env.close()  
print("SMACv2 test completed successfully!")
