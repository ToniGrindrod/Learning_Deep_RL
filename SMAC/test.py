# from smac.env import StarCraft2Env
# env = StarCraft2Env(map_name="3m")  # Adjust for SMAC Lite if needed
# obs, info = env.reset()
# print(obs)
# print(info)
# agents_actions= env.get_avail_actions()
# agents_obs=env.get_obs()
# for i in range(len(agents_obs)):
#     print(agents_obs[i]==obs[i])
# # done = False
# # step_count = 0

# # while not done:
# #     actions = [env.action_space[i].sample() for i in range(env.n_agents)]  # Random actions
# #     obs, reward, done, truncated, info = env.step(actions)
# #     step_count += 1

# # print(f"Episode ended after {step_count} steps")
# # env.close()

class Testfn:
    def __init__(self):
        self.count=0
    def update(self):
        self.count+=1
    def check(self):
        while self.count<5:
            self.update()
            print(self.count)


tester=Testfn()
tester.check()