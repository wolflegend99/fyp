import torch as T
import numpy as np
from agent import DDPGAgent
import constants as C
class MADDPG:

    def __init__(self, env, num_agents, alpha, beta, tau, input_dims, n_actions,
                 hd1_dims = 400, hd2_dims = 300, mem_size = 1000000,
                 gamma = 0.99, batch_size = 64):
        self.env = env
        self.num_agents = num_agents
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.agents = [DDPGAgent(alpha = self.alpha, beta = self.beta, tau = self.tau, 
                                 input_dims = input_dims, n_actions = n_actions, hd1_dims = hd1_dims,
                                 hd2_dims = hd2_dims, mem_size = mem_size, gamma = self.gamma,
                                 batch_size = self.batch_size, i) for i in range(self.num_agents)]
        self.agents_states = None
        
    
    def run(self, max_episode, max_steps):
        return_list = [[] for i in range(self.num_agents)]
        means = [0 for i in range(self.num_agents)]
        
        for i in range(max_episode):
            returns = [0 for i in range(self.num_agents)]
            state = self.env.reset()
            agent_states = [state for i in range(self.num_agents)]
            steps = 0
            agent_no = 0
            for i in range(self.num_agents):
                self.agents[i].actionNoise.reset()
            while steps != self.num_agents * max_steps:
                steps += 1
                action, rounded_action = self.agents[agent_no].choose_action(agent_states[agent_no], agent_no)
                print(action)
                next_state, reward = self.env.step(rounded_action, agent_no)
                done = False
                if reward == 0:
                    done = True
                self.agents[agent_no].store_transition(agent_states[agent_no], action, reward, next_state)
                print("Next state : ", next_state)
                self.agents[agent_no].learn()
                returns[agent_no] += reward
                agent_states[agent_no] = next_state
                agent_no = (agent_no + 1) % self.num_agents
                if steps % C.SYNCH_STEPS == 0:
                    synched_states = self.env.synch()
                    agent_states = [synched_states for i in range(self.num_agents)]
                print("Steps: ", steps)
                print("Returns0 : ", returns[0])
                print("Returns1 : ", returns[1])
            for i in range(self.num_agents):
                return_list[i].append(returns[i])
                means[i] = np.mean(return_list[i][-20:])
            print("Score Model1 : ",means[0])
            print("Score model2 : ",means[1])
