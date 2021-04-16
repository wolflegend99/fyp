import torch as T
import numpy as np
from agent import DDPGAgent
import constants as C
#from torch.multiprocessing import Process, Lock, Value, Array, Manager, set_start_method
import torch.multiprocessing as mp

device = T.device("cpu")
if T.cuda.is_available():
    device = T.device("cuda")

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
                                 batch_size = self.batch_size,agent_no =  i) for i in range(self.num_agents)]
        self.agents_states = None
        self.x = 0
        
    
    def run(self, max_episode, max_steps, agent_no, l, l1, common_state, env, agents):
        return_list = []
        means = 0
        env = env
        agents = agents
        max_reward = float('-inf')
        for i in range(max_episode):
            #print("Episode : {}".format(i))
            returns = 0
            state = env.reset1(agent_no)
            agent_states = state
            steps = 0
            agents[agent_no].actionNoise.reset()
            while steps != max_steps:
                steps += 1
                action, rounded_action = agents[agent_no].choose_action(agent_states, agent_no)
                
                #rint(rounded_action)
                next_state, reward = env.step(rounded_action, agent_no)
                done = False
                if reward == 0:
                    done = True
                agents[agent_no].store_transition(agent_states, action, reward, next_state)
                agents[agent_no].learn([agent_states], [action], [reward], [next_state])
                returns += reward
                agent_states = next_state
                
                l1.acquire()
                try:
                  if reward >= max_reward:
                    common_state[agent_no] = agent_states[agent_no]
                    max_reward = reward
                finally:
                  l1.release()
                
                #debug info goes here...
                if steps % C.SYNCH_STEPS == 0:
                    #l1.acquire()
                    #try:
                    print("Syncing at step ", steps, "...")
                    #common_state[agent_no] = agent_states[agent_no]
                    synched_state = [common_state[0], common_state[1]]
                    agent_states = env.synch1(synched_state, agent_no)
                    #finally:
                    #  l1.release()
                #steps+=1
                #with l:
                l.acquire()
                try:
                  print("Episode: ", i+1)
                  print("Step: ", steps)
                  print("Agent : ", agent_no)
                  print("Action: ", action)
                  print("Next state : ", next_state)
                  print("Agent ", agent_no, " -> Reward", reward)
                  print("Agent ", agent_no, " -> Returns ",returns)
                  print("\n-----------------------------------------------------------------\n")
                finally:
                  l.release()
                '''
                l.acquire()
                try:
                    print("Episode: ", i+1)
                    print("Step: ", steps)
                    print("Agent : ", agent_no)
                    print("Action: ", action)
                    print("Next state : ", next_state)
                    print("Agent ", agent_no, " -> Reward", reward)
                    print("Agent ", agent_no, " -> Returns ",returns)
                    print("\n-----------------------------------------------------------------\n")
                finally:
                    l.release()
                '''
                
            #for j in range(self.num_agents):
            return_list.append(returns)
            means = np.mean(return_list[-20:])
            print("Score Model1 : ",means)
            #print("Score model2 : ",means[1])
        '''
        l.acquire()
        try:
            common_state[agent_no] += 1
            print(common_state[agent_no])
        finally:
            l.release()
        '''
    
    def parallel(self,max_episode, max_steps):
      	
        #mp.freeze_support()
        #mp.set_start_method('fork')
        val = mp.Value('i', 1)
        arr = mp.Array('i', [3,3])
        m = mp.Manager()
        printlock = m.Lock()
        synchlock = m.Lock()
        p1 = mp.Process(target = self.run, args = (max_episode, max_steps, 0,
                                               printlock, synchlock, arr, self.env,
                                                self.agents))
        p2 = mp.Process(target = self.run, args = (max_episode, max_steps, 1,
                                               printlock, synchlock, arr, self.env,
                                                self.agents))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        '''
        p1= Process(target = self.run, args = (3,4,0,lock, val))
        p2 = Process(target = self.run, args = (4,4,1,lock, val))
        p1.start()
        p2.start()
        p1.join()
        p2.join()
        print(val.value)
        '''