from parallelAgent import MADDPG
from environment import Environment
import constants as C
import torch.multiprocessing as mp

if __name__ == '__main__':
  mp.freeze_support()
  mp.set_start_method('spawn')
  env = Environment()
  
  controller = MADDPG(env, num_agents=C.NUM_AGENTS, alpha=C.ALPHA, beta=C.BETA, tau=C.TAU, input_dims= [C.NUM_AGENTS] 
                      ,n_actions=C.N_ACTIONS, hd1_dims = C.H1_DIMS, hd2_dims = C.H2_DIMS, mem_size = C.BUF_LEN,gamma = C.GAMMA,
                      batch_size = C.BATCH_SIZE)
	#call multiAgent here
  controller.parallel(max_episode=C.MAX_EPISODES, max_steps=C.MAX_STEPS)