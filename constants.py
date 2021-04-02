#
# define constants here
#
MIN_HIDDEN_LAYERS=1
MAX_HIDDEN_LAYERS=4

MIN_NODES=7
MAX_NODES=15

SYNCH_STEPS=2

DIR_PATH='./datasets/' ###configure this on local repo

TEST_SPLIT_FRACTION=0.2
RANDOM_SAMPLE_LIMIT=5
SAMPLE_SIZE=100
EPOCHS=20

NUM_AGENTS=2
BATCH_SIZE = 100

#Agent constants
ALPHA=1
BETA=1
GAMMA=0.99
TAU=0.99
BUF_LEN=100000
H1_DIMS = 40
H2_DIMS = 30
N_ACTIONS=1
MAX_EPISODES = 30
MAX_STEPS = 50
MAX_ACTION = [4,10]

