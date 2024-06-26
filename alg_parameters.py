import datetime

""" Hyperparameters """

class EvalParameters:
    N_AGENTS = 2
    MAX_STEPS = 2 ** 8
    EPISODES = 100
    HUMAN_MOVEMENT_TYPE = 0 # 0 FOR LOOPING HUMAN, 1 FOR FIXED GOALS HUMAN
    METRICS_JSON_PATH = './all_metrics_patrol.json'
    FIXED_EPISODE_INFOS_PATH = './fixed_episode_infos_patrol'
    LOAD_FIXED_EPISODE_INFOS = False
    MODEL_PATH = "../"
    MODELS = [
        # ("PPO",  "ppolag-humpred-dangarea/net_checkpoint.pkl"),
        ("PPO-HP",  "vanilla-hp/net_checkpoint.pkl"),
        ("PPO-DA",  "vanilla-da/net_checkpoint.pkl"),
        # ("PPO-HP+DA",  "vanilla-hp-da/net_checkpoint.pkl"),
        ("PPOL-HP",  "ppolag-hp/net_checkpoint.pkl"),
        # ("PPOL-DA",  "ppolag-humpred-dangarea/net_checkpoint.pkl"),
        ("PPOL-HP+DA",  "ppolag-hp-da/net_checkpoint.pkl"),
        ("PPOL-HP+DA(PIDL-0.95)",  "ppolag-hp-da-pid95/net_checkpoint.pkl"),
        # ("PPOL-HP+DA(PIDL-0.5)",  "ppolag-hp-da-pid50/net_checkpoint.pkl")
        ("PPOL-HP+DA(PIDL-0.95)-V2",  "ppolag-hp-da-pid95-cppopid/net_checkpoint.pkl"),
        #("PPOL-HP+DA(PIDL-0.95)-V2",  "ppolag-hp-da-pid95-cppopid/net_checkpoint.pkl"),
        #("PPOL-HP+DA(PIDL-0.95)-V2",  "ppolag-hp-da-pid95-cppopid/net_checkpoint.pkl")
    ]

class EnvParameters:
    N_AGENTS = 2  # number of agents used in training
    N_ACTIONS = 5
    EPISODE_LEN = 256  # maximum episode length in training
    FOV_SIZE = 9
    FOV_Heuristic = 5
    WORLD_SIZE = (10, 40)
    OBSTACLE_PROB = (0.0, 0.3)

    ACTION_COST = -0.3
    IDLE_COST = -0.3
    GOAL_REWARD = 1.5
    COLLISION_COST = -2
    HUMAN_COLLISION_COST = -2
    REPEAT_POS = -0.35
    BLOCKING_COST = 0
    PENALTY_RADIUS = 5
    CONSTRAINT_VIOLATION_COST = -1.0
    
    LIFELONG = True


class TrainingParameters:
    lr = 1e-5
    GAMMA = 0.95  # discount factor
    LAM = 0.95  # For GAE
    CLIP_RANGE = 0.2
    MAX_GRAD_NORM = 10
    ENTROPY_COEF = 0.01
    VALUE_COEF = 0.08
    POLICY_COEF = 10
    VALID_COEF = 0.5
    BLOCK_COEF = 0.5
    COST_VALUE_COEF = 0.0
    COST_COEF = 0.0
    COST_LIMIT_PER_AGENT = 5
    N_EPOCHS = 10
    N_ENVS = 16  # number of processes
    N_MAX_STEPS = 3e7  # maximum number of time steps used in training
    N_STEPS = 2 ** 8  # number of time steps per process per data collection
    MINIBATCH_SIZE = int(2 ** 8)
    DEMONSTRATION_PROB = 0  # imitation learning rate
    
    # Nielsen's Testing
    COST_LIMIT_PER_AGENT = 5
    COST_VALUE_COEF = 0.0
    COST_COEF = 0.0
    
    USE_INFLATED_HUMAN = True
    USE_HUMAN_TRAJECTORY_PREDICTION = True
    K_TIMESTEP_PREDICT = 5
    
    # CPPO-PID (Responsive Safety in Reinforcement Learning by PID Lagrangian Methods)
    MINUS_ADV_WITH_CADV = True
    
    
class LagrangianParameters:
    LAGRANGIAN_TYPE = 0 # 0 for base Lagrangian, 1 for PID Lagrangian
    
    INIT_VALUE = 1.0
    UPPER_BOUND = 20.0
    
    # Params for Base lagrangian
    LR = 5e-2
    # Params for PID Lagrangian
    KP = 0.1
    KI = 0.01
    KD = 0.01
    
    # Set moving avg alpha to 0 to transform to classic PID
    COST_MOVING_AVG_ALPHA = 0.95
    DELTA_MOVING_AVG_ALPHA = 0.95

class NetParameters:
    NET_SIZE = 512
    NUM_CHANNEL = 5 + int(TrainingParameters.USE_HUMAN_TRAJECTORY_PREDICTION)  # number of channels of observations -[FOV_SIZE x FOV_SIZEx NUM_CHANNEL]
    GOAL_REPR_SIZE = 12
    VECTOR_LEN = 4  # [dx, dy, d total, action t-1]

class SetupParameters:
    SEED = 1234
    USE_GPU_LOCAL = False
    USE_GPU_GLOBAL = True
    NUM_GPU = 1
class RecordingParameters:
    RETRAIN = False
    WANDB = True
    TENSORBOARD = False
    TXT_WRITER = True
    ENTITY = 'nielsencugito'
    TIME = datetime.datetime.now().strftime('%d-%m-%y%H%M')
    EXPERIMENT_PROJECT = 'HumanAware'
    EXPERIMENT_NAME = 'EVAL-PATROLHUMAN-LOCAL'
    EXPERIMENT_NOTE = 'Handled Swapping coll with Humans'
    SAVE_INTERVAL = 5e5  # interval of saving model0
    BEST_INTERVAL = 0  # interval of saving model0 with the best performance
    GIF_INTERVAL = 1  # interval of saving gif
    EVAL_INTERVAL = TrainingParameters.N_ENVS * TrainingParameters.N_STEPS  # interval of evaluating training model0
    EVAL_EPISODES = 1  # number of episode used in evaluation
    RECORD_BEST = False
    MODEL_PATH = '../models' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    GIFS_PATH = '../gifs' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    SUMMARY_PATH = './summaries' + '/' + EXPERIMENT_PROJECT + '/' + EXPERIMENT_NAME + TIME
    TXT_NAME = 'alg.txt'
    LOSS_NAME = ['all_loss', 'policy_loss', 'policy_entropy', 'critic_loss', 'valid_loss',
                 'cost_critic_loss', 'cost_loss', 'clipfrac',
                 'grad_norm', 'advantage', 'cost_advantage', 'lagrangian']


all_args = {'N_AGENTS': EnvParameters.N_AGENTS, 'N_ACTIONS': EnvParameters.N_ACTIONS,
            'EPISODE_LEN': EnvParameters.EPISODE_LEN, 'FOV_SIZE': EnvParameters.FOV_SIZE,
            'WORLD_SIZE': EnvParameters.WORLD_SIZE,
            'OBSTACLE_PROB': EnvParameters.OBSTACLE_PROB,
            'ACTION_COST': EnvParameters.ACTION_COST,
            'IDLE_COST': EnvParameters.IDLE_COST, 'GOAL_REWARD': EnvParameters.GOAL_REWARD,
            'COLLISION_COST': EnvParameters.COLLISION_COST,
            'BLOCKING_COST': EnvParameters.BLOCKING_COST,
            'lr': TrainingParameters.lr, 'GAMMA': TrainingParameters.GAMMA, 'LAM': TrainingParameters.LAM,
            'CLIPRANGE': TrainingParameters.CLIP_RANGE, 'MAX_GRAD_NORM': TrainingParameters.MAX_GRAD_NORM,
            'ENTROPY_COEF': TrainingParameters.ENTROPY_COEF,
            'VALUE_COEF': TrainingParameters.VALUE_COEF,
            'POLICY_COEF': TrainingParameters.POLICY_COEF,
            'VALID_COEF': TrainingParameters.VALID_COEF, 'BLOCK_COEF': TrainingParameters.BLOCK_COEF,
            'N_EPOCHS': TrainingParameters.N_EPOCHS, 'N_ENVS': TrainingParameters.N_ENVS,
            'N_MAX_STEPS': TrainingParameters.N_MAX_STEPS,
            'N_STEPS': TrainingParameters.N_STEPS, 'MINIBATCH_SIZE': TrainingParameters.MINIBATCH_SIZE,
            'DEMONSTRATION_PROB': TrainingParameters.DEMONSTRATION_PROB,
            'NET_SIZE': NetParameters.NET_SIZE, 'NUM_CHANNEL': NetParameters.NUM_CHANNEL,
            'GOAL_REPR_SIZE': NetParameters.GOAL_REPR_SIZE, 'VECTOR_LEN': NetParameters.VECTOR_LEN,
            'SEED': SetupParameters.SEED, 'USE_GPU_LOCAL': SetupParameters.USE_GPU_LOCAL,
            'USE_GPU_GLOBAL': SetupParameters.USE_GPU_GLOBAL,
            'NUM_GPU': SetupParameters.NUM_GPU, 'RETRAIN': RecordingParameters.RETRAIN,
            'WANDB': RecordingParameters.WANDB,
            'TENSORBOARD': RecordingParameters.TENSORBOARD, 'TXT_WRITER': RecordingParameters.TXT_WRITER,
            'ENTITY': RecordingParameters.ENTITY,
            'TIME': RecordingParameters.TIME, 'EXPERIMENT_PROJECT': RecordingParameters.EXPERIMENT_PROJECT,
            'EXPERIMENT_NAME': RecordingParameters.EXPERIMENT_NAME,
            'EXPERIMENT_NOTE': RecordingParameters.EXPERIMENT_NOTE,
            'SAVE_INTERVAL': RecordingParameters.SAVE_INTERVAL, "BEST_INTERVAL": RecordingParameters.BEST_INTERVAL,
            'GIF_INTERVAL': RecordingParameters.GIF_INTERVAL, 'EVAL_INTERVAL': RecordingParameters.EVAL_INTERVAL,
            'EVAL_EPISODES': RecordingParameters.EVAL_EPISODES, 'RECORD_BEST': RecordingParameters.RECORD_BEST,
            'MODEL_PATH': RecordingParameters.MODEL_PATH, 'GIFS_PATH': RecordingParameters.GIFS_PATH,
            'SUMMARY_PATH': RecordingParameters.SUMMARY_PATH,
            'TXT_NAME': RecordingParameters.TXT_NAME}
