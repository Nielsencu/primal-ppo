import os
import numpy as np
import setproctitle
import torch
import wandb
from util import Sequence

from alg_parameters import SetupParameters, RecordingParameters, EnvParameters, EvalParameters, all_args
from mapf_gym import FixedMapfGym, MapfGym
from model import Model
from util import set_global_seeds, write_to_wandb, make_gif, OneEpPerformance, NetParameters, getFreeCell, returnAsType
from map_generator import generateWarehouse
from mapf_gym import Human
from astar_4 import astar_4

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
print("Welcome to MAPF!\n")

def main():
    """main code"""
    if RecordingParameters.WANDB:
        wandb_id = wandb.util.generate_id()
        wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                   name=RecordingParameters.EXPERIMENT_NAME,
                   entity=RecordingParameters.ENTITY,
                   notes=RecordingParameters.EXPERIMENT_NOTE,
                   config=all_args,
                   id=wandb_id,
                   resume='allow')
        print('id is:{}'.format(wandb_id))
        print('Launching wandb...\n')

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    model = Model(0, global_device, True)
    # Start evaluation
    try:
        # get data from multiple processes
        # evaluate training model
        with torch.no_grad():
            # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
            # global_device, save_gif0, curr_steps, True)
            evaluate(model, global_device, False)
    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")

    # killing
    if RecordingParameters.WANDB:
        wandb.finish()


def evaluate(model, device, greedy):
    """Evaluate Model."""
    episodePerformances = []
    fixedEpisodeInfos = []
    for i in range(EvalParameters.EPISODES):
        obstacleMap = generateWarehouse(num_block=EnvParameters.WORLD_SIZE)
        humanStart = Human.getEntrance(obstacleMap)
        humanGoal = getFreeCell(obstacleMap)
        obstacleMap[returnAsType(humanStart,'mat')] = 1
        # human_path = astar_4(obstacleMap, humanStart, humanGoal)[0]
        # human_path =  human_path[::-1]
        agentStartsList = []
        # Generate the starting pos for agents while respecting other agents' spawning point
        for agentIdx in range(len(EvalParameters.N_AGENTS)):
            agentStart = np.array(getFreeCell(obstacleMap))
            obstacleMap[agentStart] = 2
            agentStartsList.append(agentStart)
        
        pathLengths = [0 for _ in range(EvalParameters.N_AGENTS)]
        step = 0
        agentGoalSequences = [Sequence() for _ in range(len(EvalParameters.N_AGENTS))]
        goalSequencesComplete = [False for _ in range(EvalParameters.N_AGENTS)]
        while not np.all(goalSequencesComplete):
            for agentIdx in range(len(EvalParameters.N_AGENTS)):
                if goalSequencesComplete[agentIdx]:
                    continue
                agentGoalSequence = agentGoalSequences[agentIdx]
                if step == 0:
                    agentStart = agentStartsList[agentIdx]
                else:
                    agentStart = agentGoalSequence.getAtPos(-1)
                agentGoal = getFreeCell(obstacleMap)
                obstacleMap[np.array(agentGoal)] = 3
                
                agentGoalSequence.add(agentGoal)
                
                optimalAgentPath = astar_4(obstacleMap, agentStart, agentGoal)
                pathLengths[agentIdx] += optimalAgentPath
                if pathLengths[agentIdx] > EvalParameters.MAX_STEPS:
                    goalSequencesComplete[agentIdx] = True
            if step == 0:
                # Free the starting point of all agents to generate consecutive goals
                for agentStart in agentStartsList:
                    obstacleMap[agentStart] = 0
            else:
                # Free the previous goal
                for agentGoalSequence in agentGoalSequences:
                    obstacleMap[np.array(agentGoalSequence.getAtPos(-2))] = 0
            step +=1
        agentStartsList = [tuple(agentStart) for agentStart in agentStartsList]
        fixedEpisodeInfos.append((obstacleMap, agentStartsList, agentGoalSequences, humanStart, humanGoal))
    
    for model_name, net_path_checkpoint in EvalParameters.MODELS:
        net_dict = torch.load(net_path_checkpoint)
        model.network.load_state_dict(net_dict['model'])
    
        for i in range(EvalParameters.EPISODES):
            curr_episode = i
            oneEpisodePerformance = OneEpPerformance()
            episode_frames = []
            
            obstaclesMap, agentStartsList, agentGoalsList, humanStart, humanGoal = fixedEpisodeInfos[curr_episode]
            env = FixedMapfGym(obstaclesMap, agentStartsList, agentGoalsList, humanStart, humanGoal)
            
            obs, vecs = env.getAllObservations()

            numAgent = EnvParameters.N_AGENTS

            hidden_state = (torch.zeros((numAgent, NetParameters.NET_SIZE )).to(device),
                            torch.zeros((numAgent, NetParameters.NET_SIZE )).to(device))

            episode_frames.append(env._render())

            # stepping
            for _ in range(EvalParameters.MAX_STEPS):

                # predict
                actions, pre_block, hidden_state, v, ps = \
                    model.evaluate(obs, vecs, hidden_state,greedy,numAgent)                                               


                actionStatus = env.getActionStatus(actions)
                
                for  value in actionStatus:
                    if value==-1:
                        oneEpisodePerformance.staticCollide+=1
                    elif value==-2:
                        oneEpisodePerformance.humanCollide +=1
                    elif value==-3:
                        oneEpisodePerformance.agentCollide+=1


                rewards,shadowGoals = env.calculateActionReward(actions, actionStatus)
                costRewards = env.calculateCostReward(actions)

                oneEpisodePerformance.shadowGoals+=shadowGoals  
            
                goalsReached, constraintsViolated = env.jointStep(actions, actionStatus)

                for i, value in enumerate(goalsReached):
                    if(value==1):
                        rewards[0,i]+=EnvParameters.GOAL_REWARD


                oneEpisodePerformance.episodeReward += np.sum(rewards)
                
                oneEpisodePerformance.totalGoals+=np.sum(goalsReached)

                oneEpisodePerformance.episodeCostReward += np.sum(costRewards)
                oneEpisodePerformance.constraintViolations += np.sum(constraintsViolated)

                obs, vecs = env.getAllObservations() 

                episode_frames.append(env._render())
                    
            if RecordingParameters.WANDB:
                # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                write_to_wandb(curr_episode, oneEpisodePerformance, evaluate=True, greedy=False)
                print('episode: {},episode reward: {}, episode cost reward: {} human_coll: {}, static_coll: {}, agent_coll: {}, total_goals: {} shadow_goals: {} \n'.format(
                        curr_episode, oneEpisodePerformance.episodeReward, oneEpisodePerformance.episodeCostReward, oneEpisodePerformance.humanCollide, 
                        oneEpisodePerformance.staticCollide, oneEpisodePerformance.agentCollide, oneEpisodePerformance.totalGoals, oneEpisodePerformance.shadowGoals)) 
                    
            episodePerformances.append(oneEpisodePerformance)

            if not os.path.exists(RecordingParameters.GIFS_PATH):
                os.makedirs(RecordingParameters.GIFS_PATH)
            images = np.array(episode_frames[:-1])
            make_gif(images,
                '{}/episode_{:d}_reward{:.1f}_human_coll{:.1f}_totalGoals{}_shadowGoals{}_staticColl{:d}_agentColl{:d}.gif'.format(
                    RecordingParameters.GIFS_PATH,
                    curr_episode, oneEpisodePerformance.episodeReward,
                    oneEpisodePerformance.humanCollide, oneEpisodePerformance.totalGoals,
                    oneEpisodePerformance.shadowGoals, oneEpisodePerformance.staticCollide, oneEpisodePerformance.agentCollide))
    return episodePerformances

if __name__ == "__main__":
    main()

