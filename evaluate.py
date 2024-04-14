import os
import numpy as np
import setproctitle
import torch
import wandb
from util import Sequence
import copy

from alg_parameters import SetupParameters, RecordingParameters, EnvParameters, EvalParameters, all_args
from mapf_gym import FixedMapfGym
from model import Model
from util import set_global_seeds, write_to_wandb_with_run, make_gif, OneEpPerformance, NetParameters, getFreeCell, returnAsType
from map_generator import generateWarehouse
from mapf_gym import Human
from astar_4 import astar_4
import json

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
print("Welcome to MAPF!\n")

channels = [5,6]
curChannel = 1
def getOtherChannel():
    return int(not(curChannel))

def switchChannel():
    global curChannel
    curChannel = getOtherChannel()

def createFixedEpisodeInfo():
    return {'obstacleMap' : [], 'agentsSequence' : [], 'humanStart' : [], 'humanGoal' : [], 'numEpisodes': 0}

def getFixedEpisodeInfosFolder():
    return EvalParameters.FIXED_EPISODE_INFOS_PATH

def getJsonDir():
    return f"{getFixedEpisodeInfosFolder()}/infos.json"

def generateFixedEpisodeInfos():
    fixedEpisodeInfos = createFixedEpisodeInfo()
    for i in range(EvalParameters.EPISODES):
        obstacleMap = generateWarehouse(num_block=EnvParameters.WORLD_SIZE)
        tempMap = np.copy(obstacleMap)
        humanStart = Human.getEntrance(tempMap)
        humanGoal = getFreeCell(tempMap)
        tempMap[returnAsType(humanStart,'mat')] = 1
        # human_path = astar_4(tempMap, humanStart, humanGoal)[0]
        # human_path =  human_path[::-1]
        agentsSequence = [Sequence() for _ in range(EvalParameters.N_AGENTS)]
        # Generate the starting pos for agents while respecting other agents' spawning point
        for agentIdx in range(EvalParameters.N_AGENTS):
            agentStart = getFreeCell(tempMap)
            tempMap[agentStart] = 2
            agentsSequence[agentIdx].add(agentStart)
        
        pathLengths = [0 for _ in range(EvalParameters.N_AGENTS)]
        goalSequencesComplete = [False for _ in range(EvalParameters.N_AGENTS)]
        while not np.all(goalSequencesComplete):
            for agentIdx in range(EvalParameters.N_AGENTS):
                if goalSequencesComplete[agentIdx]:
                    continue
                agentSequence = agentsSequence[agentIdx]
                agentStart = agentSequence.getAtPos(-1)
                
                agentGoal = getFreeCell(tempMap)
                tempMap[agentGoal] = 3
                
                agentSequence.add(agentGoal)
                
                optimalAgentPath = astar_4(tempMap, agentStart, agentGoal)[0]
                pathLengths[agentIdx] += len(optimalAgentPath)-1
                if pathLengths[agentIdx] > EvalParameters.MAX_STEPS:
                    goalSequencesComplete[agentIdx] = True
            # Free the previous previous position
            for agentSequence in agentsSequence:
                tempMap[agentSequence.getAtPos(-2)] = 0
        fixedEpisodeInfos['obstacleMap'].append(obstacleMap)
        fixedEpisodeInfos['agentsSequence'].append(agentsSequence)
        fixedEpisodeInfos['humanStart'].append(humanStart)
        fixedEpisodeInfos['humanGoal'].append(humanGoal)
        fixedEpisodeInfos['numEpisodes'] += 1
    return fixedEpisodeInfos

def saveFixedEpisodeInfos(fixedEpisodeInfos):
    saveDict = copy.deepcopy(fixedEpisodeInfos)
    dir = getFixedEpisodeInfosFolder()
    if not os.path.exists(dir):
        os.mkdir(dir)
    obstacleMaps = saveDict['obstacleMap']
    agentsSequences = saveDict['agentsSequence']
    for i in range(saveDict['numEpisodes']):
        obstacleMapFile = f"obstacleMap{i}.npy"
        np.save(f"{dir}/{obstacleMapFile}", obstacleMaps[i])
        obstacleMaps[i] = obstacleMapFile
        
        agentsSequence = [sequence.items for sequence in agentsSequences[i]]
        agentsSequences[i] = agentsSequence
    save_json_dir = getJsonDir()
    with open(save_json_dir, 'w', encoding='utf-8') as f:
        print(f"Saving fixed episode infos to {save_json_dir}")
        json.dump(saveDict, f, ensure_ascii=False, indent=4, sort_keys=True)
    
def loadFixedEpisodeInfos():
    fixedEpisodeInfos = createFixedEpisodeInfo()
    with open(getJsonDir()) as f:
        json_load = json.load(f)
    for i in range(json_load['numEpisodes']):
        obstacleMapFile = json_load['obstacleMap'][i]
        obstacleMapPath = f"{getFixedEpisodeInfosFolder()}/{obstacleMapFile}"
        fixedEpisodeInfos['obstacleMap'].append(np.load(obstacleMapPath))
        fixedEpisodeInfos['agentsSequence'].append([Sequence(itemsIn=[tuple(item) for item in items]) for items in json_load['agentsSequence'][i]])
    fixedEpisodeInfos['humanStart'] = [tuple(humanStart) for humanStart in json_load['humanStart']]
    fixedEpisodeInfos['humanGoal'] = [tuple(humanGoal) for humanGoal in json_load['humanGoal']]
    return fixedEpisodeInfos

def main():
    """main code"""
    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    model = Model(0, global_device, True, numChannel=NetParameters.NUM_CHANNEL)
    if EvalParameters.LOAD_FIXED_EPISODE_INFOS:
        print("Loading fixed episode infos...")
        fixedEpisodeInfos = loadFixedEpisodeInfos()
    else:
        print("Generating new episode infos....")
        fixedEpisodeInfos = generateFixedEpisodeInfos()
        saveFixedEpisodeInfos(fixedEpisodeInfos)
    # Start evaluation
    try:
        # get data from multiple processes
        # evaluate training model
        with torch.no_grad():
            # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
            # global_device, save_gif0, curr_steps, True)
            evaluate(model, global_device, False, fixedEpisodeInfos)
    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")

    # killing
    if RecordingParameters.WANDB:
        wandb.finish()


def evaluate(model, device, greedy, fixedEpisodeInfos):
    """Evaluate Model."""
    episodePerformances = []
    all_metrics = {}
    metrics = {'hc' : [], 'ecr' : [], 'cv' : [], 'goals' : []}

    numChannel = NetParameters.NUM_CHANNEL
    for model_name, net_path_checkpoint in EvalParameters.MODELS:
        print(f"Evaluating using {model_name}!")
        net_dict = torch.load(EvalParameters.MODEL_PATH + net_path_checkpoint)
        try:
            model.network.load_state_dict(net_dict['model'])
        except Exception:
            print("Failed initializing model with # channel ", channels[curChannel], f" trying {channels[getOtherChannel()]}...")
            switchChannel()
            numChannel = channels[curChannel]
            model = Model(0, device, True, numChannel=numChannel)
        if RecordingParameters.WANDB:
            wandb_id = wandb.util.generate_id()
            run = wandb.init(project=RecordingParameters.EXPERIMENT_PROJECT,
                        name=RecordingParameters.EXPERIMENT_NAME + f"_{model_name}",
                        entity=RecordingParameters.ENTITY,
                        notes=RecordingParameters.EXPERIMENT_NOTE,
                        config=all_args,
                        id=wandb_id,
                        resume='allow')
            print('id is:{}'.format(wandb_id))
            print('Launching wandb...\n')
    
        for i in range(EvalParameters.EPISODES):
            curr_episode = i
            oneEpisodePerformance = OneEpPerformance()
            episode_frames = []
            
            obstaclesMap = fixedEpisodeInfos['obstacleMap'][i]
            agentsSequence = fixedEpisodeInfos['agentsSequence'][i]
            humanStart = fixedEpisodeInfos['humanStart'][i]
            humanGoal = fixedEpisodeInfos['humanGoal'][i]
            print(f"Episode {i}: HumStart {humanStart} HumGoal {humanGoal}")
            for id, agentSequence in enumerate(agentsSequence):
                print(f"Agent {id} seq : {agentSequence.items}")
            use_danger_area = "DA" in model_name
            use_human_pred = "HP" in model_name
            print(f"DA - {use_danger_area} HP - {use_human_pred}")
            env = FixedMapfGym(obstaclesMap, agentsSequence, humanStart, humanGoal, numChannel=numChannel, useDA=use_danger_area, useHP=use_human_pred)
            
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
                write_to_wandb_with_run(run, curr_episode, oneEpisodePerformance, evaluate=True, greedy=False)
            print('model: {},episode: {},episode reward: {}, episode cost reward: {} human_coll: {}, static_coll: {}, agent_coll: {}, total_goals: {} shadow_goals: {} \n'.format(
                    model_name,
                    curr_episode, oneEpisodePerformance.episodeReward, oneEpisodePerformance.episodeCostReward, oneEpisodePerformance.humanCollide, 
                    oneEpisodePerformance.staticCollide, oneEpisodePerformance.agentCollide, oneEpisodePerformance.totalGoals, oneEpisodePerformance.shadowGoals)) 
                
            metrics['hc'].append(oneEpisodePerformance.humanCollide)
            metrics['cv'].append(oneEpisodePerformance.constraintViolations)
            metrics['ecr'].append(oneEpisodePerformance.episodeCostReward)
            metrics['goals'].append(oneEpisodePerformance.totalGoals)
            episodePerformances.append(oneEpisodePerformance)

            if not os.path.exists(RecordingParameters.GIFS_PATH):
                os.makedirs(RecordingParameters.GIFS_PATH)
            images = np.array(episode_frames[:-1])
            make_gif(images,
                '{}/{}_episode_{:d}_reward{:.1f}_human_coll{:.1f}_totalGoals{}_shadowGoals{}_staticColl{:d}_agentColl{:d}.gif'.format(
                    RecordingParameters.GIFS_PATH,
                    model_name,
                    curr_episode, oneEpisodePerformance.episodeReward,
                    oneEpisodePerformance.humanCollide, oneEpisodePerformance.totalGoals,
                    oneEpisodePerformance.shadowGoals, oneEpisodePerformance.staticCollide, oneEpisodePerformance.agentCollide))
        # killing
        if RecordingParameters.WANDB:
            wandb.finish()
            
        for key, val in metrics.items():
            val = np.array(val)
            valMean = np.mean(val)
            valStd = np.std(val)
            meanPerAgent = valMean / EvalParameters.N_AGENTS
            stdPerAgent = valStd / EvalParameters.N_AGENTS
            meanPerAgentPerTimestep = meanPerAgent / EvalParameters.MAX_STEPS
            stdPerAgentPerTimestep = stdPerAgent / EvalParameters.MAX_STEPS
            
            all_metrics[f"{model_name}/{key}_per_agent/mean"] = meanPerAgent
            all_metrics[f"{model_name}/{key}_per_agent/std"] = stdPerAgent
            all_metrics[f"{model_name}/{key}_per_agent_per_timestep/mean"] = meanPerAgentPerTimestep
            all_metrics[f"{model_name}/{key}_per_agent_per_timestep/std"] = stdPerAgentPerTimestep
    with open(EvalParameters.METRICS_JSON_PATH, 'w') as f:
        print(f"Saving final metrics file to {EvalParameters.METRICS_JSON_PATH}...")
        json.dump(all_metrics, f, indent=4)   
    return episodePerformances

if __name__ == "__main__":
    main()

