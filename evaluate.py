import os
import os.path as osp

import numpy as np
import ray
import setproctitle
import torch
import wandb

from alg_parameters import TrainingParameters, SetupParameters, RecordingParameters, EnvParameters, all_args
from mapf_gym import FixedMapfGym
from model import Model
from util import set_global_seeds, write_to_wandb, make_gif, OneEpPerformance, NetParameters

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to MAPF!\n")

def main():
    """main code"""
    # preparing for training
    restore_path = ''
    net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
    net_dict = torch.load(net_path_checkpoint)

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
    global_model = Model(0, global_device, True)

    global_model.network.load_state_dict(net_dict['model'])
    global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    # Start evaluation
    try:
        # get data from multiple processes
        save_gif = True
        # evaluate training model
        with torch.no_grad():
            # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
            # global_device, save_gif0, curr_steps, True)
            evalPerformances = evaluate(global_model, global_device, save_gif, False)
        for curr_steps, evalPerformance in enumerate(evalPerformances):
            # record evaluation result
            if RecordingParameters.WANDB:
                # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                write_to_wandb(curr_steps, evalPerformance, evaluate=True, greedy=False)

            print('step: {},episode reward: {}, episode cost reward: {} human_coll: {}, static_coll: {}, agent_coll: {}, total_goals: {} shadow_goals: {} \n'.format(
                    curr_steps, evalPerformance.episodeReward, evalPerformance.episodeCostReward, evalPerformance.humanCollide, 
                    evalPerformance.staticCollide, evalPerformance.agentCollide, evalPerformance.totalGoals, evalPerformance.shadowGoals)) 
    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")

    # killing
    if RecordingParameters.WANDB:
        wandb.finish()


def evaluate(model, device, save_gif, greedy):
    """Evaluate Model."""
    episodePerformances = []
    episodeFixedInfos = []
    for i in range(RecordingParameters.EVAL_EPISODES):
        curr_episode = i
        
        oneEpisodePerformance = OneEpPerformance()
        episode_frames = []
        
        obstaclesMap, agentStartsList, agentGoalsList, humanStart, humanGoal = episodeFixedInfos[i]
        env = FixedMapfGym(obstaclesMap, agentStartsList, agentGoalsList, humanStart, humanGoal)
        
        obs, vecs = env.getAllObservations()

        numAgent = EnvParameters.N_AGENTS

        hidden_state = (torch.zeros((numAgent, NetParameters.NET_SIZE )).to(device),
                        torch.zeros((numAgent, NetParameters.NET_SIZE )).to(device))

        if save_gif:
            episode_frames.append(env._render())

        # stepping
        for _ in range(TrainingParameters.N_STEPS):

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

            if save_gif:
                episode_frames.append(env._render())
                
        episodePerformances.append(oneEpisodePerformance)

        # save gif
        if save_gif:
            if not os.path.exists(RecordingParameters.GIFS_PATH):
                os.makedirs(RecordingParameters.GIFS_PATH)
            images = np.array(episode_frames[:-1])
            if EnvParameters.LIFELONG:
                make_gif(images,
                    '{}/episode_{:d}_reward{:.1f}_human_coll{:.1f}_totalGoals{}_shadowGoals{}_staticColl{:d}_agentColl{:d}.gif'.format(
                        RecordingParameters.GIFS_PATH,
                        curr_episode, oneEpisodePerformance.episodeReward,
                        oneEpisodePerformance.humanCollide, oneEpisodePerformance.totalGoals,
                        oneEpisodePerformance.shadowGoals, oneEpisodePerformance.staticCollide, oneEpisodePerformance.agentCollide))
            save_gif = True
    return episodePerformances

if __name__ == "__main__":
    main()
