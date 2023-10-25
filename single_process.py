import os
import os.path as osp

import numpy as np
import setproctitle
import torch
from alg_parameters import *
from mapf_gym import MapfGym
from model import Model
from util import set_global_seeds, make_gif, OneEpPerformance, BatchValues

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Runner(object):
    """sub-process used to collect experience"""

    def __init__(self, env_id):
        """Initialize model and environment"""
        self.ID = env_id
        self.numAgent = EnvParameters.N_AGENTS

        self.local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
        self.local_model = Model(env_id, self.local_device)


    def run(self, weights):
        """run multiple steps and collect data for reinforcement learning"""
        with torch.no_grad():
             
            env = MapfGym()
            obs, vecs = env.getAllObservations()

            hidden_state = (
                torch.zeros((self.numAgent, NetParameters.NET_SIZE )).to(self.local_device),
                torch.zeros((self.numAgent, NetParameters.NET_SIZE )).to(self.local_device))

            mb = BatchValues() #Mini Batch values that will be used for training

            oneEpisodePerformance = OneEpPerformance()
           
            self.local_model.set_weights(weights)
            
            for _ in range(TrainingParameters.N_STEPS):
                ##------------------------------------------------------------------------------------------------##
                mb.observations.append(obs)
                mb.vectors.append(vecs)
                mb.hiddenState.append(
                    [hidden_state[0].cpu().detach().numpy(), hidden_state[1].cpu().detach().numpy()])

                ##------------------------------------------------------------------------------------------------##
                
                actions, ps, values, _, _ = \
                    self.local_model.step(observation=obs, vector=vecs, input_state=hidden_state)
                
                ##------------------------------------------------------------------------------------------------##
                
                mb.actions.append(actions)
                mb.values.append(values)
                mb.ps.append(ps)

                ##------------------------------------------------------------------------------------------------##
                
                actionStatus = env.getActionStatus(actions)
                
                for  value in actionStatus:
                    if value==-1:
                        oneEpisodePerformance.staticCollide+=1
                    elif value==-2:
                        oneEpisodePerformance.humanCollide +=1
                    elif value==-3:
                        oneEpisodePerformance.agentCollide+=1

                ##------------------------------------------------------------------------------------------------##
                
                rewards = env.calculateActionReward(actionStatus)

                mb.rewards.append(rewards)

                oneEpisodePerformance.episodeReward += np.sum(rewards)

                ##------------------------------------------------------------------------------------------------##


                trainVal = env.getTrainValid(actions)

                mb.trainValid.append(trainVal)

                ##------------------------------------------------------------------------------------------------##

                goalsReached = env.jointStep(actions, actionStatus)

                oneEpisodePerformance.totalGoals+=goalsReached

                obs, vecs = env.getAllObservations() 

                ##------------------------------------------------------------------------------------------------##

            mb.observations = np.concatenate(mb.observations, axis=0)
            mb.vectors = np.concatenate(mb.vectors, axis=0)
            mb.rewards = np.concatenate(mb.rewards, axis=0)
            mb.values = np.squeeze(np.concatenate(mb.values, axis=0), axis=-1)
            mb.trainValid = np.stack(mb.trainValid)


            mb.actions = np.asarray(mb.actions, dtype=np.int64)
            mb.ps = np.stack(mb.ps)
            mb.hiddenState = np.stack(mb.hiddenState)

            last_values  = np.squeeze(
                self.local_model.value(obs, vecs, hidden_state))

            # calculate advantages
            mb_advs = np.zeros_like(mb.rewards)
            last_gaelam = 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_nonterminal = 1.0
                    next_values = last_values
                else:
                    next_nonterminal = 1.0 
                    next_values= mb.values[t + 1]

                delta = np.subtract(np.add(mb.rewards[t], TrainingParameters.GAMMA * next_nonterminal *
                                              next_values), mb.values[t])

                mb_advs[t] = last_gaelam = np.add(delta,
                                                        TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)

            mb.returns = np.add(mb_advs, mb.values)

        return mb, oneEpisodePerformance


def main():
    """Main code."""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = ''
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    setproctitle.setproctitle(
        RecordingParameters.EXPERIMENT_PROJECT + RecordingParameters.EXPERIMENT_NAME + "@" + RecordingParameters.ENTITY)
    set_global_seeds(SetupParameters.SEED)

    # create classes
    global_device = torch.device('cuda') if SetupParameters.USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = Runner(1)

    curr_steps = curr_episodes = best_perf = 0

    update_done = True
    last_test_t = -RecordingParameters.EVAL_INTERVAL - 1
    last_model_t = -RecordingParameters.SAVE_INTERVAL - 1
    last_best_t = -RecordingParameters.BEST_INTERVAL - 1
    last_gif_t = -RecordingParameters.GIF_INTERVAL - 1

    # start training
    try:
        while curr_steps < TrainingParameters.N_MAX_STEPS:
            if update_done:
                # start a data collection
                if global_device != local_device:
                    net_weights = global_model.network.to(local_device).state_dict()
                    global_model.network.to(global_device)
                else:
                    net_weights = global_model.network.state_dict()
                mbValues, oneEpPerformance = envs.run(net_weights)
                curr_steps += TrainingParameters.N_STEPS
                curr_episodes += 1
            

            # training of reinforcement learning
            mb_loss = []
            inds = np.arange(TrainingParameters.N_STEPS)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    mb_loss.append(global_model.train(mbValues.observations[mb_inds], mbValues.vectors[mb_inds], mbValues.returns[mb_inds], \
                                                    mbValues.values[mb_inds], mbValues.actions[mb_inds], mbValues.ps[mb_inds], \
                                                    mbValues.hiddenState[mb_inds]))

            # If eval
            if (curr_steps - last_test_t) / RecordingParameters.EVAL_INTERVAL >= 1.0:
                # if save gif
                if (curr_steps - last_gif_t) / RecordingParameters.GIF_INTERVAL >= 1.0:
                    save_gif = True
                    last_gif_t = curr_steps
                else:
                    save_gif = False

                # evaluate training model
                last_test_t = curr_steps
                with torch.no_grad():
                    # greedy_eval_performance_dict = evaluate(eval_env,eval_memory, global_model,
                    # global_device, save_gif0, curr_steps, True)
                    evalPerformance = evaluate(global_model, global_device, save_gif,
                                                     curr_steps, False)
                if EnvParameters.LIFELONG:
                    print('episodes: {}, step: {},episode reward: {}, human_coll: {} \n'.format(
                        curr_episodes, curr_steps, evalPerformance.episodeReward,
                        evalPerformance.humanCollide))   

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")


def evaluate(model, device, save_gif, curr_steps, greedy):
    """Evaluate Model."""
    oneEpisodePerformance = OneEpPerformance()
    episode_frames = []

    for i in range(RecordingParameters.EVAL_EPISODES):

        env = MapfGym()
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


            rewards = env.calculateActionReward(actionStatus)

            oneEpisodePerformance.episodeReward += np.sum(rewards)


            goalsReached = env.jointStep(actions, actionStatus)

            oneEpisodePerformance.totalGoals+=goalsReached

            obs, vecs = env.getAllObservations() 

            if save_gif:
                episode_frames.append(env._render())

    # save gif
    if save_gif:
        if not os.path.exists(RecordingParameters.GIFS_PATH):
            os.makedirs(RecordingParameters.GIFS_PATH)
        images = np.array(episode_frames[:-1])
        if EnvParameters.LIFELONG:
            make_gif(images,
                '{}/steps_{:d}_reward{:.1f}_human_coll{:.1f}_greedy{:d}.gif'.format(
                    RecordingParameters.GIFS_PATH,
                    curr_steps, oneEpisodePerformance.episodeReward,
                    oneEpisodePerformance.humanCollide, greedy))
        save_gif = True

    return oneEpisodePerformance


if __name__ == "__main__":
    main()