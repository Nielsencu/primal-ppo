import os
import os.path as osp

import numpy as np
import ray
import setproctitle
import torch
import wandb

from alg_parameters import *
from mapf_gym import MapfGym
from model import Model
from runner import Runner
from util import set_global_seeds, write_to_wandb, make_gif, OneEpPerformance, BatchValues

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
ray.init(num_gpus=SetupParameters.NUM_GPU)
print("Welcome to MAPF!\n")


def main():
    """main code"""
    # preparing for training
    if RecordingParameters.RETRAIN:
        restore_path = ''
        net_path_checkpoint = restore_path + "/net_checkpoint.pkl"
        net_dict = torch.load(net_path_checkpoint)

    if RecordingParameters.WANDB:
        if RecordingParameters.RETRAIN:
            wandb_id = None
        else:
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
    local_device = torch.device('cuda') if SetupParameters.USE_GPU_LOCAL else torch.device('cpu')
    global_model = Model(0, global_device, True)

    if RecordingParameters.RETRAIN:
        global_model.network.load_state_dict(net_dict['model'])
        global_model.net_optimizer.load_state_dict(net_dict['optimizer'])

    envs = [Runner.remote(i + 1) for i in range(TrainingParameters.N_ENVS)]

    if RecordingParameters.RETRAIN:
        curr_steps = net_dict["step"]
        curr_episodes = net_dict["episode"]
        best_perf = net_dict["reward"]
    else:
        curr_steps = curr_episodes = best_perf = 0

    update_done = True
    demon = True
    job_list = []
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
                net_weights_id = ray.put(net_weights)

                
                for i, env in enumerate(envs):
                    job_list.append(env.run.remote(net_weights_id))

            # get data from multiple processes
            done_id, job_list = ray.wait(job_list, num_returns=TrainingParameters.N_ENVS)
            update_done = True if job_list == [] else False
            done_len = len(done_id)
            job_results = ray.get(done_id)
 
 
            # get reinforcement learning data
            curr_steps += done_len * TrainingParameters.N_STEPS
            mb = BatchValues()
            performance = OneEpPerformance()
            for results in range(done_len):
                
                for value in dir(BatchValues()):
                    if not value.startswith('__'):
                        temp = getattr(mb, value)
                        temp.append(getattr(job_results[results][0], value))
                        setattr(mb, value, temp)

                curr_episodes += 1
                for i in dir(performance):
                    if not i.startswith('__'):
                        # performance_dict[i].append(np.nanmean(job_results[results][-1][i]))
                        setattr(performance, i, np.nanmean(getattr(job_results[results][-1], i)))

            for i in dir(performance):
                if not i.startswith('__'):
                    setattr(performance, i, np.nanmean(getattr(performance, i)))

            for value in dir(BatchValues()):
                    if not value.startswith('__'):
                        setattr(mb, value, np.concatenate(getattr(mb, value), axis=0))

            # training of reinforcement learning
            mb_loss = []
            inds = np.arange(TrainingParameters.N_STEPS)
            for _ in range(TrainingParameters.N_EPOCHS):
                np.random.shuffle(inds)
                for start in range(0, TrainingParameters.N_STEPS, TrainingParameters.MINIBATCH_SIZE):
                    end = start + TrainingParameters.MINIBATCH_SIZE
                    mb_inds = inds[start:end]
                    mb_loss.append(global_model.train(mb.observations[mb_inds], mb.vectors[mb_inds], mb.returns[mb_inds], \
                                                    mb.values[mb_inds], mb.actions[mb_inds], mb.ps[mb_inds], \
                                                    mb.hiddenState[mb_inds], mb.trainValid[mb_inds]))

            # record training result
            if RecordingParameters.WANDB:
                write_to_wandb(curr_steps, performance, mb_loss, evaluate=False)

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
                # record evaluation result
                if RecordingParameters.WANDB:
                    # write_to_wandb(curr_steps, greedy_eval_performance_dict, evaluate=True, greedy=True)
                    write_to_wandb(curr_steps, evalPerformance, evaluate=True, greedy=False)

                print('episodes: {}, step: {},episode reward: {}, human_coll: {}, static_coll: {}, agent_coll: {}, total_goals: {} shadow_goals: {} \n'.format(
                        curr_episodes, curr_steps, evalPerformance.episodeReward, evalPerformance.humanCollide, 
                        evalPerformance.staticCollide, evalPerformance.agentCollide, evalPerformance.totalGoals, evalPerformance.shadowGoals)) 
                # save model with the best performance
                if RecordingParameters.RECORD_BEST:
                    if evalPerformance.episodeReward > best_perf and (
                            curr_steps - last_best_t) / RecordingParameters.BEST_INTERVAL >= 1.0:
                        best_perf = evalPerformance.episodeReward
                        last_best_t = curr_steps
                        print('Saving best model \n')
                        model_path = osp.join(RecordingParameters.MODEL_PATH, 'best_model')
                        if not os.path.exists(model_path):
                            os.makedirs(model_path)
                        path_checkpoint = model_path + "/net_checkpoint.pkl"
                        net_checkpoint = {"model": global_model.network.state_dict(),
                                          "optimizer": global_model.net_optimizer.state_dict(),
                                          "step": curr_steps,
                                          "episode": curr_episodes,
                                          "reward": best_perf}
                        torch.save(net_checkpoint, path_checkpoint)

            # save model
            if (curr_steps - last_model_t) / RecordingParameters.SAVE_INTERVAL >= 1.0:
                last_model_t = curr_steps
                print('Saving Model !\n')
                model_path = osp.join(RecordingParameters.MODEL_PATH, '%.5i' % curr_steps)
                os.makedirs(model_path)
                path_checkpoint = model_path + "/net_checkpoint.pkl"
                net_checkpoint = {"model": global_model.network.state_dict(),
                                  "optimizer": global_model.net_optimizer.state_dict(),
                                  "step": curr_steps,
                                  "episode": curr_episodes,
                                  "reward": evalPerformance.episodeReward}
                torch.save(net_checkpoint, path_checkpoint)

    except KeyboardInterrupt:
        print("CTRL-C pressed. killing remote workers")

    # save final model
    print('Saving Final Model !\n')
    model_path = RecordingParameters.MODEL_PATH + '/final'
    os.makedirs(model_path)
    path_checkpoint = model_path + "/net_checkpoint.pkl"
    net_checkpoint = {"model": global_model.network.state_dict(),
                      "optimizer": global_model.net_optimizer.state_dict(),
                      "step": curr_steps,
                      "episode": curr_episodes,
                       "reward": evalPerformance.episodeReward}
    torch.save(net_checkpoint, path_checkpoint)

    # killing
    for e in envs:
        ray.kill(e)
    if RecordingParameters.WANDB:
        wandb.finish()


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


            rewards,shadowGoals = env.calculateActionReward(actions, actionStatus)

            oneEpisodePerformance.shadowGoals+=shadowGoals  
        
            goalsReached = env.jointStep(actions, actionStatus)

            for i, value in enumerate(goalsReached):
                if(value==1):
                    rewards[0,i]+=EnvParameters.GOAL_REWARD


            oneEpisodePerformance.episodeReward += np.sum(rewards)
            
            oneEpisodePerformance.totalGoals+=np.sum(goalsReached)

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
                '{}/steps_{:d}_reward{:.1f}_human_coll{:.1f}_totalGoals{}_shadowGoals{}_staticColl{:d}_agentColl{:d}.gif'.format(
                    RecordingParameters.GIFS_PATH,
                    curr_steps, oneEpisodePerformance.episodeReward,
                    oneEpisodePerformance.humanCollide, oneEpisodePerformance.totalGoals,
                    oneEpisodePerformance.shadowGoals, oneEpisodePerformance.staticCollide, oneEpisodePerformance.agentCollide))
        save_gif = True

    return oneEpisodePerformance



if __name__ == "__main__":
    main()

