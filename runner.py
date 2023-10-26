import numpy as np
import ray
import torch

from alg_parameters import *
from mapf_gym import MapfGym
from model import Model
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import OutOfTimeError, NoSolutionError
from util import OneEpPerformance, BatchValues


@ray.remote(num_cpus=1, num_gpus=SetupParameters.NUM_GPU / (TrainingParameters.N_ENVS + 1))
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
                
                actions, ps, values, _, _, constraintValues = \
                    self.local_model.step(observation=obs, vector=vecs, input_state=hidden_state)
                
                ##------------------------------------------------------------------------------------------------##
                
                mb.actions.append(actions)
                mb.values.append(values)
                mb.ps.append(ps)
                mb.constraintValues.append(constraintValues)

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
                
                rewards, shadowGoals = env.calculateActionReward(actions, actionStatus)

                oneEpisodePerformance.shadowGoals+=shadowGoals

                ##------------------------------------------------------------------------------------------------##

                trainVal = env.getTrainValid(actions)

                mb.trainValid.append(trainVal)

                ##------------------------------------------------------------------------------------------------##
                goalsReached = env.jointStep(actions, actionStatus)

                for i, value in enumerate(goalsReached):
                    if(value==1):
                        rewards[0,i]+=EnvParameters.GOAL_REWARD

                mb.rewards.append(rewards)

                oneEpisodePerformance.episodeReward += np.sum(rewards)

                oneEpisodePerformance.totalGoals+=np.sum(goalsReached)
                obs, vecs = env.getAllObservations() 

                ##------------------------------------------------------------------------------------------------##

                constraintRewards = env.calculateConstraintReward()

                oneEpisodePerformance.episodeCostReward += np.sum(constraintRewards)

                mb.constraintRewards.append(constraintRewards)

                ##------------------------------------------------------------------------------------------------##

            mb.observations = np.concatenate(mb.observations, axis=0)
            mb.vectors = np.concatenate(mb.vectors, axis=0)
            mb.rewards = np.concatenate(mb.rewards, axis=0)
            mb.constraintRewards = np.concatenate(mb.constraintRewards, axis=0)
            mb.values = np.squeeze(np.concatenate(mb.values, axis=0), axis=-1)
            mb.constraintValues = np.squeeze(np.concatenate(mb.constraintValues, axis=0), axis=-1)
            mb.trainValid = np.stack(mb.trainValid)
            

            mb.actions = np.asarray(mb.actions, dtype=np.int64)
            mb.ps = np.stack(mb.ps)
            mb.hiddenState = np.stack(mb.hiddenState)

            last_values, last_constraint_values  = np.squeeze(
                self.local_model.value(obs, vecs, hidden_state))

            # calculate advantages
            mb_advs = np.zeros_like(mb.rewards)
            last_gaelam = 0
            next_nonterminal = 1.0

            mb_constraint_advs = np.zeros_like(mb.rewards)
            last_constraint_gaelam = 0
            for t in reversed(range(TrainingParameters.N_STEPS)):
                if t == TrainingParameters.N_STEPS - 1:
                    next_values = last_values
                    next_constraint_values = last_constraint_values
                else:
                    next_values= mb.values[t + 1]
                    next_constraint_values = mb.constraintValues[t+1]

                delta = np.subtract(np.add(mb.rewards[t], TrainingParameters.GAMMA * next_nonterminal *
                                              next_values), mb.values[t])

                mb_advs[t] = last_gaelam = np.add(delta,
                                                        TrainingParameters.GAMMA * TrainingParameters.LAM
                                                        * next_nonterminal * last_gaelam)
                
                constraint_delta = np.subtract(np.add(mb.constraintRewards[t], TrainingParameters.GAMMA * next_nonterminal *
                                        next_constraint_values), mb.constraintValues[t])
                
                mb_constraint_advs[t] = last_constraint_gaelam = np.add(constraint_delta, TrainingParameters.GAMMA * TrainingParameters.LAM
                                        * next_nonterminal * last_constraint_gaelam)

            mb.returns = np.add(mb_advs, mb.values)
            mb.constraintReturns = np.add(mb_constraint_advs, mb.constraintValues)

        return mb, oneEpisodePerformance