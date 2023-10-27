import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from alg_parameters import *
from net import SCRIMPNet
import torch.nn as nn


class Model(object):
    """model0 of agents"""

    def __init__(self, env_id, device, global_model=False):
        """initialization"""
        self.ID = env_id
        self.device = device
        self.network = SCRIMPNet().to(device)  # neural network
        if global_model:
            self.net_optimizer = optim.Adam(self.network.parameters(), lr=TrainingParameters.lr)
            self.lagrangian_param = torch.tensor(1.0, requires_grad=True).float()
            self.lagrangian_optimizer = optim.Adam([self.lagrangian_param], lr=TrainingParameters.LAGRANGIAN_LR)
            self.net_scaler = GradScaler()  # automatic mixed precision

    def step(self, observation = np.zeros(1), vector = np.zeros(1), input_state =  torch.zeros(1), num_agent = EnvParameters.N_AGENTS):
        """using neural network in training for prediction"""
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v, block, _, output_state, _, cv = self.network(observation, vector, input_state)

        actions = np.zeros(num_agent)
        ps = np.squeeze(ps.cpu().detach().numpy())
        v = v.cpu().detach().numpy()  # intrinsic state values
        block = np.squeeze(block.cpu().detach().numpy())
        cv = cv.cpu().detach().numpy()

        for i in range(num_agent):
            # choose action from complete action distribution
            actions[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        return actions, ps, v, block, output_state, cv

    def evaluate(self, observation, vector, input_state, greedy, num_agent):
        """using neural network in evaluations of training code for prediction"""
        eval_action = np.zeros(num_agent)
        observation = torch.from_numpy(np.asarray(observation)).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        ps, v, block, _, output_state, _, cv = self.network(observation, vector, input_state,)

        ps = np.squeeze(ps.cpu().detach().numpy())
        block = np.squeeze(block.cpu().detach().numpy())
        greedy_action = np.argmax(ps, axis=-1)
        v = v.cpu().detach().numpy()

        for i in range(num_agent):
            if not greedy:
                eval_action[i] = np.random.choice(range(EnvParameters.N_ACTIONS), p=ps[i].ravel())
        if greedy:
            eval_action = greedy_action
        return eval_action, block, output_state, v, ps

    def value(self, obs, vector, input_state):
        """using neural network to predict state values"""
        obs = torch.from_numpy(obs).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        _, v, _, _, _, _, cv = self.network(obs, vector, input_state)
        v = v.cpu().detach().numpy()
        cv = cv.cpu().detach().numpy()
        return v, cv

    def generate_state(self, obs, vector, input_state):
        """generate corresponding hidden states and messages in imitation learning"""
        obs = torch.from_numpy(obs).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        _, _, _, _, output_state, _, _ = self.network(obs, vector, input_state)
        return output_state

    def train(self, observation, vector, returns, constraint_returns, old_v, old_cv, action,
              old_ps, input_state, train_valid, episode_cost):
        """train model0 by reinforcement learning"""
        self.net_optimizer.zero_grad()
        # from numpy to torch
        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)

        returns = torch.from_numpy(returns).to(self.device)

        old_v = torch.from_numpy(old_v).to(self.device)

        action = torch.from_numpy(action).to(self.device)
        action = torch.unsqueeze(action, -1)
        old_ps = torch.from_numpy(old_ps).to(self.device)

        train_valid = torch.from_numpy(train_valid).to(self.device)
        # target_blockings = torch.from_numpy(target_blockings).to(self.device)

        constraint_returns = torch.from_numpy(constraint_returns).to(self.device)
        old_cv = torch.from_numpy(old_cv).to(self.device)

        input_state_h = torch.from_numpy(
            np.reshape(input_state[:, 0], (-1, NetParameters.NET_SIZE))).to(self.device)
        input_state_c = torch.from_numpy(
            np.reshape(input_state[:, 1], (-1, NetParameters.NET_SIZE))).to(self.device)
        input_state = (input_state_h, input_state_c)

        normalize_advantage = lambda x : (x-x.mean()) / (x.std() + 1e-6)
        advantage = normalize_advantage(returns - old_v)

        #dp_network = nn.DataParallel(self.network)
        cost_advantage = normalize_advantage(constraint_returns - old_cv)
        with autocast():
            new_ps, new_v, block, policy_sig, _, _, new_cv = self.network(observation, vector, input_state)
            new_p = new_ps.gather(-1, action)
            old_p = old_ps.gather(-1, action)
            ratio = torch.exp(torch.log(torch.clamp(new_p, 1e-6, 1.0)) - torch.log(torch.clamp(old_p, 1e-6, 1.0)))

            entropy = torch.mean(-torch.sum(new_ps * torch.log(torch.clamp(new_ps, 1e-6, 1.0)), dim=-1, keepdim=True))

            # critic loss
            new_v = torch.squeeze(new_v)
            new_v_clipped = old_v+ torch.clamp(new_v - old_v, - TrainingParameters.CLIP_RANGE,
                                               TrainingParameters.CLIP_RANGE)
            value_losses1 = torch.square(new_v - returns)
            value_losses2= torch.square(new_v_clipped - returns)
            critic_loss = torch.mean(torch.maximum(value_losses1, value_losses2))

            new_cv = torch.squeeze(new_cv)
            new_cv_clipped = old_cv+ torch.clamp(new_cv - old_cv, - TrainingParameters.CLIP_RANGE,
                                               TrainingParameters.CLIP_RANGE)
            value_losses1 = torch.square(new_cv - constraint_returns)
            value_losses2= torch.square(new_cv_clipped - constraint_returns)
            constraint_critic_loss = torch.mean(torch.maximum(value_losses1, value_losses2))

            # actor loss
            ratio = torch.squeeze(ratio)
            policy_losses = advantage * ratio
            policy_losses2 = advantage * torch.clamp(ratio, 1.0 - TrainingParameters.CLIP_RANGE,
                                                     1.0 + TrainingParameters.CLIP_RANGE)
            policy_loss = torch.mean(torch.min(policy_losses, policy_losses2))

            # valid loss and blocking loss decreased by supervised learning
            valid_loss = - torch.mean(torch.log(torch.clamp(policy_sig, 1e-6, 1.0 - 1e-6)) *
                                      train_valid + torch.log(torch.clamp(1 - policy_sig, 1e-6, 1.0 - 1e-6)) * (
                                              1 - train_valid))
            block = torch.squeeze(block)
            # blocking_loss = - torch.mean(target_blockings * torch.log(torch.clamp(block, 1e-6, 1.0 - 1e-6))
            #                              + (1 - target_blockings) * torch.log(torch.clamp(1 - block, 1e-6, 1.0 - 1e-6)))

            # penalty loss
            cost_loss = torch.mean(ratio * cost_advantage)
            penalty = 0.0
            if TrainingParameters.COST_COEF > 0.0:
                penalty = F.softplus(self.lagrangian_param).item()

            # total loss
            all_loss = -policy_loss - entropy * TrainingParameters.ENTROPY_COEF + \
                TrainingParameters.VALUE_COEF * critic_loss  \
                + TrainingParameters.VALID_COEF * valid_loss \
                + TrainingParameters.COST_VALUE_COEF * constraint_critic_loss \
                + TrainingParameters.COST_COEF * penalty * cost_loss \
                # + TrainingParameters.BLOCK_COEF * blocking_loss \ 
                
            all_loss /= (1+penalty)

        clip_frac = torch.mean(torch.greater(torch.abs(ratio - 1.0), TrainingParameters.CLIP_RANGE).float())

        self.net_scaler.scale(all_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)

        cost_deviation = (episode_cost - TrainingParameters.COST_LIMIT)
        loss_penalty = -self.lagrangian_param * cost_deviation

        self.lagrangian_optimizer.zero_grad()
        loss_penalty.backward()
        self.lagrangian_optimizer.step()
        print("Lagrangian param", self.lagrangian_param.item())

        # Clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)

        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        stats_list = [all_loss.cpu().detach().numpy(), policy_loss.cpu().detach().numpy(),
                      entropy.cpu().detach().numpy(),
                      critic_loss.cpu().detach().numpy(),
                      valid_loss.cpu().detach().numpy(),
                      constraint_critic_loss.cpu().detach().numpy(),
                      cost_loss.cpu().detach().numpy(),
                    #   blocking_loss.cpu().detach().numpy(),
                      clip_frac.cpu().detach().numpy(), grad_norm.cpu().detach().numpy(),
                      torch.mean(advantage).cpu().detach().numpy()]  # for recording

        return stats_list

    def set_weights(self, weights):
        """load global weights to local models"""
        self.network.load_state_dict(weights)

    def imitation_train(self, observation, vector, optimal_action, input_state):
        """train model0 by imitation learning"""
        self.net_optimizer.zero_grad()

        observation = torch.from_numpy(observation).to(self.device)
        vector = torch.from_numpy(vector).to(self.device)
        optimal_action = torch.from_numpy(optimal_action).to(self.device)
        input_state_h = torch.from_numpy(
            np.reshape(input_state[:, 0], (-1, NetParameters.NET_SIZE ))).to(self.device)
        input_state_c = torch.from_numpy(
            np.reshape(input_state[:, 1], (-1, NetParameters.NET_SIZE ))).to(self.device)

        input_state = (input_state_h, input_state_c)

        with autocast():
            _, _,  _, _, _, logits, _ = self.network(observation, vector, input_state)
            logits = torch.swapaxes(logits, 1, 2)
            imitation_loss = F.cross_entropy(logits, optimal_action)

        self.net_scaler.scale(imitation_loss).backward()
        self.net_scaler.unscale_(self.net_optimizer)
        # clip gradient
        grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), TrainingParameters.MAX_GRAD_NORM)
        self.net_scaler.step(self.net_optimizer)
        self.net_scaler.update()

        return [imitation_loss.cpu().detach().numpy(), grad_norm.cpu().detach().numpy()]  # for recording
