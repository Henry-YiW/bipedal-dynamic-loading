import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rsl_rl.modules import ActorCritic
from rsl_rl.storage import RolloutStorage


class GRPO:
    actor_critic: ActorCritic

    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.01,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 alpha=1.0,
                 regularization_type='kl',
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 num_proprio_encoder_substeps=1):

        self.device = device

        self.actor_critic = actor_critic.to(self.device)
        self.transition = RolloutStorage.Transition()
        self.storage = None
        self.optimizer = optim.Adam(self.actor_critic.parameters(),
            lr=self.learning_rate)

        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm

        self.alpha = alpha
        self.regularization_type = regularization_type
        self.schedule = schedule
        self.desired_kl = desired_kl

        self.num_proprio_encoder_substeps = num_proprio_encoder_substeps

        

        

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

        
    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, obs_hisotry, load_obs, critic_obs):
        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()

        self.transition.actions = self.actor_critic.act(obs, obs_hisotry, critic_obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs, load_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        self.transition.observations = obs.clone()
        self.transition.observations_history = obs_hisotry.clone()
        self.transition.load_observations = load_obs.clone()
        self.transition.critic_observations = critic_obs.clone()
        return self.transition.actions

    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def compute_returns(self, last_critic_obs, load_obs):
        last_values = self.actor_critic.evaluate(last_critic_obs, load_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        mean_value_loss = 0
        mean_grpo_loss = 0
        mean_load_esti_loss = 0

        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(
                self.num_mini_batches, self.num_learning_epochs)

        for (obs_batch, obs_history_batch, obs_load_batch,
             critic_obs_batch, actions_batch, target_values_batch,
             advantages_batch, returns_batch, old_actions_log_prob_batch,
             old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch) in generator:

            self.actor_critic.act(obs_batch, obs_history_batch, critic_obs_batch,
                                  masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
            value_batch = self.actor_critic.evaluate(critic_obs_batch, obs_load_batch,
                                                     masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.actor_critic.action_mean
            sigma_batch = self.actor_critic.action_std
            entropy_batch = self.actor_critic.entropy

            log_ratio = actions_log_prob_batch - old_actions_log_prob_batch.squeeze()

            if self.regularization_type == 'kl':
                reg = log_ratio.exp() * log_ratio
            elif self.regularization_type == 'reverse_kl':
                reg = -log_ratio
            elif self.regularization_type == 'chi2':
                ratio = log_ratio.exp()
                reg = (ratio - 1) ** 2
            else:
                raise NotImplementedError(f"Unsupported regularization type: {self.regularization_type}")

            grpo_loss = -(advantages_batch.squeeze() * log_ratio - self.alpha * reg).mean()

            value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = grpo_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_grpo_loss += grpo_loss.item()
            mean_value_loss += value_loss.item()

            for _ in range(self.num_proprio_encoder_substeps):
                load_state_estimation = self.actor_critic.load_state_estimator(obs_history_batch)
                load_esti_loss = F.mse_loss(load_state_estimation, obs_load_batch)
                self.load_esti_optimizer.zero_grad()
                load_esti_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.load_state_estimator.parameters(), self.max_grad_norm)
                self.load_esti_optimizer.step()
                mean_load_esti_loss += load_esti_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_grpo_loss /= num_updates
        mean_value_loss /= num_updates
        mean_load_esti_loss /= (num_updates * self.num_proprio_encoder_substeps)

        self.storage.clear()
        return mean_value_loss, mean_grpo_loss, mean_load_esti_loss
