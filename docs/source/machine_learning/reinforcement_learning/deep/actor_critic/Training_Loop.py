import dataclasses
import itertools

import gym
import numpy as np
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from docs.source.machine_learning.reinforcement_learning.deep.actor_critic.SAC import SAC
from docs.source.utils.WandB import WandBLogger


@dataclasses.dataclass
class args(object):
    gamma                   = 0.99
    tau                     = 0.005
    alpha                   = 0.2
    lr                      = 0.0003
    policy                  = "Gaussian"
    target_update_interval  = 1
    replay_size             = 1000000
    automatic_entropy_tuning= False
    cuda                    = True
    hidden_size             = 256
    start_steps             = 10000
    batch_size              = 256
    num_steps               = 1000000
    eval                    = True          # If we evaluate the model to get actual performance of model
    env_name                = 'Hopper-v4'




env = gym.make(args.env_name)
agent = SAC(env.observation_space.shape[0], env.action_space, args)
memory:ReplayBuffer = ReplayBuffer(
    buffer_size=args.replay_size,
    observation_space=env.observation_space,
    action_space=env.action_space,
    device=torch.device("cuda")
)
wandb_logger = WandBLogger()

# Training loop
total_num_steps = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_num_steps:
            action = env.action_space.sample() # random action
        else:
            action = agent.select_action(state) # sample action from policy

        if memory.size() > args.batch_size:
            # update parameters of all networks
            critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, total_num_steps)
            wandb_logger.log('loss/critic_1', critic_1_loss, total_num_steps)
            wandb_logger.log('loss/critic_1', critic_1_loss, total_num_steps)
            wandb_logger.log('loss/critic_2', critic_2_loss, total_num_steps)
            wandb_logger.log('loss/policy', policy_loss, total_num_steps)
            wandb_logger.log('loss/entropy_loss', ent_loss, total_num_steps)
            wandb_logger.log('entropy_temprature/alpha', alpha, total_num_steps)

        next_state, reward, done, info = env.step(action)
        episode_steps += 1
        total_num_steps += 1
        episode_reward += reward

        # Ignore done signal if it comes from hitting the time horizon
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)

        memory.add(np.array([state]).astype('float32'),
                   np.array([next_state]).astype('float32'),
                   np.array([action]).astype('float32'),
                   np.array([reward]).astype('float32'),
                   np.array([mask]).astype('float32'),
                   np.array([info]))

        state = next_state

    if i_episode % 10 == 0:
        agent.save_checkpoint(env_name=args.env_name, ckpt_path="/home/jack/PHD/Blog/docs/source/machine_learning/reinforcement_learning/deep/actor_critic/checkpoints/sac.pth")

    if total_num_steps > args.num_steps:
        break

    wandb_logger.log('reward/train', episode_reward, total_num_steps)
    print(f"episode: {i_episode} steps: {total_num_steps} episode_steps: {episode_steps} episode_reward: {episode_reward}")

    if i_episode % 10 == 0 and args.eval is True:

        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):

            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(torch.Tensor(state).float(), evaluate=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes

        wandb_logger.log('avg_reward/test', avg_reward, total_num_steps)
        print("----------------------------------------")
        print(f"Test Episodes: {episodes}, Avg. Reward: {round(avg_reward, 2)}")
        print("----------------------------------------")


