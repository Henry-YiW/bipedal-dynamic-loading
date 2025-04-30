import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import torch

# Path to your log directory
log_dir = "/home/haoran/robotics/limx_rl/pointfoot-legged-gym/logs/pointfoot_rough/Mar26_18-16-45_"

def plot_training_metrics(log_dir):
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0
        }
    )
    ea.Reload()
    
    # Print available tags first
    print("Available scalar tags:", ea.Tags()['scalars'])
    
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 20))
    
    # Plot reward
    if 'Train/mean_reward' in ea.Tags()['scalars']:
        reward_data = ea.Scalars('Train/mean_reward')
        steps = np.array([x.step for x in reward_data])
        rewards = np.array([x.value for x in reward_data])
        
        # Clip rewards at -200
        clipped_rewards = np.clip(rewards, -200, None)
        
        # Calculate smoothed rewards
        window = 50
        smoothed_rewards = np.convolve(clipped_rewards, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        
        # Plot
        ax1.plot(steps, clipped_rewards, 'b-', alpha=0.3, label='Raw Rewards')
        ax1.plot(smoothed_steps, smoothed_rewards, 'r-', linewidth=2, label=f'Smoothed (window={window})')
        
        ax1.set_title('Training Reward')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Reward')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Set y-axis limits to show the important range
        ax1.set_ylim([-250, max(clipped_rewards) + 50])  # Add some padding
        
        # Print statistics
        print("\nReward statistics:")
        print(f"Mean: {np.mean(clipped_rewards):.3f}")
        print(f"Recent mean (last 10%): {np.mean(clipped_rewards[-len(clipped_rewards)//10:]):.3f}")
        print(f"Max: {np.max(clipped_rewards):.3f}")
        print(f"Min: {np.min(clipped_rewards):.3f}")
        print(f"Percentage of rewards > -200: {np.mean(rewards > -200):.2%}")
    
    # Plot episode length with similar smoothing
    if 'Train/mean_episode_length' in ea.Tags()['scalars']:
        length_data = ea.Scalars('Train/mean_episode_length')
        steps = np.array([x.step for x in length_data])
        lengths = np.array([x.value for x in length_data])
        
        # Calculate smoothed episode lengths
        smoothed_lengths = np.convolve(lengths, np.ones(window)/window, mode='valid')
        smoothed_steps = steps[window-1:]
        
        ax2.plot(steps, lengths, 'g-', alpha=0.3, label='Raw Episode Length')
        ax2.plot(smoothed_steps, smoothed_lengths, 'r-', linewidth=2, label=f'Smoothed (window={window})')
        ax2.set_title('Episode Length')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Length')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Print episode length statistics
        print("\nEpisode Length statistics:")
        print(f"Mean: {np.mean(lengths):.3f}")
        print(f"Recent mean (last 10%): {np.mean(lengths[-len(lengths)//10:]):.3f}")
        print(f"Max: {np.max(lengths):.3f}")
        print(f"Min: {np.min(lengths):.3f}")
    
    # Plot value loss
    if 'Loss/value_function' in ea.Tags()['scalars']:
        value_loss_data = ea.Scalars('Loss/value_function')
        steps_v = np.array([x.step for x in value_loss_data])
        value_losses = np.array([x.value for x in value_loss_data]) / 10.0
        value_losses = np.clip(value_losses, None, np.percentile(value_losses, 95))
        
        window = 50
        smoothed_value_losses = np.convolve(value_losses, np.ones(window)/window, mode='valid')
        smoothed_steps_v = steps_v[window-1:]
        
        ax3.plot(steps_v, value_losses, 'b-', alpha=0.3, label='Value Loss')
        ax3.plot(smoothed_steps_v, smoothed_value_losses, 'r-', linewidth=2, label='Smoothed Value Loss')
        ax3.set_title('Value Loss')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Loss')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        print("\nValue Loss statistics:")
        print(f"  Mean: {np.mean(value_losses):.3f}")
        print(f"  Recent mean (last 10%): {np.mean(value_losses[-len(value_losses)//10:]):.3f}")
        print(f"  Max: {np.max(value_losses):.3f}")
        print(f"  Min: {np.min(value_losses):.3f}")
    
    # Plot surrogate loss with adjusted scale
    if 'Loss/surrogate' in ea.Tags()['scalars']:
        surrogate_loss_data = ea.Scalars('Loss/surrogate')
        steps_s = np.array([x.step for x in surrogate_loss_data])
        surrogate_losses = np.array([x.value for x in surrogate_loss_data])
        
        # Scale up the surrogate losses by 1000 to make them visible
        surrogate_losses = surrogate_losses * 1000  # Convert from ~0.001 to ~1.0 scale
        
        window = 50
        smoothed_surrogate_losses = np.convolve(surrogate_losses, np.ones(window)/window, mode='valid')
        smoothed_steps_s = steps_s[window-1:]
        
        ax4.plot(steps_s, surrogate_losses, 'g-', alpha=0.3, label='Surrogate Loss (×1000)')
        ax4.plot(smoothed_steps_s, smoothed_surrogate_losses, 'y-', linewidth=2, label='Smoothed Surrogate Loss (×1000)')
        ax4.set_title('Surrogate Loss (Scaled ×1000)')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Loss (×1000)')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        print("\nSurrogate Loss statistics (original scale):")
        print(f"  Mean: {np.mean(surrogate_losses/1000):.6f}")
        print(f"  Recent mean (last 10%): {np.mean(surrogate_losses[-len(surrogate_losses)//10:]/1000):.6f}")
        print(f"  Max: {np.max(surrogate_losses/1000):.6f}")
        print(f"  Min: {np.min(surrogate_losses/1000):.6f}")
    
    plt.tight_layout()
    plt.show()

def inspect_tensorboard_data(log_dir):
    ea = event_accumulator.EventAccumulator(
        log_dir,
        size_guidance={
            event_accumulator.SCALARS: 0,
            event_accumulator.TENSORS: 0,
        }
    )
ea.Reload()

    # Print all available tags first
    print("\nAvailable tags:")
    for tag_type, tags in ea.Tags().items():
        print(f"\n{tag_type}:")
        for tag in tags:
            print(f"  - {tag}")
    
    # For each scalar tag, print detailed information about the first event
    print("\nDetailed first event for each scalar:")
    for tag in ea.Tags()['scalars']:
        events = ea.Scalars(tag)
        if events:
            first_event = events[0]
            print(f"\nTag: {tag}")
            print(f"  Event structure:")
            print(f"    - step: {first_event.step}")
            print(f"    - wall_time: {first_event.wall_time}")
            print(f"    - value: {first_event.value}")
            # Print the full event object to see all attributes
            print(f"  Full event object:")
            print(f"    {first_event}")

def load_and_plot_checkpoints(log_dir):
    # Get all model checkpoint files
    checkpoint_files = sorted([f for f in os.listdir(log_dir) if f.startswith('model_') and f.endswith('.pt')],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    steps = []
    rewards = []
    episode_lengths = []
    
    print("Loading checkpoint files...")
    for checkpoint_file in checkpoint_files:
        step = int(checkpoint_file.split('_')[1].split('.')[0])
        checkpoint_path = os.path.join(log_dir, checkpoint_file)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        # Extract training statistics
        if 'statistics' in checkpoint:
            stats = checkpoint['statistics']
            if 'mean_reward' in stats:
                steps.append(step)
                rewards.append(stats['mean_reward'])
            if 'mean_episode_length' in stats:
                episode_lengths.append(stats['mean_episode_length'])
        
        print(f"Loaded {checkpoint_file}")
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot rewards
    ax1.plot(steps, rewards, 'b-', label='Mean Reward')
    ax1.set_title('Training Reward')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Reward')
    ax1.grid(True)
    ax1.legend()
    
    # Plot episode lengths
    if episode_lengths:
        ax2.plot(steps, episode_lengths, 'g-', label='Episode Length')
        ax2.set_title('Episode Length')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Length')
        ax2.grid(True)
        ax2.legend()
    
    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\nTraining Statistics:")
    print(f"Number of checkpoints: {len(steps)}")
    print(f"Mean reward: {np.mean(rewards):.3f}")
    print(f"Min reward: {np.min(rewards):.3f}")
    print(f"Max reward: {np.max(rewards):.3f}")

def inspect_checkpoint(log_dir):
    # Load the latest checkpoint
    checkpoint_files = sorted([f for f in os.listdir(log_dir) if f.startswith('model_') and f.endswith('.pt')],
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if checkpoint_files:
        latest_checkpoint = checkpoint_files[-1]
        checkpoint_path = os.path.join(log_dir, latest_checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        print(f"\nContents of {latest_checkpoint}:")
        for key in checkpoint.keys():
            print(f"\nKey: {key}")
            if isinstance(checkpoint[key], dict):
                print("Contents:")
                for subkey in checkpoint[key].keys():
                    print(f"  - {subkey}")

# Plot the metrics
plot_training_metrics(log_dir)

# Inspect the data
inspect_tensorboard_data(log_dir)

# Load and plot from checkpoints
load_and_plot_checkpoints(log_dir)

# Inspect checkpoint contents
inspect_checkpoint(log_dir)