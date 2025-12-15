"""
Reinforcement Learning Agent Module for AI Resume Screening System
====================================================================
Implements Q-Learning agent for adaptive hiring decisions.

Author: AI Project Team
Course: CS-351 Artificial Intelligence
"""

import numpy as np
import pickle
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Experience:
    """Represents a single RL experience tuple."""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class HiringRLAgent:
    """
    Q-Learning Reinforcement Learning Agent for Hiring Decisions.
    
    The agent learns an optimal policy for making hiring decisions
    (Shortlist, Hold, Reject) based on model confidence scores.
    
    State Space: Discretized confidence scores (0-9)
    Action Space: [0: Shortlist, 1: Hold, 2: Reject]
    
    Q-Learning Update Rule:
    Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
    
    Attributes:
        n_states: Number of discrete states
        n_actions: Number of possible actions
        q_table: Q-value table (state x action)
        lr: Learning rate (α)
        gamma: Discount factor (γ)
        epsilon: Exploration rate (ε)
    """
    
    ACTIONS = ['Shortlist', 'Hold', 'Reject']
    
    def __init__(self, n_states: int = 10, n_actions: int = 3, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 1.0):
        """
        Initialize the RL agent.
        
        Args:
            n_states: Number of states (confidence bins)
            n_actions: Number of actions (3: Shortlist/Hold/Reject)
            learning_rate: Learning rate alpha (0.1 default)
            discount_factor: Discount factor gamma (0.95 default)
            epsilon: Initial exploration rate (1.0 = full exploration)
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = 0.995
        self.min_epsilon = 0.01
        
        # Training history
        self.episode_rewards = []
        self.cumulative_rewards = []
        self.epsilon_history = []
    
    def get_state(self, confidence: float) -> int:
        """
        Convert continuous confidence score to discrete state.
        
        Args:
            confidence: Model confidence (0.0 to 1.0)
            
        Returns:
            Discrete state index (0 to n_states-1)
        """
        return min(int(confidence * self.n_states), self.n_states - 1)
    
    def choose_action(self, state: int, training: bool = True) -> int:
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state index
            training: Whether in training mode (uses exploration)
            
        Returns:
            Action index (0: Shortlist, 1: Hold, 2: Reject)
        """
        if training and np.random.rand() < self.epsilon:
            # Exploration: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploitation: best Q-value action
            return np.argmax(self.q_table[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int):
        """
        Update Q-table using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
        """
        # Find best next action value
        best_next_q = np.max(self.q_table[next_state])
        
        # TD target
        td_target = reward + self.gamma * best_next_q
        
        # TD error
        td_error = td_target - self.q_table[state, action]
        
        # Update Q-value
        self.q_table[state, action] += self.lr * td_error
    
    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    
    def get_decision(self, confidence: float) -> Tuple[str, int]:
        """
        Get hiring decision for a given confidence score.
        
        Args:
            confidence: Model confidence (0.0 to 1.0)
            
        Returns:
            Tuple of (decision_name, action_index)
        """
        state = self.get_state(confidence)
        action = self.choose_action(state, training=False)
        return self.ACTIONS[action], action
    
    def train(self, n_episodes: int = 1000, verbose: bool = True) -> Dict:
        """
        Train the agent through simulated hiring scenarios.
        
        Args:
            n_episodes: Number of training episodes
            verbose: Whether to print progress
            
        Returns:
            Training statistics dictionary
        """
        cumulative_reward = 0
        
        for episode in range(n_episodes):
            # Simulate a candidate (random confidence)
            confidence = np.random.random()
            is_good_match = confidence > 0.7  # Ground truth assumption
            
            state = self.get_state(confidence)
            action = self.choose_action(state)
            reward = self._get_reward(action, is_good_match, confidence)
            
            # Next state (independent candidate)
            next_confidence = np.random.random()
            next_state = self.get_state(next_confidence)
            
            # Update Q-table
            self.update(state, action, reward, next_state)
            
            # Decay exploration
            self.decay_epsilon()
            
            # Track metrics
            cumulative_reward += reward
            
            if episode % 10 == 0:
                self.episode_rewards.append(reward)
                self.cumulative_rewards.append(cumulative_reward)
                self.epsilon_history.append(self.epsilon)
        
        if verbose:
            print(f"✅ Training complete after {n_episodes} episodes")
            print(f"   Final epsilon: {self.epsilon:.4f}")
            print(f"   Cumulative reward: {cumulative_reward:.2f}")
        
        return {
            'episodes': n_episodes,
            'final_epsilon': self.epsilon,
            'cumulative_reward': cumulative_reward,
            'q_table': self.q_table.copy()
        }
    
    def _get_reward(self, action: int, is_good_match: bool, confidence: float) -> float:
        """
        Calculate reward for action given ground truth.
        
        Reward Structure:
        - Correct shortlist (good match): +10
        - Wrong shortlist (bad match): -10
        - Correct reject (bad match): +5
        - Wrong reject (good match): -5
        - Hold: -1 (slight penalty for indecision)
        
        Args:
            action: Action taken (0: Shortlist, 1: Hold, 2: Reject)
            is_good_match: Whether candidate is actually a good match
            confidence: Model confidence score
            
        Returns:
            Reward value
        """
        if action == 0:  # Shortlist
            return 10.0 if is_good_match else -10.0
        elif action == 2:  # Reject
            return 5.0 if not is_good_match else -5.0
        else:  # Hold
            return -1.0  # Slight penalty for indecision
    
    def get_policy_summary(self) -> Dict:
        """
        Get a summary of the learned policy.
        
        Returns:
            Dictionary with policy information
        """
        policy = {}
        
        for state in range(self.n_states):
            best_action = np.argmax(self.q_table[state])
            conf_low = state / self.n_states
            conf_high = (state + 1) / self.n_states
            
            policy[f"{conf_low:.1f}-{conf_high:.1f}"] = {
                'action': self.ACTIONS[best_action],
                'q_values': self.q_table[state].tolist()
            }
        
        return policy
    
    def plot_training_progress(self, figsize: Tuple[int, int] = (14, 5)):
        """Plot training progress visualization."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Cumulative rewards
        ax1 = axes[0]
        episodes = list(range(0, len(self.cumulative_rewards) * 10, 10))
        ax1.plot(episodes, self.cumulative_rewards, linewidth=2, color='#2ecc71')
        ax1.set_xlabel('Episode', fontweight='bold')
        ax1.set_ylabel('Cumulative Reward', fontweight='bold')
        ax1.set_title('Learning Progress', fontweight='bold')
        ax1.grid(alpha=0.3)
        
        # Epsilon decay
        ax2 = axes[1]
        ax2.plot(episodes, self.epsilon_history, linewidth=2, color='#e74c3c')
        ax2.set_xlabel('Episode', fontweight='bold')
        ax2.set_ylabel('Epsilon (ε)', fontweight='bold')
        ax2.set_title('Exploration Rate Decay', fontweight='bold')
        ax2.grid(alpha=0.3)
        
        # Q-table heatmap
        ax3 = axes[2]
        import seaborn as sns
        sns.heatmap(self.q_table, annot=True, fmt='.2f', cmap='RdYlGn',
                    xticklabels=self.ACTIONS,
                    yticklabels=[f'{i/10:.1f}-{(i+1)/10:.1f}' for i in range(self.n_states)],
                    ax=ax3)
        ax3.set_xlabel('Action', fontweight='bold')
        ax3.set_ylabel('Confidence Range (State)', fontweight='bold')
        ax3.set_title('Q-Table (State-Action Values)', fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def save(self, filepath: str):
        """Save the agent to disk."""
        data = {
            'q_table': self.q_table,
            'n_states': self.n_states,
            'n_actions': self.n_actions,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'cumulative_rewards': self.cumulative_rewards
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load the agent from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.n_states = data['n_states']
            self.n_actions = data['n_actions']
            self.lr = data['lr']
            self.gamma = data['gamma']
            self.epsilon = data['epsilon']
            self.episode_rewards = data.get('episode_rewards', [])
            self.cumulative_rewards = data.get('cumulative_rewards', [])
    
    def print_policy(self):
        """Pretty print the learned policy."""
        print("\n" + "=" * 60)
        print("📋 LEARNED HIRING POLICY")
        print("=" * 60)
        
        policy = self.get_policy_summary()
        
        print(f"\n{'Confidence Range':<20} {'Decision':<15} {'Q-Values'}")
        print("-" * 60)
        
        for conf_range, info in policy.items():
            q_str = f"S:{info['q_values'][0]:+.2f} H:{info['q_values'][1]:+.2f} R:{info['q_values'][2]:+.2f}"
            print(f"{conf_range:<20} {info['action']:<15} {q_str}")
        
        print("\n" + "=" * 60)
        print("Policy Interpretation:")
        print("  • High confidence (>0.7): Shortlist (hire)")
        print("  • Medium confidence (0.3-0.7): Hold for review")
        print("  • Low confidence (<0.3): Reject")
        print("=" * 60)


def demo_rl_agent():
    """Demonstrate the RL agent."""
    print("\n" + "=" * 60)
    print("🤖 RL AGENT DEMONSTRATION")
    print("=" * 60)
    
    # Create and train agent
    agent = HiringRLAgent(n_states=10, n_actions=3)
    print("\n🔄 Training agent for 1000 episodes...")
    
    stats = agent.train(n_episodes=1000)
    
    # Print learned policy
    agent.print_policy()
    
    # Test decisions
    print("\n📊 Sample Decisions:")
    test_confidences = [0.95, 0.75, 0.50, 0.25, 0.10]
    
    for conf in test_confidences:
        decision, _ = agent.get_decision(conf)
        print(f"   Confidence {conf:.0%} → {decision}")
    
    return agent


if __name__ == "__main__":
    demo_rl_agent()
