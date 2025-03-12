import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque

class AdvancedInventoryQLearningAgent:
    def __init__(
        self,
        max_inventory,
        demand_lambda,
        cost_holding,
        cost_shortage,
        discount=0.9,
        alpha=0.1,
        alpha_min=0.01,
        alpha_decay=0.999,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        num_episodes=3000,
        max_steps=200
    ):
        # Environment parameters
        self.max_inventory = max_inventory
        self.demand_lambda = demand_lambda
        self.cost_holding = cost_holding
        self.cost_shortage = cost_shortage

        # Q-learning hyperparams
        self.discount = discount
        self.alpha = alpha
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.num_episodes = num_episodes
        self.max_steps = max_steps

        # Initialize Q-table
        self.Q = {}
        self._init_Q()

        # Tracking performance
        self.episode_costs = []
        self.rolling_avg_cost = []

    def _init_Q(self):
        """Initialize Q for all valid (state, action) pairs."""
        for on_hand in range(self.max_inventory + 1):
            for on_order in range(self.max_inventory + 1 - on_hand):
                state = (on_hand, on_order)
                self.Q[state] = {}
                max_order = self.max_inventory - (on_hand + on_order)
                for action in range(max_order + 1):
                    self.Q[state][action] = np.random.uniform(0, 1)

    def _get_demand(self):
        """Sample demand from Poisson distribution."""
        return np.random.poisson(self.demand_lambda)

    def _env_step(self, state, action):
        """
        Environment transition:
          - Demand arrives
          - leftover = on_hand + on_order - demand (if positive)
          - reward = negative of holding & shortage costs
          - next_state = (leftover, action)
        """
        on_hand, on_order = state
        total_inv = on_hand + on_order
        demand = self._get_demand()
        sold = min(total_inv, demand)
        leftover = total_inv - sold

        holding_cost = leftover * self.cost_holding
        shortage_cost = max(0, demand - total_inv) * self.cost_shortage
        reward = - (holding_cost + shortage_cost)

        next_state = (leftover, action)
        return next_state, reward

    def _choose_action(self, state):
        """
        Epsilon-greedy action selection.
        """
        actions = self.Q[state]
        if np.random.rand() < self.epsilon:
            return np.random.choice(list(actions.keys()))
        else:
            return max(actions, key=actions.get)

    def _update_q(self, state, action, reward, next_state):
        """
        Q-learning update:
          Q(s,a) += alpha * [reward + gamma*max_a'(Q(s',a')) - Q(s,a)]
        """
        best_next_action = max(self.Q[next_state], key=self.Q[next_state].get)
        td_target = reward + self.discount * self.Q[next_state][best_next_action]
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += self.alpha * td_error

    def train(self):
        """
        Main training loop. Tracks episode costs and rolling average.
        """
        cost_window = deque(maxlen=50)
        for ep in range(self.num_episodes):
            # Random initial state
            on_hand = np.random.randint(0, self.max_inventory + 1)
            on_order = np.random.randint(0, self.max_inventory - on_hand + 1)
            state = (on_hand, on_order)

            total_reward = 0
            for _ in range(self.max_steps):
                action = self._choose_action(state)
                next_state, reward = self._env_step(state, action)
                self._update_q(state, action, reward, next_state)
                state = next_state
                total_reward += reward

            # Track cost (negative of reward)
            episode_cost = -total_reward
            self.episode_costs.append(episode_cost)
            cost_window.append(episode_cost)
            self.rolling_avg_cost.append(np.mean(cost_window))

            # Decay alpha & epsilon
            self.alpha = max(self.alpha_min, self.alpha * self.alpha_decay)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        """Return the best action for each state."""
        policy = {}
        for s, actions in self.Q.items():
            best_a = max(actions, key=actions.get)
            policy[s] = best_a
        return policy

    def evaluate_policy(self, policy, test_episodes=1000):
        """
        Simulate the environment under the given policy to estimate total reward.
        """
        total_reward = 0.0
        for _ in range(test_episodes):
            # Random initial state
            on_hand = np.random.randint(0, self.max_inventory + 1)
            on_order = np.random.randint(0, self.max_inventory - on_hand + 1)
            state = (on_hand, on_order)

            for _ in range(self.max_steps):
                action = policy.get(state, 0)
                next_state, reward = self._env_step(state, action)
                total_reward += reward
                state = next_state

        return total_reward


# Example 
if __name__ == "__main__":
    # Seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    agent = AdvancedInventoryQLearningAgent(
        max_inventory=10,
        demand_lambda=4,
        cost_holding=8,
        cost_shortage=10,
        discount=0.9,
        alpha=0.1,
        alpha_min=0.01,
        alpha_decay=0.999,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        num_episodes=3000,
        max_steps=200
    )
    agent.train()

    # Plot training progress
    plt.figure(figsize=(10,5))
    plt.plot(agent.episode_costs, label='Episode Cost', alpha=0.3)
    plt.plot(agent.rolling_avg_cost, label='Rolling Avg (window=50)', color='red')
    plt.title('Training Progress: Cost per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # RL policy
    rl_policy = agent.get_policy()

    # Evaluate
    test_episodes = 5000
    rl_reward = agent.evaluate_policy(rl_policy, test_episodes=test_episodes)
    rl_cost = -rl_reward
    print(f"Total Cost with RL Policy (over {test_episodes} episodes): {rl_cost:,.2f}")

    # Benchmark: order up to full capacity
    def order_up_to_policy(state, capacity):
        on_hand, on_order = state
        current_inv = on_hand + on_order
        order_needed = capacity - current_inv
        return max(0, order_needed)

    bm_policy = {s: order_up_to_policy(s, agent.max_inventory) for s in agent.Q.keys()}
    bm_reward = agent.evaluate_policy(bm_policy, test_episodes=test_episodes)
    bm_cost = -bm_reward
    print(f"Total Cost with Benchmark Policy (over {test_episodes} episodes): {bm_cost:,.2f}")

    # Compare policies
    states = list(rl_policy.keys())
    orders_rl = [rl_policy[s] for s in states]
    orders_bm = [bm_policy[s] for s in states]

    x_pos = np.arange(len(states))
    bar_width = 0.35

    plt.figure(figsize=(14,6))
    plt.bar(x_pos, orders_rl, width=bar_width, color='purple', label='RL Policy')
    plt.bar(x_pos + bar_width, orders_bm, width=bar_width, color='orange', label='Benchmark')
    plt.title('Comparison: RL vs. Order-Up-To Policy')
    plt.xlabel('State (on_hand, on_order)')
    plt.ylabel('Order Quantity')
    plt.xticks(x_pos + bar_width/2, states, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Final cost comparison
    plt.figure(figsize=(6,5))
    plt.bar(['RL Policy','Benchmark'], [rl_cost, bm_cost], color=['purple','coral'])
    plt.title('Total Cost Comparison of Policies')
    plt.ylabel('Total Cost')
    plt.tight_layout()
    plt.show()
