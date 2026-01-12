import random
import os
import json
import math

class RLAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2, q_table_path=None):
        """
        Simple Q-Learning Agent.
        
        Args:
            actions (list): List of possible actions (e.g., prompt keys).
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            epsilon (float): Exploration rate.
            q_table_path (str): Path to load/save Q-table.
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table_path = q_table_path
        self.q_table = {} # Maps state -> {action: q_value}
        
        if q_table_path and os.path.exists(q_table_path):
            self.load()

    def get_q(self, state, action):
        return self.q_table.get(state, {}).get(action, 0.0)

    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        
        # Greedy
        current_q = self.q_table.get(state, {})
        if not current_q:
            return random.choice(self.actions)
        
        # Find action with max Q
        max_q = max(current_q.values())
        best_actions = [a for a, q in current_q.items() if q == max_q]
        
        # If we haven't explored some actions for this state, they might be 0.0
        # If all known are negative, 0.0 (unexplored) is better. 
        # But efficiently, we just pick from best of what we know + unexplored defaults to 0
        # For simplicity in this dict structure:
        
        # Check if any action is missing from q_table (implicit 0.0)
        # If max_q < 0, then missing actions (0.0) are better.
        if max_q < 0:
            missing = [a for a in self.actions if a not in current_q]
            if missing:
                return random.choice(missing)
                
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """Q-Learning update."""
        current_q = self.get_q(state, action)
        
        # Max Q for next state
        next_q_values = self.q_table.get(next_state, {})
        max_next_q = max(next_q_values.values()) if next_q_values else 0.0
        
        # Update
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q

    def save(self):
        if self.q_table_path:
            with open(self.q_table_path, 'w') as f:
                json.dump(self.q_table, f)

    def load(self):
        try:
            with open(self.q_table_path, 'r') as f:
                self.q_table = json.load(f)
        except:
            print("Failed to load Q-table, starting fresh.")

