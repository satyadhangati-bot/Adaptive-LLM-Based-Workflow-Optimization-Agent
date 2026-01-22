import torch
from env.workflow_env import WorkflowEnv
from src.llm_agent import LLMAgent
from src.rl_policy import RLPolicy
from src.reward_model import compute_reward

def train(episodes=50):
    env = WorkflowEnv()
    llm = LLMAgent()
    policy = RLPolicy()

    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = llm.propose_action(state)
            next_state, env_reward, done, _, _ = env.step(action)
            reward = compute_reward(env_reward)

            policy.update(
                torch.tensor(state).float().unsqueeze(0),
                action,
                reward
            )

            state = next_state

        print(f"Episode {episode+1} completed")

    return policy
