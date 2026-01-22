from env.workflow_env import WorkflowEnv
from src.llm_agent import LLMAgent

def evaluate(episodes=10):
    env = WorkflowEnv()
    agent = LLMAgent()
    total_reward = 0

    for _ in range(episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = agent.propose_action(state)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward

    print("Average Reward:", total_reward / episodes)
