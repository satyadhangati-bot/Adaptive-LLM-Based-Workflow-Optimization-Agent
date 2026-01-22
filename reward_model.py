def compute_reward(env_reward, latency_penalty=0.1):
    """
    Combines environment reward with cost/latency penalty.
    """
    return env_reward - latency_penalty
