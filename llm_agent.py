from transformers import pipeline

class LLMAgent:
    def __init__(self):
        self.generator = pipeline(
            "text-generation",
            model="google/flan-t5-base",
            max_length=128
        )

    def propose_action(self, state):
        prompt = f"""
        Workflow complexity: {state[0]}
        Suggest an optimization strategy (0, 1, or 2):
        """
        response = self.generator(prompt)
        return int(state[0] % 3)  # deterministic fallback
