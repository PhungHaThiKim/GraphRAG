import numpy as np
import random
from collections import defaultdict


class PromptConstructorMAB:
    """
    Multi-Armed Bandit (MAB)-based Prompt Constructor.
    Dynamically selects prompt templates based on observed rewards (e.g., GPT accuracy).
    """

    def __init__(self, templates: dict, temperature: float = 1.0):
        """
        Initialize the bandit-based template selector.

        Args:
            templates (dict): Dict[str, callable], mapping template_name → function to format prompt.
            temperature (float): Softmax temperature for exploration vs. exploitation.
        """
        self.templates = templates
        self.template_names = list(templates.keys())
        self.counts = defaultdict(int)        # Track how many times each template is used
        self.successes = defaultdict(int)     # Track how many successes (reward=1) per template
        self.temperature = temperature

    def select_template(self) -> str:
        """
        Select a template name using softmax exploration over success rates.

        Returns:
            str: The selected template name.
        """
        if all(self.counts[name] == 0 for name in self.template_names):
            return random.choice(self.template_names)  # Uniformly random on first use

        # Compute success rates (exploitation)
        rates = np.array([
            self.successes[name] / self.counts[name] if self.counts[name] > 0 else 0.0
            for name in self.template_names
        ])

        # Softmax with temperature (higher T → more exploration)
        probs = np.exp(rates / self.temperature)
        probs /= np.sum(probs)

        return np.random.choice(self.template_names, p=probs)

    def record_feedback(self, template_name: str, reward: int):
        """
        Update statistics for a template based on reward feedback.

        Args:
            template_name (str): Name of the template used.
            reward (int): Reward received (1 = correct, 0 = incorrect).
        """
        self.counts[template_name] += 1
        self.successes[template_name] += reward

    def format_prompt(self, question_text: str, paths: list, template_name: str, options: list) -> str:
        """
        Format a prompt given a question, paths and selected template.

        Args:
            question_text (str): The natural language question.
            paths (list): A list of reasoning paths (strings or triples).
            template_name (str): Name of template to use.
            options (list): Multiple-choice answer options.

        Returns:
            str: The fully formatted prompt string.
        """
        return self.templates[template_name](question_text, paths, options)

    def reset(self):
        """
        Reset all internal counters (e.g., for new training round).
        """
        self.counts.clear()
        self.successes.clear()
