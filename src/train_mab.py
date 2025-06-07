import random
from typing import List, Tuple, Dict, Callable, Any


def train_mab_on_data(
    results: List[Dict[str, Any]],
    mab,
    simulate_fn: Callable[[str, str, List[str]], Tuple[int, str]],
    max_samples: int = 100,
    top_k_paths: int = 3
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate prompt templates using MAB on a sampled subset of results.
    Each sample contains a question, reasoning paths, options, and correct answer.

    Args:
        results (List[Dict]): List of question samples + reasoning paths.
        mab (PromptConstructorMAB): MAB selector for prompt templates.
        simulate_fn (callable): Function to simulate GPT (prompt, answer, options) → (reward, pred).
        max_samples (int): Max number of questions to evaluate.
        top_k_paths (int): Number of top paths to include in the prompt.

    Returns:
        Tuple[float, List[Dict]]: (Accuracy over samples, Evaluation log with per-question metadata)
    """
    total = 0
    correct = 0
    log = []

    # Sample max_samples items from results
    sampled = random.sample(results, min(len(results), max_samples))

    for idx, r in enumerate(sampled):
        # Ensure data is valid
        if "correct_answer" not in r or "options" not in r:
            continue

        # Get list of reasoning paths (default to single 'path' if 'paths' is missing)
        all_paths = r.get("paths", [r['path']])
        selected_paths = all_paths[:top_k_paths] if len(all_paths) >= top_k_paths else all_paths

        # Select and apply prompt template
        template_name = mab.select_template()
        prompt = mab.format_prompt(r['question_text'], selected_paths, template_name, r['options'])

        print(f"\n📚 [{idx+1}] Template: {template_name}, QID: {r['question_id']}, Path len: {len(selected_paths)}")

        # Send prompt to GPT (or simulated model) and receive reward
        reward, prediction = simulate_fn(prompt, r['correct_answer'], r['options'])

        # Record feedback for MAB
        mab.record_feedback(template_name, reward)

        total += 1
        correct += reward

        # Save result log for later analysis
        log.append({
            "template": template_name,
            "question_id": r['question_id'],
            "prompt": prompt,
            "predicted": prediction,
            "correct": r['correct_answer'],
            "reward": reward
        })

    acc = correct / total if total > 0 else 0.0
    return acc, log
