from collections import defaultdict
import os
import pandas as pd

from data.conceptnet import ConceptNet


def load_or_cache_conceptnet(path: str, sample_size: int = None, cache_path: str = None) -> pd.DataFrame:
    """
    Load ConceptNet triples with weights from a cached pickle file if available,
    otherwise process the raw TSV file and save the processed DataFrame to cache.

    Args:
        path (str): Path to the raw ConceptNet CSV/TSV file.
        sample_size (int, optional): Number of edges to sample (randomly). Default: None (load all).
        cache_path (str): Path to save or load the cached pickle file.

    Returns:
        pd.DataFrame: Processed ConceptNet triples in format (relation, head, tail, weight).
    """
    if cache_path and os.path.exists(cache_path):
        print(f"🔄 Loading from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"📥 Loading and processing raw ConceptNet...")
    cn = ConceptNet(path, max_edges=sample_size)
    df = cn.load()  # df must contain: relation, head, tail, weight

    if cache_path:
        df.to_pickle(cache_path)
        print(f"✅ Saved cache to: {cache_path}")

    return df


def print_top_prompts(log: list, top_k: int = 5) -> None:
    good_prompts = [x for x in log if x['reward'] == 1]
    sorted_prompts = sorted(good_prompts, key=lambda x: len(x['prompt']))[:top_k]

    for i, entry in enumerate(sorted_prompts):
        print(f"--- Top Prompt #{i+1} ---")
        print(entry['prompt'])
        print("Prediction:", entry['predicted'])
        print("Correct:", entry['correct'])
        print()


def print_template_statistics(log: list) -> None:
    stats = defaultdict(lambda: [0, 0])  # {template: [correct, total]}
    for row in log:
        stats[row['template']][0] += row['reward']
        stats[row['template']][1] += 1

    print("\n📊 Template Statistics:")
    for template, (correct, total) in stats.items():
        acc = correct / total if total > 0 else 0.0
        print(f"  {template}: {acc:.3f} ({correct}/{total})")


def save_log_csv(log: list, path: str = "mab_eval_log.csv") -> None:
    df = pd.DataFrame(log)
    df.to_csv(path, index=False)
    print(f"✅ Log saved to {path}")
