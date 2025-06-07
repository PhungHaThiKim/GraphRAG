import torch
import numpy as np
import random
from typing import List, Dict, Any, Optional


def train_rl_multi_epoch(
    policy_net,
    env,
    samples: List[Dict[str, Any]],
    num_epochs: int = 5,
    repeats_per_sample: int = 3,
    gamma: float = 0.99,
    use_triple: bool = False,
    triplet_to_vec: Optional[Dict] = None,
    entity_to_vec: Optional[Dict] = None,
    print_every: int = 1
) -> List[Dict[str, Any]]:
    """
    Train the policy network with REINFORCE over multiple epochs.

    Args:
        policy_net (nn.Module): The policy network (PolicyNet or TriplePolicyNet).
        env (KGEnvRL): RL environment with KG and questions.
        samples (List[Dict]): List of question samples.
        num_epochs (int): Number of epochs.
        repeats_per_sample (int): How many episodes to run per question.
        gamma (float): Discount factor.
        use_triple (bool): Whether to use triple embedding instead of entity.
        triplet_to_vec (dict): Precomputed triple embeddings.
        entity_to_vec (dict): Precomputed entity embeddings.
        print_every (int): Print progress every N repeats.

    Returns:
        List[Dict]: List of successful paths with metadata.
    """
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-3)
    all_results = []

    for epoch in range(1, num_epochs + 1):
        print(f"\n🌍 === EPOCH {epoch}/{num_epochs} ===")

        for idx, sample in enumerate(samples):
            env.sample = sample  # fix the sample before reset
            paths_for_sample = []

            for repeat in range(repeats_per_sample):
                state = env.reset(fixed_sample=True)
                log_probs, rewards = [], []

                for step in range(env.max_steps):
                    actions = env.get_actions()
                    if not actions:
                        break

                    # === Encode actions: triple or entity ===
                    if use_triple:
                        triple_candidates = []
                        for t in actions:
                            if env.graph.has_edge(env.current, t):
                                for edge_key in env.graph.get_edge_data(env.current, t):
                                    r = env.graph[env.current][t][edge_key]['relation']
                                    triple_candidates.append((env.current, r, t))

                        if not triple_candidates:
                            break

                        triple_vecs = np.array([
                            triplet_to_vec.get(tri, np.zeros(768)) for tri in triple_candidates
                        ])
                        triple_vecs = torch.tensor(triple_vecs, dtype=torch.float32).unsqueeze(0)
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        logits = policy_net(state_tensor, triple_vecs)
                        probs = torch.softmax(logits, dim=-1)
                    else:
                        action_vecs = np.array([
                            entity_to_vec.get(a, np.zeros(384)) for a in actions
                        ])
                        action_vecs = torch.tensor(action_vecs, dtype=torch.float32).unsqueeze(0)
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                        logits = policy_net(state_tensor, action_vecs)
                        probs = torch.softmax(logits, dim=-1)

                    # === Handle invalid probs ===
                    if torch.isnan(probs).any() or torch.isinf(probs).any():
                        break

                    m = torch.distributions.Categorical(probs)
                    action_idx = m.sample()
                    log_prob = m.log_prob(action_idx)

                    if use_triple:
                        _, _, next_node = triple_candidates[action_idx.item()]
                    else:
                        next_node = actions[action_idx.item()]

                    next_state, reward, done, info = env.step(next_node)
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    state = next_state

                    if done:
                        break

                # === REINFORCE update ===
                returns, R = [], 0
                for r in reversed(rewards):
                    R = r + gamma * R
                    returns.insert(0, R)

                if not returns:
                    continue

                returns = torch.tensor(returns)
                if len(returns) > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                log_probs = torch.stack(log_probs)
                loss = - (log_probs * returns).sum()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if done and reward > 0:
                    paths_for_sample.append({
                        "epoch": epoch,
                        "question_id": sample["question_id"],
                        "question_text": sample["question_text"],
                        "source": sample["source_entities"][0],
                        "target": sample["target_entity"],
                        "path": env.path.copy(),
                        "reward": float(sum(rewards)),
                        "steps": len(env.path),
                    })

                if repeat % print_every == 0:
                    print(f"🧪 Epoch {epoch} | Sample {idx+1}/{len(samples)} | Repeat {repeat+1}/{repeats_per_sample} | "
                          f"Reward: {sum(rewards):.3f} | Path: {' → '.join(env.path)}")

            all_results.extend(paths_for_sample)
