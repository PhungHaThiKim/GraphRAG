import random
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class KGEnvRL:
    """
    Reinforcement Learning environment for reasoning over Knowledge Graphs (KGs).
    The agent aims to find a path from a source entity to a target entity
    while optimizing for contextual relevance and conciseness.
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        samples: List[Dict],
        entity_to_vec: Optional[Dict[str, np.ndarray]] = None,
        triplet_to_vec: Optional[Dict[Union[str, Tuple[str, str, str]], np.ndarray]] = None,
        embed_model: Optional[SentenceTransformer] = None,
        max_steps: int = 10,
        lambda_context: float = 0.5,
        lambda_concise: float = 0.3,
        use_triple_embedding: bool = False,
    ):
        """
        Initialize the environment.

        Args:
            graph: NetworkX MultiDiGraph of the KG.
            samples: List of QA samples with 'source_entities', 'target_entity', 'question_text'.
            entity_to_vec: Optional dictionary mapping entity → vector.
            triplet_to_vec: Optional dictionary mapping (h, r, t) or formatted triple string → vector.
            embed_model: SentenceTransformer model for on-the-fly embedding.
            max_steps: Max number of actions per episode.
            lambda_context: Weight for context-relevance reward.
            lambda_concise: Weight for concise-path reward.
            use_triple_embedding: If True, use triple-based embedding; else use entity vectors.
        """
        self.graph = graph
        self.samples = samples
        self.entity_to_vec = entity_to_vec or {}
        self.triplet_to_vec = triplet_to_vec or {}
        self.embed_model = embed_model
        self.max_steps = max_steps
        self.lambda_context = lambda_context
        self.lambda_concise = lambda_concise
        self.use_triple_embedding = use_triple_embedding

    def reset(self, fixed_sample: bool = False) -> np.ndarray:
        """
        Reset the environment for a new episode.

        Args:
            fixed_sample: If True, reuse the current sample. Else, pick a random valid sample.

        Returns:
            The initial state embedding.
        """
        if fixed_sample:
            assert hasattr(self, "sample"), "Must set `env.sample` before reset(fixed_sample=True)"
            self.source = self.sample['source_entities'][0].lower()
            self.target = self.sample['target_entity'].lower()
        else:
            for _ in range(50):  # avoid infinite loop
                candidate = random.choice(self.samples)
                src = candidate['source_entities'][0].lower()
                tgt = candidate['target_entity'].lower()
                if src in self.graph and tgt in self.graph:
                    try:
                        nx.shortest_path(self.graph, src, tgt)
                        self.sample = candidate
                        self.source = src
                        self.target = tgt
                        break
                    except nx.NetworkXNoPath:
                        continue

        self.current = self.source
        self.steps = 0
        self.path = [self.current]
        self.context_embedding = self.embed_model.encode(self.sample['question_text'])

        return self._get_state()

    def _get_state(self) -> np.ndarray:
        """
        Get current state embedding based on position and target.

        Returns:
            State embedding vector (np.ndarray).
        """
        if self.use_triple_embedding:
            relation = self._get_relation_between(self.current, self.target) or "relatedTo"
            triple = (self.current, relation, self.target)
            if self.triplet_to_vec:
                return self.triplet_to_vec.get(triple, np.zeros(768))
            else:
                triple_str = f"[H] {self.current} [R] {relation} [T] {self.target}"
                return self.embed_model.encode(triple_str)
        else:
            e_cur = self.entity_to_vec.get(self.current, np.zeros(384))
            e_tgt = self.entity_to_vec.get(self.target, np.zeros(384))
            return np.concatenate([e_cur, e_tgt - e_cur])

    def _get_relation_between(self, h: str, t: str) -> Optional[str]:
        """
        Get a relation label between two nodes in the graph.

        Returns:
            The relation name, or None if no edge exists.
        """
        if self.graph.has_edge(h, t):
            rel_data = self.graph.get_edge_data(h, t)
            return rel_data[list(rel_data.keys())[0]]['relation']
        return None

    def _get_path_embedding(self) -> np.ndarray:
        """
        Compute embedding for the current path using triple or node vectors.

        Returns:
            Embedding of the full path so far.
        """
        if self.use_triple_embedding:
            triple_vecs = []
            for i in range(len(self.path) - 1):
                h, t = self.path[i], self.path[i + 1]
                relation = self._get_relation_between(h, t) or "relatedTo"
                triple_str = f"[H] {h} [R] {relation} [T] {t}"
                if self.triplet_to_vec:
                    vec = self.triplet_to_vec.get(triple_str, np.zeros(384))
                else:
                    vec = self.embed_model.encode(triple_str)
                triple_vecs.append(vec)
            return np.mean(triple_vecs, axis=0) if triple_vecs else np.zeros(384)
        else:
            node_vecs = [self.entity_to_vec.get(e, np.zeros(384)) for e in self.path]
            return np.mean(node_vecs, axis=0)

    def get_actions(self) -> List[str]:
        """
        Get all possible next-step actions from current node.

        Returns:
            List of neighbor node names.
        """
        return list(self.graph[self.current]) if self.current in self.graph else []

    def step(self, action: str) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take one step in the environment.

        Args:
            action: Node name to move to.

        Returns:
            - New state
            - Reward
            - Done flag
            - Info dictionary with reward components
        """
        if action not in self.get_actions():
            return self._get_state(), -1.0, True, {"msg": "Invalid action"}

        self.current = action
        self.steps += 1
        self.path.append(self.current)
        done = self.current == self.target or self.steps >= self.max_steps

        # Reward components
        r_reach = 1.0 if self.current == self.target else 0.0
        r_context = cosine_similarity(
            self._get_path_embedding().reshape(1, -1),
            self.context_embedding.reshape(1, -1)
        )[0][0]
        r_concise = 1.0 / len(self.path)

        reward = r_reach + self.lambda_context * r_context + self.lambda_concise * r_concise

        return self._get_state(), reward, done, {
            "r_reach": r_reach,
            "r_context": r_context,
            "r_concise": r_concise,
            "path": list(self.path),
            "done": done
        }

    def render(self):
        """
        Print current state of the environment.
        """
        print(f"[Step {self.steps}] Current: {self.current}, Target: {self.target}")
        print("→ Path so far:", " → ".join(self.path))
