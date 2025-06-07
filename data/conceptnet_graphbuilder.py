import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple
from sentence_transformers import SentenceTransformer

class ConceptNetGraphBuilder:
    def __init__(self, triples: pd.DataFrame):
        """
        Initialize the graph builder with a DataFrame of ConceptNet triples.

        Args:
            triples (pd.DataFrame): DataFrame with columns ['relation', 'head', 'tail']
        """
        required_cols = {'relation', 'head', 'tail'}
        if not required_cols.issubset(triples.columns):
            raise ValueError("DataFrame must contain columns: 'relation', 'head', 'tail'")

        self.triples = triples
        self.graph = nx.MultiDiGraph()  # Directed multigraph to support multiple relations

    def build_graph(self):
        """
        Build the directed multigraph from ConceptNet triples.
        Each edge is labeled with the relation type.
        """
        for _, row in self.triples.iterrows():
            head, relation, tail = row['head'], row['relation'], row['tail']
            self.graph.add_edge(head, tail, relation=relation)
        return self.graph

    def get_neighbors(self, node):
        """
        Get all direct neighbors of a node.
        """
        return list(self.graph.neighbors(node))

    def get_relation(self, source, target):
        """
        Get all relations between two nodes.
        """
        if self.graph.has_edge(source, target):
            return [data['relation'] for data in self.graph.get_edge_data(source, target).values()]
        else:
            return []

    def shortest_path(self, source, target):
        """
        Get the shortest path (sequence of nodes) between two concepts.
        """
        try:
            return nx.shortest_path(self.graph, source=source, target=target)
        except nx.NetworkXNoPath:
            return []

    def get_graph(self):
        """
        Return the internal NetworkX graph.
        """
        return self.graph

    def build_entity_embeddings(self, model_name='all-MiniLM-L6-v2', batch_size=256) -> Dict[str, np.ndarray]:
        """
        Generate sentence embeddings for all nodes in the graph.

        Returns:
            dict: A dictionary {entity: embedding_vector}
        """
        print(f"🔠 Encoding {self.graph.number_of_nodes():,} nodes using model '{model_name}'")
        model = SentenceTransformer(model_name)
        nodes = list(self.graph.nodes)
        embeddings = model.encode(nodes, batch_size=batch_size, show_progress_bar=True)
        return {node: vec for node, vec in zip(nodes, embeddings)}

    def build_triple_embeddings(self, model_name='all-MiniLM-L6-v2', batch_size=256, use_special_tokens=False) -> Dict[Tuple[str, str, str], np.ndarray]:
        """
        Generate sentence embeddings for all (head, relation, tail) triples.

        Args:
            model_name (str): Name of the SentenceTransformer model.
            batch_size (int): Batch size for encoding.
            use_special_tokens (bool): Whether to use [H], [R], [T] format.

        Returns:
            dict: A dictionary {(head, relation, tail): embedding_vector}
        """
        print(f"🧠 Encoding {len(self.triples):,} triples using model '{model_name}'")
        model = SentenceTransformer(model_name)
        triples = list(self.triples.itertuples(index=False, name=None))  # (relation, head, tail)

        if use_special_tokens:
            texts = [f"[H] {h} [R] {r} [T] {t}" for r, h, t in triples]
        else:
            texts = [f"{h} {r} {t}" for r, h, t in triples]

        embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
        return {(h, r, t): vec for (r, h, t), vec in zip(triples, embeddings)}
