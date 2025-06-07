import pandas as pd
from typing import List, Tuple


class ConceptNet:
    """
    A utility class to load and extract English ConceptNet triples from a raw TSV/CSV file.
    """

    def __init__(self, path: str, max_edges: int = None):
        """
        Initialize a ConceptNet object.

        Args:
            path (str): Path to the ConceptNet TSV or CSV file.
            max_edges (int, optional): Maximum number of edges to load. Defaults to None (load all).
        """
        self.path = path
        self.max_edges = max_edges
        self.df: pd.DataFrame = None  # Will hold cleaned triple data

    def load(self) -> pd.DataFrame:
        """
        Load and preprocess ConceptNet data.
        Keeps only English-language triples and extracts clean head/relation/tail fields.

        Returns:
            pd.DataFrame: DataFrame with columns ['relation', 'head', 'tail']
        """
        df = pd.read_csv(
            self.path,
            sep='\t',
            header=None,
            usecols=[1, 2, 3],
            names=['relation', 'start', 'end'],
            engine='c',
            on_bad_lines='skip'
        )

        # Keep only English concepts
        df = df[df['start'].str.startswith('/c/en/') & df['end'].str.startswith('/c/en/')]

        # Extract concept names
        df['head'] = df['start'].str.extract(r'^/c/en/([^/]+)')[0].str.replace('_', ' ', regex=False)
        df['tail'] = df['end'].str.extract(r'^/c/en/([^/]+)')[0].str.replace('_', ' ', regex=False)

        # Extract relation name
        df['relation'] = df['relation'].str.extract(r'^/r/([^/]+)')[0]

        # Optionally sample
        if self.max_edges:
            df = df.sample(n=self.max_edges, random_state=42)

        self.df = df[['relation', 'head', 'tail']]
        return self.df

    def get_edges(self) -> pd.DataFrame:
        """
        Get the preprocessed DataFrame of triples.

        Returns:
            pd.DataFrame: DataFrame with columns ['relation', 'head', 'tail']
        """
        if self.df is None:
            raise ValueError("You must call .load() before accessing edges.")
        return self.df

    def get_triples(self) -> List[Tuple[str, str, str]]:
        """
        Get the triples as a list of (head, relation, tail) tuples.

        Returns:
            List[Tuple[str, str, str]]: List of concept triples.
        """
        if self.df is None:
            raise ValueError("You must call .load() before accessing triples.")
        return list(self.df.itertuples(index=False, name=None))
