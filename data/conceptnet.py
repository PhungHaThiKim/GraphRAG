import pandas as pd
import json
from typing import List, Tuple


class ConceptNet:
    def __init__(self, path: str, max_edges: int = None):
        self.path = path
        self.max_edges = max_edges
        self.df: pd.DataFrame = None

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.path,
            sep='\t',
            header=None,
            usecols=[1, 2, 3, 4],
            names=['relation', 'start', 'end', 'meta'],
            engine='c',
            on_bad_lines='skip'
        )

        # Chỉ giữ tiếng Anh
        df = df[df['start'].str.startswith('/c/en/') & df['end'].str.startswith('/c/en/')]

        # Extract head, tail
        df['head'] = df['start'].str.extract(r'^/c/en/([^/]+)')[0].str.replace('_', ' ', regex=False)
        df['tail'] = df['end'].str.extract(r'^/c/en/([^/]+)')[0].str.replace('_', ' ', regex=False)

        # Extract relation name
        df['relation'] = df['relation'].str.extract(r'^/r/([^/]+)')[0]

        # ✅ Extract weight from JSON string in 'meta'
        def extract_weight(meta_str):
            try:
                meta = json.loads(meta_str)
                return float(meta.get('weight', 1.0))
            except:
                return 1.0

        df['weight'] = df['meta'].apply(extract_weight)

        # Clean up
        df = df.dropna(subset=['head', 'tail', 'relation'])
        df['head'] = df['head'].str.lower().str.strip()
        df['tail'] = df['tail'].str.lower().str.strip()

        # Sample if needed
        if self.max_edges:
            df = df.sample(n=self.max_edges, random_state=42)

        self.df = df[['relation', 'head', 'tail', 'weight']]
        return self.df

    def get_edges(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("You must call .load() before accessing edges.")
        return self.df

    def get_triples(self) -> List[Tuple[str, str, str]]:
        if self.df is None:
            raise ValueError("You must call .load() before accessing triples.")
        return list(self.df[['head', 'relation', 'tail']].itertuples(index=False, name=None))
