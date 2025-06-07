import json
from typing import List, Dict

class CommonsenseQA:
    def __init__(self, filepath: str):
        """
        Load and parse CommonsenseQA dataset from a .jsonl file.

        Args:
            filepath (str): Path to the CommonsenseQA file (in JSONL format).
        """
        self.filepath = filepath
        self.samples = self._load()

    def _load(self) -> List[Dict]:
        """
        Internal method to load and parse the dataset.

        Returns:
            List[Dict]: List of parsed question samples.
        """
        samples = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                q = entry['question']
                answer_label = entry['answerKey']
                answer_text = next(c['text'] for c in q['choices'] if c['label'] == answer_label)

                samples.append({
                    "question_id": entry['id'],
                    "question_text": q['stem'],
                    "options": [c['text'] for c in q['choices']],
                    "correct_answer": answer_text,
                    "correct_answer_label": answer_label,
                    "source_entities": [q['question_concept'].lower()],
                    "target_entity": answer_text.lower()
                })
        return samples

    def get_all(self) -> List[Dict]:
        """
        Return all parsed question samples.

        Returns:
            List[Dict]: All question dictionaries.
        """
        return self.samples

    def get_question_by_id(self, question_id: str) -> Dict:
        """
        Retrieve a question sample by its ID.

        Args:
            question_id (str): ID of the question.

        Returns:
            Dict: The corresponding sample.
        """
        return next((s for s in self.samples if s['question_id'] == question_id), None)

    def get_source_target_pairs(self) -> List[Dict[str, str]]:
        """
        Extract (source_entity, target_entity) pairs for all questions.

        Returns:
            List[Dict[str, str]]: List of source-target pairs.
        """
        return [
            {"source": s["source_entities"][0], "target": s["target_entity"]}
            for s in self.samples
        ]
