"""Re-score the published run from raw model outputs and fixed gold data."""

import hashlib
import json
import unittest
from pathlib import Path

from finance_lora.evaluate import evaluate, read_jsonl
from finance_lora.experiment import CONDITIONS, SPLITS


class PublishedResultsTests(unittest.TestCase):
    def test_published_scores_match_predictions_and_data(self):
        root = Path(__file__).resolve().parents[1]
        results = root / "results"
        run = json.loads((results / "experiment.json").read_text(encoding="utf-8"))
        self.assertEqual(set(run["conditions"]), {name for name, _, _ in CONDITIONS})
        for name, digest in run["plan"]["data_sha256"].items():
            path = root / "data" / "synthetic-v1" / f"{name}.jsonl"
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
        for tag, outcome in run["conditions"].items():
            combined = hashlib.sha256()
            for split in SPLITS:
                with self.subTest(condition=tag, split=split):
                    gold = read_jsonl(root / "data" / "synthetic-v1" / f"{split}.jsonl")
                    raw = results / "predictions" / f"{tag}-{split}.jsonl"
                    combined.update(raw.read_bytes())
                    self.assertEqual(evaluate(gold, read_jsonl(raw))[0], outcome["metrics"][split])
            self.assertEqual(combined.hexdigest(), outcome["prediction_sha256"])
