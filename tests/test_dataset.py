import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from finance_lora.dataset import generate_dataset
from finance_lora.evaluate import read_jsonl, validate_target


class DatasetTests(unittest.TestCase):
    def test_shipped_dataset_and_split_boundaries(self):
        root = Path(__file__).resolve().parents[1] / 'data' / 'synthetic-v1'
        manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        companies, texts, ids = set(), set(), set()
        train_templates, stress_templates = set(), set()
        for split, expected in {'train': 1024, 'validation': 64, 'test': 128, 'stress': 48}.items():
            details = manifest['splits'][split]
            path = root / details['file']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), details['sha256'])
            rows = read_jsonl(path)
            self.assertEqual(len(rows), expected)
            for row in rows:
                validate_target(row['target'])
                self.assertNotIn(row['company_id'], companies)
                self.assertNotIn(row['id'], ids)
                self.assertNotIn(row['text'], texts)
                companies.add(row['company_id'])
                ids.add(row['id'])
                texts.add(row['text'])
            if split == 'train':
                train_templates = {row['template_id'] for row in rows}
            if split == 'stress':
                stress_templates = {row['template_id'] for row in rows}
        self.assertTrue(train_templates)
        self.assertTrue(stress_templates)
        self.assertFalse(train_templates & stress_templates)

    def test_generator_reproduces_published_hashes(self):
        root = Path(__file__).resolve().parents[1] / 'data' / 'synthetic-v1'
        published = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as temporary:
            manifest = generate_dataset(Path(temporary), seed=published['seed'])
            self.assertEqual(manifest, published)
