"""Focused tests for publication additions and retained thesis helpers."""
import importlib
import json
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'notebooks'))


class LegacyKLTest(unittest.TestCase):
    def test_legacy_estimator_excludes_empty_reference_bins_and_upper_tail(self):
        self.assertTrue((ROOT / 'notebooks/legacy_kl.py').exists(), 'Independent KL implementation is missing')
        compute = importlib.import_module('legacy_kl').computeKL
        # First bin P=1/2 Q=1/4; second bin Q=0 is ignored;
        # score >=.98 is excluded, but normalization uses all observations.
        left = np.array([.005, .005, .025, .99])
        right = np.array([.005, .045, .045, .99])
        self.assertAlmostEqual(compute(left, right), .5 * np.log(2))
        self.assertEqual(compute(left, left), 0.)


class ResultsOverviewTest(unittest.TestCase):
    def test_chart_uses_five_seed_sample_statistics_and_is_deterministic(self):
        import csv
        import hashlib
        import os
        import statistics
        import subprocess
        import tempfile
        script = ROOT / 'scripts/make_overview.py'
        self.assertTrue(script.exists(), 'Results overview script is missing')
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as temporary:
            out = Path(temporary)
            command = [sys.executable, str(script), '--output-dir', str(out)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            with (out / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 120)
            self.assertEqual({r['n_seeds'] for r in rows}, {'5'})
            with (ROOT / 'results/results.csv').open() as stream:
                source = list(csv.DictReader(stream))
            values = [float(r['mae_blind']) for r in source if r['fusion'] == 'early' and r['condition'] == 'none']
            summary = next(r for r in rows if r['fusion'] == 'early' and r['condition'] == 'none' and r['metric'] == 'mae_blind')
            self.assertAlmostEqual(float(summary['mean']), statistics.mean(values))
            self.assertAlmostEqual(float(summary['std']), statistics.stdev(values))
            png = out / 'results_overview.png'
            self.assertEqual(png.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            before = hashlib.sha256(png.read_bytes()).hexdigest()
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(hashlib.sha256(png.read_bytes()).hexdigest(), before)


class RetainedThesisTest(unittest.TestCase):
    def test_notebooks_are_clean_valid_and_code_compiles(self):
        import nbformat
        notebooks = sorted((ROOT / 'notebooks').rglob('*.ipynb'))
        self.assertEqual(len(notebooks), 7)
        for path in notebooks:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            self.assertFalse(notebook.metadata)
            for idx, cell in enumerate(notebook.cells):
                self.assertFalse(cell.metadata)
                if cell.cell_type == 'code':
                    self.assertEqual(cell.outputs, [])
                    self.assertIsNone(cell.execution_count)
                    compile(cell.source, f'{path.name}:{idx}', 'exec')

    def test_helper_metrics_and_model_forward_smoke(self):
        import ast
        import torch
        import torch.nn as nn
        import helpers
        gender = np.tile([0., 1.], 100)
        scores = np.linspace(.1, .9, 200)
        parity, female_share = helpers.computeDemographicParity(scores, gender)
        self.assertEqual(parity, 1.)
        self.assertEqual(female_share, .5)
        self.assertEqual(helpers.computeMAE(scores, scores), 0.)
        self.assertAlmostEqual(helpers.computeNDCG(scores, scores), 1.)
        gap, selected = helpers.computeEqualityOfOpportunity(scores, gender, scores, scores)
        self.assertEqual(gap, 0.)
        self.assertEqual(selected, 50.)
        for filename, classname, dimensions in [('early_fusion.ipynb', 'EarlyFusion', (9,)),
                                                ('mid_fusion.ipynb', 'MidFusion', (2, 4, 3, 5)),
                                                ('late_fusion.ipynb', 'LateFusion', (9,))]:
            data = json.loads((ROOT / 'notebooks' / filename).read_text())
            code = '\n'.join(''.join(c['source']) for c in data['cells'] if c['cell_type'] == 'code')
            classes = [node for node in ast.parse(code).body if isinstance(node, ast.ClassDef)]
            namespace = {'torch': torch, 'nn': nn}
            exec(compile(ast.Module(body=classes, type_ignores=[]), filename, 'exec'), namespace)
            model = namespace[classname](*dimensions).eval()
            inputs = [torch.ones(4, dim) for dim in (dimensions[:3] if classname == 'MidFusion' else dimensions)]
            with torch.no_grad():
                prediction = model(*inputs)
            self.assertEqual(tuple(prediction.shape), (4, 1))
            self.assertTrue(torch.isfinite(prediction).all())
            self.assertTrue(((prediction >= 0) & (prediction <= 1)).all())
        previous = helpers.DEVICE
        helpers.DEVICE = torch.device('cpu')
        try:
            inputs = np.ones((8, 3), dtype=np.float32)
            predictions, model = helpers.train_and_evaluate(lambda: nn.Sequential(nn.Linear(3, 1), nn.Sigmoid()),
                                                            [inputs], [inputs[:4]], np.full(8, .5),
                                                            seed=32, epochs=1, batch_size=4)
            self.assertEqual(predictions.shape, (4,))
            self.assertTrue(np.isfinite(predictions).all())
        finally:
            helpers.DEVICE = previous

    def test_all_feature_conditions_with_explicit_synthetic_fixture(self):
        import helpers
        import tempfile
        with tempfile.TemporaryDirectory(dir=ROOT.parent) as directory:
            folder = Path(directory)
            rng = np.random.default_rng(42)
            fixture = {'Profiles Train': rng.normal(size=(8, 51)), 'Profiles Test': rng.normal(size=(4, 51)),
                       'Biased Labels Train (Gender)': np.full(8, .4), 'Blind Labels Train': np.full(8, .5),
                       'Biased Labels Test (Gender)': np.full(4, .4), 'Blind Labels Test': np.full(4, .5)}
            np.save(folder / 'FairCVdb.npy', fixture)
            for variant in ('original', 'blinded'):
                np.save(folder / f'{variant}_bios_train_bert.npy', rng.normal(size=(8, 6)))
                np.save(folder / f'{variant}_bios_test_bert.npy', rng.normal(size=(4, 6)))
            previous = helpers.DATA_PATH
            helpers.DATA_PATH = directory
            try:
                for condition in ('none', 'tabular', 'tabular-text', 'tabular-image', 'tabular-image-text'):
                    result = helpers.build_features(condition)
                    self.assertEqual(len(result), 10)
                    self.assertEqual(result[4].shape, (8, 8 if condition == 'none' else 7))
                    self.assertEqual(result[6].shape, (8, 6))
                    self.assertEqual(result[8].shape, (8, 20))
                    self.assertTrue(all(np.isfinite(array).all() for array in result))
                self.assertEqual(helpers.load_gender_test().shape, (4,))
            finally:
                helpers.DATA_PATH = previous


if __name__ == '__main__':
    unittest.main()
