"""Integration checks using tiny CPU models and temporary run directories."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import numpy as np
from PIL import Image

import train
from checkpoint_selection import AP_EVALUATION


class TinyDetector(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_classes = kwargs.get('num_classes', 1)
        self.backbone = torch.nn.Linear(1, 1)
        self.head = torch.nn.Linear(1, 1)
        self.backbone.out_strides = (8, 16, 32)
        self.encoder = torch.nn.Identity()
        self.encoder.out_strides = (8, 16, 32)

    def forward(self, images):
        if self.training:
            raise AssertionError('AP must run in inference/eval mode')
        batch = len(images)
        return {
            'pred_logits': torch.full((batch, 1, 1), 10.0),
            'pred_boxes': torch.tensor([.5, .5, .25, .25]).repeat(batch, 1, 1),
        }


class APIntegrationTests(unittest.TestCase):
    def test_shared_evaluator_perfect_detection_and_empty_labels(self):
        model = TinyDetector()
        target = {
            'labels': torch.tensor([0]),
            'boxes': torch.tensor([[.5, .5, .25, .25]]),
            'orig_size': torch.tensor([64, 64]),
        }
        loader = [(torch.zeros(1, 3, 64, 64), [target])]
        with contextlib.redirect_stdout(io.StringIO()):
            metrics = train.run_ap_validation(model, loader, torch.device('cpu'), False)
        self.assertEqual(metrics, {'map50': 1.0, 'map50_95': 1.0})
        target['labels'] = torch.empty(0, dtype=torch.long)
        target['boxes'] = torch.empty(0, 4)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'undefined'):
            train.run_ap_validation(model, loader, torch.device('cpu'), False)

    def make_args(self, root, output, epochs, resume=''):
        argv = [
            'train.py', '--root', str(root), '--output-dir', str(output),
            '--epochs', str(epochs), '--num-classes', '1', '--workers', '0',
            '--batch-size', '1', '--img-size', '64', '--device', 'cpu', '--no-amp',
            '--num-queries', '4', '--hidden-dim', '32', '--decoder-layers', '1',
            '--num-denoising', '4',
        ]
        if resume:
            argv += ['--resume', str(resume)]
        with patch('sys.argv', argv):
            return train.parse_args()

    def make_dataset(self, root):
        for split in ('train', 'val'):
            (root / 'images' / split).mkdir(parents=True)
            (root / 'labels' / split).mkdir(parents=True)
            pixels = np.random.default_rng(123).integers(0, 256, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / 'images' / split / 'one.png')
            (root / 'labels' / split / 'one.txt').write_text('0 .5 .5 .25 .25\n')

    def run_epochs(self, args, losses, aps):
        def training_epoch(model, loader, criterion, optimizer, *rest):
            # Real parameter/optimizer state gives save/load tests actual tensors.
            optimizer.zero_grad()
            sum(p.sum() for p in model.parameters()).backward()
            optimizer.step()
            return {'loss': 2.0}

        with patch.object(train, 'RTDETR', TinyDetector), \
             patch.object(train, 'run_training_epoch', side_effect=training_epoch), \
             patch.object(train, 'run_validation_epoch', side_effect=[{'loss': v} for v in losses]), \
             patch.object(train, 'run_ap_validation', side_effect=[{'map50': v, 'map50_95': v} for v in aps]), \
             contextlib.redirect_stdout(io.StringIO()):
            train.train(args)

    def test_training_selects_ap_and_resume_keeps_prior_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'data'
            output = Path(temp) / 'run'
            self.make_dataset(root)
            self.run_epochs(self.make_args(root, output, 3), [3., 4., 1.], [.2, .6, .4])
            best = torch.load(output / 'best.pt', weights_only=True)
            last = torch.load(output / 'last.pt', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 2)
            self.assertEqual(best['validation_metrics']['loss'], 4.)
            self.assertEqual(last['best_validation_loss'], 1.)
            self.assertEqual(last['best_validation_ap'], .6)
            self.assertEqual(last['ap_evaluation'], AP_EVALUATION)
            before = (output / 'best.pt').read_bytes()
            self.run_epochs(self.make_args(root, output, 5, output / 'last.pt'), [.5, .4], [.5, .6])
            self.assertEqual((output / 'best.pt').read_bytes(), before)
            history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
            self.assertEqual(history[-1]['best_ap_epoch'], 2)  # Human-readable epoch.
            self.assertEqual(history[-1]['validation']['map50_95'], .6)
            self.run_epochs(self.make_args(root, output, 6, output / 'last.pt'), [.3], [.7])
            self.assertEqual(torch.load(output / 'best.pt', weights_only=True)['epoch'], 5)

    def test_legacy_resume_preserves_file_and_baselines_resumed_weights(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'data'
            output = Path(temp) / 'run'
            self.make_dataset(root)
            self.run_epochs(self.make_args(root, output, 1), [2.], [.6])
            old = torch.load(output / 'last.pt', weights_only=True)
            for key in ('checkpoint_metric', 'ap_evaluation', 'best_validation_ap', 'best_ap_epoch'):
                del old[key]
            torch.save(old, output / 'last.pt')
            torch.save(old, output / 'best.pt')
            previous = (output / 'best.pt').read_bytes()
            # First value is the legacy baseline, second is the next epoch.
            self.run_epochs(self.make_args(root, output, 2, output / 'last.pt'), [2., 1.], [.6, .4])
            self.assertEqual((output / 'best_before_ap_selection.pt').read_bytes(), previous)
            best = torch.load(output / 'best.pt', weights_only=True)
            self.assertEqual(best['epoch'], 0)
            self.assertEqual(best['best_validation_ap'], .6)
            self.assertEqual(torch.load(output / 'last.pt', weights_only=True)['epoch'], 1)

    def test_real_model_epoch_checkpoint_and_standalone_evaluation_agree(self):
        from validate import build_model_from_checkpoint

        old_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / 'data'
                output = Path(temp) / 'run'
                self.make_dataset(root)
                with contextlib.redirect_stdout(io.StringIO()):
                    train.train(self.make_args(root, output, 1))
                    checkpoint = torch.load(output / 'best.pt', weights_only=True)
                    model, _ = build_model_from_checkpoint(checkpoint, torch.device('cpu'))
                    dataset = train.YoloTxtDataset(str(root), 'val', 64, 1)
                    loader = torch.utils.data.DataLoader(dataset, collate_fn=train.collate_fn)
                    measured = train.run_ap_validation(model, loader, torch.device('cpu'), False)
                self.assertEqual(measured['map50_95'], checkpoint['best_validation_ap'])
                self.assertEqual(checkpoint['best_ap_epoch'], 0)
                self.assertTrue((output / 'last.pt').is_file())
        finally:
            torch.set_num_threads(old_threads)


if __name__ == '__main__':
    unittest.main()
