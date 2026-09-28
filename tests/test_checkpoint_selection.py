import unittest

from checkpoint_selection import (
    AP_EVALUATION, CHECKPOINT_METRIC, is_ap_improvement, restore_ap_selection,
)


class SelectionPolicyTests(unittest.TestCase):
    def test_zero_initial_score_and_strict_improvement(self):
        self.assertTrue(is_ap_improvement(0.0, None))
        self.assertFalse(is_ap_improvement(0.0, 0.0))
        self.assertFalse(is_ap_improvement(0.4, 0.6))
        self.assertTrue(is_ap_improvement(0.7, 0.6))

    def test_invalid_ap_is_rejected(self):
        for score in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(score=score), self.assertRaises(ValueError):
                is_ap_improvement(score, None)

    def test_legacy_loss_is_not_interpreted_as_ap(self):
        self.assertEqual(restore_ap_selection({'best_validation_loss': 0.01}), (None, None))

    def test_resume_restores_best_not_latest_score(self):
        checkpoint = {
            'checkpoint_metric': CHECKPOINT_METRIC,
            'ap_evaluation': dict(AP_EVALUATION),
            'best_validation_ap': 0.6, 'best_ap_epoch': 1, 'epoch': 2,
            'validation_metrics': {'map50_95': 0.4},
        }
        self.assertEqual(restore_ap_selection(checkpoint), (0.6, 1))
        checkpoint['ap_evaluation'] = {**AP_EVALUATION, 'max_detections': 100}
        with self.assertRaises(ValueError):
            restore_ap_selection(checkpoint)


if __name__ == '__main__':
    unittest.main()
