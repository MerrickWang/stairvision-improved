"""Small checks for label geometry and invalid-depth masking."""
import unittest
import numpy as np
import torch
from prepare_data import line_to_pose
from depth_model import masked_loss


class CoreTests(unittest.TestCase):
    def test_horizontal_border_line(self):
        row = line_to_pose([0, 512, 100, 0, 100], 512, 512, 'pixels')
        self.assertEqual(len(row), 11)
        self.assertGreater(row[4], 0)
        self.assertEqual(row[5], 0)
        self.assertEqual(row[8], 1)
        self.assertTrue(np.all((np.array(row[1:7]) >= 0)))

    def test_bad_endpoint_rejected(self):
        with self.assertRaises(ValueError):
            line_to_pose([0, -10, 0, 100, 100], 512, 512, 'pixels')

    def test_invalid_pixels_have_no_gradient(self):
        pred = torch.full((1, 8, 8), .5, requires_grad=True)
        target = torch.zeros_like(pred)
        mask = torch.ones_like(pred, dtype=torch.bool); mask[:, 0, :] = False
        loss, _ = masked_loss(pred, target, mask); loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertEqual(pred.grad[:, 0, :].abs().sum().item(), 0)
        self.assertGreater(pred.grad[:, 1:, :].abs().sum().item(), 0)

    def test_empty_depth_rejected(self):
        with self.assertRaises(ValueError):
            masked_loss(torch.zeros(1, 8, 8), torch.zeros(1, 8, 8), torch.zeros(1, 8, 8, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
