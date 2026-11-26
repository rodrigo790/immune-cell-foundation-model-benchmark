import numpy as np
import unittest
from analysis import folds


class SplitTest(unittest.TestCase):
    def test_each_cell_once(self):
        d = np.array(["a", "b", "a", "c"])
        seen = []
        for _, tr, te in folds(d):
            self.assertFalse(set(d[tr]) & set(d[te]))
            seen.extend(te)
        self.assertEqual(sorted(seen), list(range(4)))
