# -*- coding: utf-8 -*-
"""
Tests for the Takao et al. (2013) surface rupture probability model.

Golden-truth values are pinned to the oq-pfdha reference implementation
(openquake.fdha.primary_surf_rup.takao2013.Takao2013PrimarySR,
logistic fx = -32.03 + 4.9*mag).
"""
import unittest

import numpy as np

from openquake.hazardlib.contexts import RuptureContext
from openquake.fdha.primary_surf_rup.takao2013 import Takao2013PrimarySR


class Takao2013PrimarySRTestCase(unittest.TestCase):
    """Tests for Takao2013PrimarySR."""

    def setUp(self):
        self.model = Takao2013PrimarySR()

    def test_golden_truth(self):
        mags = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
        # Pinned to the oq-pfdha reference implementation.
        expected = np.array([
            0.0005364503232741418, 0.006181460891611156,
            0.0672324505878501, 0.45512110762642,
            0.9063617877622646, 0.9911635995393923,
        ])
        got = self.model.get_prob(RuptureContext([('mag', mags)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_scalar(self):
        got = self.model.get_prob(RuptureContext([('mag', 6.0)]))
        self.assertIsInstance(got, float)
        np.testing.assert_allclose(got, 0.0672324505878501, rtol=1e-10)
