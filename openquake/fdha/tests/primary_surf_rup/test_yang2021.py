# -*- coding: utf-8 -*-
"""
Tests for the Yang et al. (2021) surface rupture probability model.

Golden-truth values are pinned to the oq-pfdha reference implementation
(openquake.fdha.primary_surf_rup.yang2021.Yang2021PrimarySR,
logistic fx = -24.59 + 4.0*mag; regression valid for 4.0 <= Mw <= 6.6).
"""
import unittest

import numpy as np

from openquake.hazardlib.contexts import RuptureContext
from openquake.fdha.primary_surf_rup.yang2021 import Yang2021PrimarySR


class Yang2021PrimarySRTestCase(unittest.TestCase):
    """Tests for Yang2021PrimarySR."""

    def setUp(self):
        self.model = Yang2021PrimarySR()

    def test_golden_truth(self):
        mags = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
        # Pinned to the oq-pfdha reference implementation.
        expected = np.array([
            0.010050813883473756, 0.0697847828765801,
            0.35663485430559827, 0.8037659436342208,
            0.9680156025104667, 0.9955482663398963,
        ])
        got = self.model.get_prob(RuptureContext([('mag', mags)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_scalar(self):
        got = self.model.get_prob(RuptureContext([('mag', 6.0)]))
        self.assertIsInstance(got, float)
        np.testing.assert_allclose(got, 0.35663485430559827, rtol=1e-10)
