# -*- coding: utf-8 -*-
"""
Tests for the Wells and Coppersmith (1993) surface rupture probability model.

Golden-truth values are pinned to the oq-pfdha reference implementation
(openquake.fdha.primary_surf_rup.wells_coppersmith1993.WC1993PrimarySR,
logistic fx = -12.51 + 2.053*mag, all faulting styles).
"""
import unittest

import numpy as np

from openquake.hazardlib.contexts import RuptureContext
from openquake.fdha.primary_surf_rup.wells_coppersmith1993 import (
    WC1993PrimarySR,
)


class WC1993PrimarySRTestCase(unittest.TestCase):
    """Tests for WC1993PrimarySR."""

    def setUp(self):
        self.model = WC1993PrimarySR()

    def test_golden_truth(self):
        mags = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
        # Pinned to the oq-pfdha reference implementation.
        expected = np.array([
            0.09578162809632779, 0.2282005304310663,
            0.45214691443837246, 0.6973055895562362,
            0.8654134636152279, 0.9472250468090221,
        ])
        got = self.model.get_prob(RuptureContext([('mag', mags)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_scalar(self):
        got = self.model.get_prob(RuptureContext([('mag', 6.0)]))
        self.assertIsInstance(got, float)
        np.testing.assert_allclose(got, 0.45214691443837246, rtol=1e-10)
