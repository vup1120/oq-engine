# -*- coding: utf-8 -*-
"""
Tests for the Moss and Ross (2011) surface rupture probability model.

Golden-truth values are pinned to the oq-pfdha reference implementation
(openquake.fdha.primary_surf_rup.moss_ross2011.MossRoss2011PrimarySR,
logistic P = 1 / (1 + exp(7.3 - 1.03*mag))).
"""
import unittest

import numpy as np

from openquake.hazardlib.contexts import RuptureContext
from openquake.fdha.primary_surf_rup.moss_ross2011 import (
    MossRoss2011PrimarySR,
)


class MossRoss2011PrimarySRTestCase(unittest.TestCase):
    """Tests for MossRoss2011PrimarySR."""

    def setUp(self):
        self.model = MossRoss2011PrimarySR()

    def test_golden_truth(self):
        mags = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])
        # Pinned to the oq-pfdha reference implementation.
        expected = np.array([
            0.10433122311900135, 0.16314656214056913,
            0.2460112835510519, 0.3532006074420145,
            0.4775151752081999, 0.6046790847140094,
        ])
        got = self.model.get_prob(RuptureContext([('mag', mags)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_scalar(self):
        got = self.model.get_prob(RuptureContext([('mag', 6.0)]))
        self.assertIsInstance(got, float)
        np.testing.assert_allclose(got, 0.2460112835510519, rtol=1e-10)
