# -*- coding: utf-8 -*-
"""
Tests for the fixed primary surface rupture probability model
(:class:`openquake.fdha.primary_surf_rup.fixed.FixedPrimarySR`).
"""
import unittest

import numpy as np

from openquake.hazardlib.contexts import RuptureContext
from openquake.fdha.primary_surf_rup.fixed import FixedPrimarySR

MAGS = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])


class FixedPrimarySRTestCase(unittest.TestCase):
    """Tests for FixedPrimarySR."""

    def test_default_value(self):
        model = FixedPrimarySR()
        got = model.get_prob(RuptureContext([('mag', MAGS)]))
        np.testing.assert_allclose(got, np.ones_like(MAGS), rtol=1e-10)

    def test_custom_value_array(self):
        model = FixedPrimarySR(0.42)
        got = model.get_prob(RuptureContext([('mag', MAGS)]))
        expected = np.full_like(MAGS, 0.42)
        np.testing.assert_allclose(got, expected, rtol=1e-10)

    def test_custom_value_scalar(self):
        model = FixedPrimarySR(0.42)
        got = model.get_prob(RuptureContext([('mag', 6.0)]))
        self.assertIsInstance(got, float)
        np.testing.assert_allclose(got, 0.42, rtol=1e-10)

    def test_value_above_one_raises(self):
        with self.assertRaises(ValueError):
            FixedPrimarySR(1.5)

    def test_value_below_zero_raises(self):
        with self.assertRaises(ValueError):
            FixedPrimarySR(-0.1)
