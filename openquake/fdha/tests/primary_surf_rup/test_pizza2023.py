# -*- coding: utf-8 -*-
"""
Tests for the Pizza et al. (2023) surface rupture probability models.

Golden-truth values are pinned to the oq-pfdha reference implementation
(openquake.fdha.primary_surf_rup.pizza2023.Pizza2023PrimarySR called with
the matching ``style`` argument), logistic fx = a + b*mag.
"""
import unittest

import numpy as np

from openquake.hazardlib.contexts import RuptureContext
from openquake.fdha.primary_surf_rup.pizza2023 import (
    Pizza2023PrimarySR,
    Pizza2023PrimarySR_Normal,
    Pizza2023PrimarySR_Reverse,
    Pizza2023PrimarySR_SS,
)

MAGS = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])


class Pizza2023AllTestCase(unittest.TestCase):
    """Tests for Pizza2023PrimarySR (all faulting styles, a=-14.47, b=2.177)."""

    def setUp(self):
        self.model = Pizza2023PrimarySR()

    def test_golden_truth(self):
        # Pinned to the oq-pfdha reference (style='all').
        expected = np.array([
            0.02698810746349781, 0.07610390755994374,
            0.19654970052743376, 0.4207976065133961,
            0.683304533966245, 0.8650052864928219,
        ])
        got = self.model.get_prob(RuptureContext([('mag', MAGS)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)


class Pizza2023NormalTestCase(unittest.TestCase):
    """Tests for Pizza2023PrimarySR_Normal (a=-13.5, b=2.159)."""

    def setUp(self):
        self.model = Pizza2023PrimarySR_Normal()

    def test_golden_truth(self):
        # Pinned to the oq-pfdha reference (style='normal').
        expected = np.array([
            0.06267896139944466, 0.1644477490260145,
            0.366792936518415, 0.630299060786877,
            0.8338274804632062, 0.9365826328455757,
        ])
        got = self.model.get_prob(RuptureContext([('mag', MAGS)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)


class Pizza2023ReverseTestCase(unittest.TestCase):
    """Tests for Pizza2023PrimarySR_Reverse (a=-10.75, b=1.427)."""

    def setUp(self):
        self.model = Pizza2023PrimarySR_Reverse()

    def test_golden_truth(self):
        # Pinned to the oq-pfdha reference (style='reverse').
        expected = np.array([
            0.026211395056649066, 0.05207946251741497,
            0.10083328071638166, 0.18625959949417045,
            0.318429194772539, 0.4881272322437473,
        ])
        got = self.model.get_prob(RuptureContext([('mag', MAGS)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)


class Pizza2023SSTestCase(unittest.TestCase):
    """Tests for Pizza2023PrimarySR_SS (strike-slip, a=-28.56, b=4.436)."""

    def setUp(self):
        self.model = Pizza2023PrimarySR_SS()

    def test_golden_truth(self):
        # Pinned to the oq-pfdha reference (style='strike-slip').
        expected = np.array([
            0.0016922543773797093, 0.015337471801488244,
            0.12520907224904, 0.5680746343565887,
            0.9235790838081033, 0.9910755847915687,
        ])
        got = self.model.get_prob(RuptureContext([('mag', MAGS)]))
        np.testing.assert_allclose(got, expected, rtol=1e-10)
