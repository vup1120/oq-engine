# The Hazard Library
# Copyright (C) 2012-2026 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Tests for the Wells and Coppersmith (1994) displacement relations
(Table 2B) used for fault displacement hazard analysis.

Golden values are pinned to the oq-pfdha reference implementation
(openquake.fdha.scalerel.wc1994), which is itself validated against the
published paper coefficients.
"""
import unittest

from openquake.hazardlib.scalerel.wc1994 import WC1994


class WC1994AverageDisplacementTestCase(unittest.TestCase):
    """
    Tests for the average displacement (AD) relations of WC1994.
    Rakes: 0 strike-slip, -90 normal, 90 reverse (falls back to "All"),
    None "All".
    """

    def setUp(self):
        self.msr = WC1994()

    def test_strike_slip(self):
        for mag, expected in [(6.0, 0.120226), (7.0, 0.954993),
                              (7.5, 2.691535)]:
            self.assertAlmostEqual(
                self.msr.get_average_displacement(mag, 0.0), expected,
                places=5)
        ad, sigma = self.msr.get_average_displacement(
            7.0, 0.0, return_sigma=True)
        self.assertAlmostEqual(ad, 0.954993, places=5)
        self.assertAlmostEqual(sigma, 0.28)

    def test_normal(self):
        for mag, expected in [(6.0, 0.213796), (7.0, 0.912011),
                              (7.5, 1.883649)]:
            self.assertAlmostEqual(
                self.msr.get_average_displacement(mag, -90.0), expected,
                places=5)
        _, sigma = self.msr.get_average_displacement(
            7.0, -90.0, return_sigma=True)
        self.assertAlmostEqual(sigma, 0.33)

    def test_reverse_uses_all(self):
        # reverse regressions are not significant at the 95% level;
        # the "All" coefficients apply
        for rake in (90.0, None):
            for mag, expected in [(6.0, 0.218776), (7.0, 1.071519),
                                  (7.5, 2.371374)]:
                self.assertAlmostEqual(
                    self.msr.get_average_displacement(mag, rake), expected,
                    places=5)
            _, sigma = self.msr.get_average_displacement(
                7.0, rake, return_sigma=True)
            self.assertAlmostEqual(sigma, 0.36)

    def test_inverse(self):
        self.assertAlmostEqual(
            self.msr.get_median_mag_from_ad(2.691535, 0.0), 7.4227,
            places=4)
        self.assertAlmostEqual(
            self.msr.get_median_mag_from_ad(1.883649, -90.0), 6.9587,
            places=4)
        self.assertAlmostEqual(
            self.msr.get_median_mag_from_ad(2.371374, None), 7.2375,
            places=4)
        self.assertAlmostEqual(
            self.msr.get_std_dev_mag_from_ad(1.0, 0.0), 0.28)
        self.assertAlmostEqual(
            self.msr.get_std_dev_mag_from_ad(1.0, -90.0), 0.33)
        self.assertAlmostEqual(
            self.msr.get_std_dev_mag_from_ad(1.0, 90.0), 0.39)


class WC1994MaximumDisplacementTestCase(unittest.TestCase):
    """
    Tests for the maximum displacement (MD) relations of WC1994.
    """

    def setUp(self):
        self.msr = WC1994()

    def test_strike_slip(self):
        for mag, expected in [(6.0, 0.141254), (7.0, 1.513561),
                              (7.5, 4.954502)]:
            self.assertAlmostEqual(
                self.msr.get_maximum_displacement(mag, 0.0), expected,
                places=5)
        md, sigma = self.msr.get_maximum_displacement(
            7.0, 0.0, return_sigma=True)
        self.assertAlmostEqual(md, 1.513561, places=5)
        self.assertAlmostEqual(sigma, 0.34)

    def test_normal(self):
        for mag, expected in [(6.0, 0.275423), (7.0, 2.137962),
                              (7.5, 5.956621)]:
            self.assertAlmostEqual(
                self.msr.get_maximum_displacement(mag, -90.0), expected,
                places=5)
        _, sigma = self.msr.get_maximum_displacement(
            7.0, -90.0, return_sigma=True)
        self.assertAlmostEqual(sigma, 0.38)

    def test_reverse_uses_all(self):
        for rake in (90.0, None):
            for mag, expected in [(6.0, 0.288403), (7.0, 1.905461),
                                  (7.5, 4.897788)]:
                self.assertAlmostEqual(
                    self.msr.get_maximum_displacement(mag, rake), expected,
                    places=5)
            _, sigma = self.msr.get_maximum_displacement(
                7.0, rake, return_sigma=True)
            self.assertAlmostEqual(sigma, 0.42)

    def test_inverse(self):
        self.assertAlmostEqual(
            self.msr.get_median_mag_from_md(4.954502, 0.0), 7.3521,
            places=4)
        self.assertAlmostEqual(
            self.msr.get_median_mag_from_md(5.956621, -90.0), 7.1602,
            places=4)
        self.assertAlmostEqual(
            self.msr.get_median_mag_from_md(4.897788, None), 7.2006,
            places=4)
        self.assertAlmostEqual(
            self.msr.get_std_dev_mag_from_md(1.0, 0.0), 0.29)
        self.assertAlmostEqual(
            self.msr.get_std_dev_mag_from_md(1.0, -90.0), 0.34)
        self.assertAlmostEqual(
            self.msr.get_std_dev_mag_from_md(1.0, 90.0), 0.40)
