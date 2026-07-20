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
Tests for :class:`openquake.hazardlib.scalerel.leonard2010
.Leonard2010_Interplate`, the interplate dip-slip scaling used for fault
displacement hazard analysis.

Golden values are pinned to the oq-pfdha reference implementation
(openquake.fdha.scalerel.leonard2010.Leonard2010).
"""
import numpy

from openquake.hazardlib.scalerel.leonard2010 import Leonard2010_Interplate
from openquake.hazardlib.tests.scalerel.msr_test import BaseMSRTestCase


class Leonard2010_InterplateTestCase(BaseMSRTestCase):

    MSR_CLASS = Leonard2010_Interplate

    def test_median_length(self):
        # bilinear relation: log10(L) = 0.5m - 1.9 for m <= 7.1,
        # else log10(L) = m - 4.7
        for mag, expected in [(5.0, 3.981072), (6.0, 12.589254),
                              (7.0, 39.810717), (7.1, 44.668359),
                              (7.5, 630.957344), (8.0, 1995.262315)]:
            numpy.testing.assert_allclose(
                self.msr.get_median_length(mag), expected, rtol=1e-6)
        self.assertEqual(self.msr.get_std_dev_length(7.0), 0.23)

    def test_median_width(self):
        # W = 1.95 * L^(2/3), capped at 20 km
        numpy.testing.assert_allclose(
            self.msr.get_median_width(6.0), 10.552806, rtol=1e-6)
        numpy.testing.assert_allclose(
            self.msr.get_median_width(7.5), 20.0, rtol=1e-12)

    def test_median_area(self):
        for mag, expected in [(6.0, 132.851953), (7.0, 796.214341),
                              (7.5, 12619.146890)]:
            numpy.testing.assert_allclose(
                self.msr.get_median_area(mag, 90), expected, rtol=1e-6)
        self.assertEqual(self.msr.get_std_dev_area(7.0, 90), 0.23)

    def test_median_magnitude(self):
        for area, expected in [(100.0, 5.851958), (2000.0, 7.413194),
                               (5000.0, 7.097940)]:
            numpy.testing.assert_allclose(
                self.msr.get_median_mag(area, 90), expected, rtol=1e-6)
        self.assertEqual(self.msr.get_std_dev_mag(1000.0, 90), 0.23)

    def test_average_displacement(self):
        # AD = 1.7e-5 * L with L in metres
        for mag, expected in [(5.0, 0.06767822), (6.0, 0.21401732),
                              (7.0, 0.67678219), (7.5, 10.72627486)]:
            numpy.testing.assert_allclose(
                self.msr.get_average_displacement(mag), expected, rtol=1e-6)
        self.assertEqual(self.msr.get_std_dev_displacement(7.0), 0.23)
