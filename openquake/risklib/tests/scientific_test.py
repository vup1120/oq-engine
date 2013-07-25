# -*- coding: utf-8 -*-

# Copyright (c) 2013, GEM Foundation.
#
# OpenQuake Risklib is free software: you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License
# as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# OpenQuake Risklib is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public
# License along with OpenQuake Risklib. If not, see
# <http://www.gnu.org/licenses/>.

import unittest
import numpy
from openquake.risklib import DegenerateDistribution


class DegenerateDistributionTest(unittest.TestCase):
    def setUp(self):
        self.distribution = DegenerateDistribution()

    def test_survival_zero_mean(self):
        self.assertEqual(
            0, self.distribution.survival(numpy.random.random(), 0, None))

    def test_survival_nonzeromean(self):
        loss_ratio = numpy.random.random()
        mean = loss_ratio - numpy.random.random()

        self.assertEqual(
            0, self.distribution.survival(loss_ratio, mean, None))

        mean = loss_ratio + numpy.random.random()
        self.assertEqual(
            1, self.distribution.survival(loss_ratio, mean, None))
