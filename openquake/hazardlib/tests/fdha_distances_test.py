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
Tests for the FDHA distance parameters `rtor` and `x_l` and the FDHA
rupture parameter `length` through the engine distance dispatch
(get_distances) and the ContextMaker.
"""
import unittest

import numpy

from openquake.hazardlib import const
from openquake.hazardlib.geo import Line, Point
from openquake.hazardlib.geo.surface.simple_fault import SimpleFaultSurface
from openquake.hazardlib.gsim.base import GMPE
from openquake.hazardlib.contexts import simple_cmaker, get_distances
from openquake.hazardlib.source.rupture import BaseRupture
from openquake.hazardlib.site import Site, SiteCollection

aac = numpy.testing.assert_allclose


class FDHAFakeGSIM(GMPE):
    """
    Fake GSIM-like model declaring the FDHA requirements, used to drive
    the ContextMaker.
    """
    DEFINED_FOR_TECTONIC_REGION_TYPE = const.TRT.ACTIVE_SHALLOW_CRUST
    DEFINED_FOR_INTENSITY_MEASURE_TYPES = set()
    DEFINED_FOR_INTENSITY_MEASURE_COMPONENT = const.IMC.GEOMETRIC_MEAN
    DEFINED_FOR_STANDARD_DEVIATION_TYPES = {const.StdDev.TOTAL}
    REQUIRES_SITES_PARAMETERS = set()
    REQUIRES_RUPTURE_PARAMETERS = {'mag', 'rake', 'length'}
    REQUIRES_DISTANCES = {'rtor', 'x_l', 'rx'}

    def compute(self, ctx: numpy.recarray, imts, mean, sig, tau, phi):
        pass


def _make_rupture():
    trace = Line([Point(0.0, 0.0), Point(0.0, 1.0)])
    surface = SimpleFaultSurface.from_fault_data(
        trace, upper_seismogenic_depth=0.0, lower_seismogenic_depth=15.0,
        dip=45.0, mesh_spacing=1.0)
    rupture = BaseRupture(
        mag=7.0, rake=90.0, tectonic_region_type='*',
        hypocenter=Point(0.0, 0.5, 7.0), surface=surface)
    rupture.occurrence_rate = 1.0
    return rupture


class GetDistancesTestCase(unittest.TestCase):
    """Test the get_distances dispatch for rtor and x_l."""

    def setUp(self):
        self.rup = _make_rupture()
        self.sites = SiteCollection([
            Site(Point(0.1, 0.5), vs30=760.,
                 z1pt0=100., z2pt5=5.),   # hanging wall side
            Site(Point(-0.1, 0.2), vs30=760.,
                 z1pt0=100., z2pt5=5.),   # footwall side
            Site(Point(0.0, 1.3), vs30=760.,
                 z1pt0=100., z2pt5=5.)])  # off the north end

    def test_rtor(self):
        rtor = get_distances(self.rup, self.sites, 'rtor')
        expected = self.rup.surface.get_tor_distance(self.sites.mesh)
        aac(rtor, expected)
        # hanging-wall site: above the dipping plane, rjb = 0, rtor > 0
        rjb = get_distances(self.rup, self.sites, 'rjb')
        self.assertAlmostEqual(rjb[0], 0.0, delta=0.01)
        self.assertGreater(rtor[0], 10.0)

    def test_x_l(self):
        x_l = get_distances(self.rup, self.sites, 'x_l')
        expected, l_km = self.rup.surface.get_x_l_ratio(self.sites.mesh)
        aac(x_l, expected)
        # site at mid-trace, site at 20% of the trace, site beyond the
        # north end (clipped to 1)
        self.assertAlmostEqual(x_l[0], 0.5, delta=0.02)
        self.assertAlmostEqual(x_l[1], 0.2, delta=0.02)
        self.assertAlmostEqual(x_l[2], 1.0, delta=0.001)
        self.assertAlmostEqual(l_km, 111.2, delta=0.7)


class ContextMakerFDHATestCase(unittest.TestCase):
    """Build contexts with a fake FDHA model and check rtor/x_l/length."""

    def test_ctx(self):
        rup = _make_rupture()
        sites = SiteCollection([
            Site(Point(0.1, 0.5), vs30=760., z1pt0=100., z2pt5=5.),
            Site(Point(-0.1, 0.2), vs30=760., z1pt0=100., z2pt5=5.)])
        cmaker = simple_cmaker([FDHAFakeGSIM()], ['PGA'])
        [ctx] = cmaker.get_ctxs([rup], sites)
        mesh = sites.mesh
        aac(ctx.rtor, rup.surface.get_tor_distance(mesh))
        x_l, l_km = rup.surface.get_x_l_ratio(mesh)
        aac(ctx.x_l, x_l)
        aac(ctx.length, rup.surface.get_tor_length())
        self.assertAlmostEqual(l_km, rup.surface.get_tor_length())
        aac(ctx.mag, 7.0)
        aac(ctx.rake, 90.0)
