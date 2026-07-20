# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 softtabstop=4
#
# Copyright (C) 2012-2026 GEM Foundation
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake. If not, see <http://www.gnu.org/licenses/>.

"""
Module :mod:`openquake.hazardlib.scalerel.leonard2010` implements
:class:`Leonard2010_SCR`
:class:`Leonard2010_SCR_M0`
:class:`Leonard2010_SCR_MX`
:class:`Leonard2010_Interplate`
"""
import math
import numpy
from numpy import power, log10
from openquake.hazardlib.scalerel.base import BaseMSRSigma, BaseASRSigma


class Leonard2010_SCR(BaseMSRSigma, BaseASRSigma):
    """
    Leonard, Mark. "Earthquake fault scaling: Self-consistent relating of rupture 
    length, width, average displacement, and moment release." Bulletin of the 
    Seismological Society of America 100.5A (2010): 1971-1988.

    Implements both magnitude-area and area-magnitude scaling relationships from 
    Table 6, but only for the category SCR
    """
    def get_median_area(self, mag, rake):
        """
        Calculates median fault area from magnitude.
        """
        #based on table 6 relationship for SCR
        return power(10.0, (mag - 4.19))

    def get_std_dev_area(self, mag, rake):
        """
        Returns zero for now
        """
        return 0.0

    def get_median_mag(self, area, rake):
        """
        Returns magnitude for a given fault area
        """
        #based on table 6 relationship for SCR
        return log10(area) + 4.19

    def get_std_dev_mag(self, area, rake):
        """
        Returns zero for now
        """
        return 0.0


class Leonard2010_SCR_M0(Leonard2010_SCR):
    """
    Leonard, Mark. "Earthquake fault scaling: Self-consistent relating of rupture 
    length, width, average displacement, and moment release." Bulletin of the 
    Seismological Society of America 100.5A (2010): 1971-1988.

    modifies Leonard2010_SCR for a term based on Table 5 and a more precise
    conversion between M0 and Mw
    """
    def get_median_area(self, mag, rake):
        """
        Calculates median fault area from magnitude.
        """
        #based on table 6 relationship for SCR with modification
        return power(10.0, (mag - 4.22))

    def get_median_mag(self, area, rake):
        """
        Returns magnitude for a given fault area
        """
        #based on table 6 relationship for SCR with modification
        return log10(area) + 4.22


class Leonard2010_SCR_MX(Leonard2010_SCR):
    """
    Modified for specific individual use. NOT RECOMMENDED!
    """
    def get_median_area(self, mag, rake):
        """
        Calculates median fault area from magnitude.
        """
        #based on table 6 relationship for SCR with modification
        return power(10.0, (mag - 4.00))

    def get_median_mag(self, area, rake):
        """
        Returns magnitude for a given fault area
        """
        #based on table 6 relationship for SCR with modification
        return log10(area) + 4.00


class Leonard2010_Interplate(BaseMSRSigma, BaseASRSigma):
    """
    Leonard, Mark. "Earthquake fault scaling: Self-consistent relating of
    rupture length, width, average displacement, and moment release."
    Bulletin of the Seismological Society of America 100.5A (2010):
    1971-1988.

    Self-consistent scaling for interplate dip-slip faults, as used for
    fault displacement hazard analysis (ported from the oq-pfdha tool,
    which is the validation reference): bilinear magnitude to rupture
    length, width derived from length and capped at 20 km, and average
    displacement AD = 1.7e-5 * L (L in metres).

    NB: the bilinear L(m) relation switches branch at m = 7.1, where the
    two branches do not join continuously (44.7 km below, 251 km above);
    the inverse relation switches at L = 99 km. This reproduces the
    reference oq-pfdha implementation exactly; tests are pinned to it.
    """

    #: Standard deviation of log10 rupture length
    SIGMA_L = 0.23

    def _width_from_length(self, length):
        # W = 1.95 * L^(2/3), capped at 20 km
        width = 1.95 * power(length, 2.0 / 3.0)
        return numpy.where(width > 20.0, 20.0, width)

    def get_median_area(self, mag, rake):
        """
        Calculates median fault area (in km^2) from magnitude as
        L(m) * W(L). The rake is ignored (dip-slip relation).
        """
        length = self.get_median_length(mag)
        return length * self._width_from_length(length)

    def get_std_dev_area(self, mag, rake):
        """
        Returns the standard deviation of log10 area (from length).
        """
        return self.SIGMA_L

    def get_median_mag(self, area, rake):
        """
        Returns magnitude for a given fault area (in km^2), inverting
        the bilinear relations. The rake is ignored.
        """
        area = numpy.asarray(area)
        thresh = 1.95 * math.pow(99.0, 5.0 / 3.0)
        length = numpy.where(
            area <= thresh,
            power(area / 1.95, 3.0 / 5.0),
            area / 20.0,
        )
        return numpy.where(
            length <= 99.0,
            2.0 * (log10(length) + 1.9),
            log10(length) + 4.7,
        )

    def get_std_dev_mag(self, area, rake):
        """
        Returns the standard deviation on the magnitude.
        """
        return self.SIGMA_L

    def get_median_length(self, mag):
        """
        Calculates median rupture length (in km) from magnitude with the
        bilinear relation log10(L) = 0.5 m - 1.9 for m <= 7.1, else
        log10(L) = m - 4.7.
        """
        mag = numpy.asarray(mag)
        log_l = numpy.where(mag <= 7.1, 0.5 * mag - 1.9, mag - 4.7)
        return power(10.0, log_l)

    def get_std_dev_length(self, mag):
        """
        Returns std of log10 rupture length. Magnitude is ignored.
        """
        return self.SIGMA_L

    def get_median_width(self, mag):
        """
        Calculates median rupture width (in km) from magnitude as
        W = 1.95 * L^(2/3), capped at 20 km.
        """
        return self._width_from_length(self.get_median_length(mag))

    def get_average_displacement(self, mag):
        """
        Calculates median average displacement (in m) from magnitude as
        AD = 1.7e-5 * L, with L in metres.
        """
        return 1.7e-5 * self.get_median_length(mag) * 1000.0

    def get_std_dev_displacement(self, mag):
        """
        Returns std of log10 average displacement (from length).
        """
        return self.SIGMA_L


