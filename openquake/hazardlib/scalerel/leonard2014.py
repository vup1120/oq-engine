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
Module :mod:`openquake.hazardlib.scalerel.leonard2014` implements
:class:`Leonard2014_SCR`
:class:`Leonard2014_Interplate`
"""
from numpy import power, log10
from openquake.hazardlib.scalerel.base import BaseMSRSigma, BaseASRSigma


class Leonard2014_SCR(BaseMSRSigma, BaseASRSigma):
    """
    Leonard, M., 2014. Self-consistent earthquake fault-scaling relations:
    Update and extension to stable continental strike-slip faults.
    Bulletin of the Seismological Society of America, 104(6), pp 2953-2965.

    Implements both magnitude-area and area-magnitude scaling relationships.
    """
    def get_median_area(self, mag, rake):
        """
        Calculates median fault area from magnitude.
        """
        if rake is None:
            # Return average of strike-slip and dip-slip curves
            return power(10.0, (mag - 4.185))
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike-slip
            return power(10.0, (mag - 4.18))
        else:
            # Dip-slip (thrust or normal), and undefined rake
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
        if rake is None:
            # Return average of strike-slip and dip-slip curves
            return log10(area) + 4.185
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return log10(area) + 4.18
        else:
            # Dip slip (thrust or normal), and undefined rake
            return log10(area) + 4.19

    def get_std_dev_mag(self, area, rake):
        """
        Returns zero for now
        """
        return 0.0

    def get_median_width(self, mag, rake):
        """
        Calculates median rupture width (in km) from magnitude, inverting
        the Leonard (2014) SCR relation m = a + 2.5 * log10(W), with
        a = 4.22 for strike-slip and a = 4.14 for dip-slip faulting
        (the constants used by Mammarella et al. 2024, Table 1).

        Setting the rake to ``None`` returns the average of the
        strike-slip and dip-slip curves.
        """
        if rake is None:
            # average of strike-slip and dip-slip curves
            return power(10.0, (mag - 4.18) / 2.5)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike-slip
            return power(10.0, (mag - 4.22) / 2.5)
        else:
            # dip-slip (thrust or normal)
            return power(10.0, (mag - 4.14) / 2.5)

    def get_std_dev_width(self, mag, rake):
        """
        Returns the standard deviation of log10(W). Magnitude and rake
        are ignored.
        """
        return 0.15


class Leonard2014_Interplate(BaseMSRSigma, BaseASRSigma):
    """
    Leonard, M., 2014. Self-consistent earthquake fault-scaling relations:
    Update and extension to stable continental strike-slip faults.
    Bulletin of the Seismological Society of America, 104(6), pp 2953-2965.

    Implements both magnitude-area and area-magnitude scaling relationships.
    """
    def get_median_area(self, mag, rake):
        """
        Calculates median fault area from magnitude.
        """
        if rake is None:
            # Return average of strike-slip and dip-slip curves
            return power(10.0, (mag - 3.995))
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return power(10.0, (mag - 3.99))
        else:
            # Dip slip (thrust or normal), and undefined rake
            return power(10.0, (mag - 4.00))

    def get_std_dev_area(self, mag, rake):
        """
        Returns zero for now
        """
        return 0.0

    def get_median_mag(self, area, rake):
        """
        Calculates median magnitude from fault area.
        """
        if rake is None:
            # Return average of strike-slip and dip-slip curves
            return log10(area) + 3.995
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return log10(area) + 3.99
        else:
            # Dip slip (thrust or normal), and undefined rake
            return log10(area) + 4.00

    def get_std_dev_mag(self, area, rake):
        """
        Returns None for now
        """
        return 0.0

    def get_median_width(self, mag, rake):
        """
        Calculates median rupture width (in km) from magnitude, inverting
        the Leonard (2014) interplate relation m = a + 2.5 * log10(W),
        with a = 3.88 for strike-slip and a = 3.63 for dip-slip faulting
        (the constants used by Mammarella et al. 2024, Table 1).

        Setting the rake to ``None`` returns the average of the
        strike-slip and dip-slip curves.
        """
        if rake is None:
            # average of strike-slip and dip-slip curves
            return power(10.0, (mag - 3.755) / 2.5)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike-slip
            return power(10.0, (mag - 3.88) / 2.5)
        else:
            # dip-slip (thrust or normal)
            return power(10.0, (mag - 3.63) / 2.5)

    def get_std_dev_width(self, mag, rake):
        """
        Returns the standard deviation of log10(W). Magnitude and rake
        are ignored.
        """
        return 0.15
