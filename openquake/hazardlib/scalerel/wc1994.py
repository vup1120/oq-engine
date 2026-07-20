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
Module :mod:`openquake.hazardlib.scalerel.wc1994` implements :class:`WC1994`.
"""
from math import log10
from openquake.hazardlib.scalerel.base import BaseMSRSigma, BaseASRSigma


class WC1994(BaseMSRSigma, BaseASRSigma):
    """
    Wells and Coppersmith magnitude -- rupture parameters relationships,
    see 1994, Bull. Seism. Soc. Am., pages 974-2002.

    Implements scaling relationships for:
    - Moment Magnitude (M)
    - Rupture Area (RA)
    - Surface Rupture Length (SRL)
    - Subsurface Rupture Length (RLD)
    - Rupture Width (RW)
    """
    def get_median_area(self, mag, rake):
        """
        Calculates median area from magnitude.

        The values are a function of both magnitude and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param mag:
            Moment magnitude.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 10.0 ** (-3.49 + 0.91 * mag)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 10.0 ** (-3.42 + 0.90 * mag)
        elif rake > 0:
            # thrust/reverse
            return 10.0 ** (-3.99 + 0.98 * mag)
        else:
            # normal
            return 10.0 ** (-2.87 + 0.82 * mag)

    def get_std_dev_area(self, mag, rake):
        """
        Returns std of the logarithm of rupture area. Magnitude is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.24
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.22
        elif rake > 0:
            # thrust/reverse
            return 0.26
        else:
            # normal
            return 0.22

    def get_median_mag(self, area, rake):
        """
        Calculates median magnitude from area.

        The values are a function of both area and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param area:
            Area in square km.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 4.07 + 0.98 * log10(area)
        elif (-45 <= rake <= 45) or (rake > 135) or (rake < -135):
            # strike slip
            return 3.98 + 1.02 * log10(area)
        elif rake > 0:
            # thrust/reverse
            return 4.33 + 0.90 * log10(area)
        else:
            # normal
            return 3.93 + 1.02 * log10(area)

    def get_std_dev_mag(self, area, rake):
        """
        Returns std for magnitude. Area is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.24
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.23
        elif rake > 0:
            # thrust/reverse
            return 0.25
        else:
            # normal
            return 0.25

    def get_median_srl(self, mag, rake):
        """
        Calculates median surface rupture length from magnitude.

        The values are a function of both magnitude and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param mag:
            Moment magnitude.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 10.0 ** (-3.22 + 0.69 * mag)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 10.0 ** (-3.55 + 0.74 * mag)
        elif rake > 0:
            # thrust/reverse
            return 10.0 ** (-2.86 + 0.63 * mag)
        else:
            # normal
            return 10.0 ** (-2.01 + 0.50 * mag)

    def get_std_dev_srl(self, mag, rake):
        """
        Returns std of the logarithm of surface rupture length. Magnitude is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.22
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.23
        elif rake > 0:
            # thrust/reverse
            return 0.20
        else:
            # normal
            return 0.21

    def get_median_mag_from_srl(self, srl, rake):
        """
        Calculates median magnitude from surface rupture length.

        The values are a function of both surface rupture length and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param srl:
            Surface rupture length in km.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 5.08 + 1.16 * log10(srl)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 5.16 + 1.12 * log10(srl)
        elif rake > 0:
            # thrust/reverse
            return 5.00 + 1.22 * log10(srl)
        else:
            # normal
            return 4.86 + 1.32 * log10(srl)

    def get_std_dev_mag_from_srl(self, srl, rake):
        """
        Returns std for magnitude. Surface rupture length is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.28
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.28
        elif rake > 0:
            # thrust/reverse
            return 0.28
        else:
            # normal
            return 0.34

    def get_median_rld(self, mag, rake):
        """
        Calculates median subsurface rupture length from magnitude.

        The values are a function of both magnitude and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param mag:
            Moment magnitude.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 10.0 ** (-2.44 + 0.59 * mag)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 10.0 ** (-2.57 + 0.62 * mag)
        elif rake > 0:
            # thrust/reverse
            return 10.0 ** (-2.42 + 0.58 * mag)
        else:
            # normal
            return 10.0 ** (-1.88 + 0.50 * mag)

    def get_std_dev_rld(self, mag, rake):
        """
        Returns std of the logarithm of subsurface rupture length. Magnitude is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.16
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.15
        elif rake > 0:
            # thrust/reverse
            return 0.16
        else:
            # normal
            return 0.17

    def get_median_mag_from_rld(self, rld, rake):
        """
        Calculates median magnitude from subsurface rupture length.

        The values are a function of both subsurface rupture length and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param rld:
            Subsurface rupture length in km.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 4.38 + 1.49 * log10(rld)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 4.33 + 1.49 * log10(rld)
        elif rake > 0:
            # thrust/reverse
            return 4.49 + 1.49 * log10(rld)
        else:
            # normal
            return 4.34 + 1.54 * log10(rld)

    def get_std_dev_mag_from_rld(self, rld, rake):
        """
        Returns std for magnitude. Subsurface rupture length is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.26
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.24
        elif rake > 0:
            # thrust/reverse
            return 0.26
        else:
            # normal
            return 0.31

    def get_median_rw(self, mag, rake):
        """
        Calculates median rupture width from magnitude.

        The values are a function of both magnitude and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param mag:
            Moment magnitude.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 10.0 ** (-1.01 + 0.32 * mag)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 10.0 ** (-0.76 + 0.27 * mag)
        elif rake > 0:
            # thrust/reverse
            return 10.0 ** (-1.61 + 0.41 * mag)
        else:
            # normal
            return 10.0 ** (-1.14 + 0.35 * mag)

    def get_std_dev_rw(self, mag, rake):
        """
        Returns std of the logarithm of rupture width. Magnitude is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.15
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.14
        elif rake > 0:
            # thrust/reverse
            return 0.15
        else:
            # normal
            return 0.12

    def get_median_mag_from_rw(self, rw, rake):
        """
        Calculates median magnitude from rupture width.

        The values are a function of both rupture width and rake.

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied.

        :param rw:
            Rupture width in km.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is None:
            # their "All" case
            return 4.06 + 2.25 * log10(rw)
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 3.80 + 2.59 * log10(rw)
        elif rake > 0:
            # thrust/reverse
            return 4.37 + 1.95 * log10(rw)
        else:
            # normal
            return 4.04 + 2.11 * log10(rw)

    def get_std_dev_mag_from_rw(self, rw, rake):
        """
        Returns std for magnitude. Rupture width is ignored.
        """
        if rake is None:
            # their "All" case
            return 0.41
        elif (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135):
            # strike slip
            return 0.45
        elif rake > 0:
            # thrust/reverse
            return 0.32
        else:
            # normal
            return 0.31

    def get_average_displacement(self, mag, rake, return_sigma=False):
        """
        Calculates median average displacement (AD, in metres) from
        magnitude, from their Table 2B regression log10(AD) = a + b * M.

        The values are a function of both magnitude and rake. Setting the
        rake to ``None`` causes their "All" rupture-types to be applied.
        NB: for thrust/reverse events the "All" coefficients are used as
        well, since Wells and Coppersmith (1994) report their reverse
        displacement regressions as not significant at the 95% level and
        recommend the all-slip-type relation instead.

        :param mag:
            Moment magnitude.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        :param return_sigma:
            If True, returns a ``(median, sigma)`` pair where sigma is the
            standard deviation of log10(AD).
        """
        if rake is not None and (
                (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135)):
            # strike slip
            ad, sigma = 10.0 ** (-6.32 + 0.90 * mag), 0.28
        elif rake is not None and rake < 0:
            # normal
            ad, sigma = 10.0 ** (-4.45 + 0.63 * mag), 0.33
        else:
            # their "All" case, also covering thrust/reverse
            ad, sigma = 10.0 ** (-4.80 + 0.69 * mag), 0.36
        return (ad, sigma) if return_sigma else ad

    def get_maximum_displacement(self, mag, rake, return_sigma=False):
        """
        Calculates median maximum displacement (MD, in metres) from
        magnitude, from their Table 2B regression log10(MD) = a + b * M.

        The values are a function of both magnitude and rake. Setting the
        rake to ``None`` causes their "All" rupture-types to be applied.
        NB: for thrust/reverse events the "All" coefficients are used as
        well, since Wells and Coppersmith (1994) report their reverse
        displacement regressions as not significant at the 95% level and
        recommend the all-slip-type relation instead.

        :param mag:
            Moment magnitude.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        :param return_sigma:
            If True, returns a ``(median, sigma)`` pair where sigma is the
            standard deviation of log10(MD).
        """
        if rake is not None and (
                (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135)):
            # strike slip
            md, sigma = 10.0 ** (-7.03 + 1.03 * mag), 0.34
        elif rake is not None and rake < 0:
            # normal
            md, sigma = 10.0 ** (-5.90 + 0.89 * mag), 0.38
        else:
            # their "All" case, also covering thrust/reverse
            md, sigma = 10.0 ** (-5.46 + 0.82 * mag), 0.42
        return (md, sigma) if return_sigma else md

    def get_median_mag_from_ad(self, ad, rake):
        """
        Calculates median magnitude from average displacement, from their
        Table 2B regression M = a + b * log10(AD).

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied; thrust/reverse also uses "All" (see
        :meth:`get_average_displacement`).

        :param ad:
            Average displacement in metres.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is not None and (
                (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135)):
            # strike slip
            return 7.04 + 0.89 * log10(ad)
        elif rake is not None and rake < 0:
            # normal
            return 6.78 + 0.65 * log10(ad)
        else:
            # their "All" case, also covering thrust/reverse
            return 6.93 + 0.82 * log10(ad)

    def get_std_dev_mag_from_ad(self, ad, rake):
        """
        Returns std for magnitude. Average displacement is ignored.
        """
        if rake is not None and (
                (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135)):
            # strike slip
            return 0.28
        elif rake is not None and rake < 0:
            # normal
            return 0.33
        else:
            # their "All" case, also covering thrust/reverse
            return 0.39

    def get_median_mag_from_md(self, md, rake):
        """
        Calculates median magnitude from maximum displacement, from their
        Table 2B regression M = a + b * log10(MD).

        Setting the rake to ``None`` causes their "All" rupture-types
        to be applied; thrust/reverse also uses "All" (see
        :meth:`get_maximum_displacement`).

        :param md:
            Maximum displacement in metres.
        :param rake:
            Rake angle (the rupture propagation direction) in degrees,
            from -180 to 180.
        """
        if rake is not None and (
                (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135)):
            # strike slip
            return 6.81 + 0.78 * log10(md)
        elif rake is not None and rake < 0:
            # normal
            return 6.61 + 0.71 * log10(md)
        else:
            # their "All" case, also covering thrust/reverse
            return 6.69 + 0.74 * log10(md)

    def get_std_dev_mag_from_md(self, md, rake):
        """
        Returns std for magnitude. Maximum displacement is ignored.
        """
        if rake is not None and (
                (-45 <= rake <= 45) or (rake >= 135) or (rake <= -135)):
            # strike slip
            return 0.29
        elif rake is not None and rake < 0:
            # normal
            return 0.34
        else:
            # their "All" case, also covering thrust/reverse
            return 0.40

