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
Module :mod:`openquake.fdha.primary_surf_rup.pizza2023` implements the
model of Pizza et al. (2023) in :class:`Pizza2023PrimarySR` and its
faulting-style variants.
"""

import numpy as np
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


class Pizza2023PrimarySR(BasePrimarySurfRup):
    """
    Principal surface-rupture probability model of Pizza et al. (2023).

    Logistic model of the probability of principal surface rupture as a
    function of magnitude, P = exp(fx) / (1 + exp(fx)), fx = a + b * mag.
    Subclasses set :attr:`coeff_a` and :attr:`coeff_b` (logistic regression
    coefficients) for the ``normal``, ``reverse`` and ``strike-slip``
    faulting styles; this class itself carries the ``all`` (pooled)
    coefficients (a=-14.47, b=2.177).

    References
    ----------
    Pizza, M., Ferrario, M.F., Thomas, F., Tringali, G., & Livio, F. (2023).
    Likelihood of primary surface faulting: updating of empirical regressions.
    Bulletin of the Seismological Society of America, 113(5), 2106-2118.
    https://doi.org/10.1785/0120230019
    """

    coeff_a = -14.47
    coeff_b = 2.177

    def get_prob(self, ctx):
        """
        Compute the probability of surface rupture from magnitude using the
        logistic model P = exp(fx) / (1 + exp(fx)), fx = a + b * mag.

        :param ctx:
            Context object with attribute ``mag`` (magnitude, scalar or
            array).
        :returns:
            Probability as float (scalar) or :class:`numpy.ndarray`, same
            shape as ``ctx.mag``.
        """
        m = np.asarray(ctx.mag, dtype=float)
        fx = self.coeff_a + self.coeff_b * m
        prob = np.exp(fx) / (1.0 + np.exp(fx))
        return prob.item() if prob.shape == () else prob


class Pizza2023PrimarySR_Normal(Pizza2023PrimarySR):
    """
    Pizza et al. (2023) primary surface rupture model for normal faulting.
    Coefficients (a, b) = (-13.5, 2.159).
    """

    coeff_a = -13.5
    coeff_b = 2.159


class Pizza2023PrimarySR_Reverse(Pizza2023PrimarySR):
    """
    Pizza et al. (2023) primary surface rupture model for reverse faulting.
    Coefficients (a, b) = (-10.75, 1.427).
    """

    coeff_a = -10.75
    coeff_b = 1.427


class Pizza2023PrimarySR_SS(Pizza2023PrimarySR):
    """
    Pizza et al. (2023) primary surface rupture model for strike-slip
    faulting. Coefficients (a, b) = (-28.56, 4.436).
    """

    coeff_a = -28.56
    coeff_b = 4.436
