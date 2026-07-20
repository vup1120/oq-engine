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
Module :mod:`openquake.fdha.primary_surf_rup.wells_coppersmith1993` implements
the model of Wells and Coppersmith (1993) in :class:`WC1993PrimarySR`.
"""

import numpy as np
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


class WC1993PrimarySR(BasePrimarySurfRup):
    """
    Principal surface-rupture probability model of Wells and Coppersmith
    (1993).

    Logistic model of the probability of principal surface rupture as a
    function of magnitude, applicable to all faulting styles. The single
    logistic regression (a=-12.51, b=2.053) covers all faulting styles, so no
    style variants are declared.

    References
    ----------
    Wells, D.L., and Coppersmith, K.J. (1993). Likelihood of surface rupture
    as a function of magnitude (abstract). Seismological Research Letters,
    64(1), 54. Coefficients as reported by Youngs et al. (2003), Earthquake
    Spectra, 19(1), 191-219, and Petersen et al. (2011), Bulletin of the
    Seismological Society of America, 101(2), 805-825.
    """

    def get_prob(self, ctx):
        """
        Compute the probability of surface rupture for ruptures with any
        faulting mechanism using the logistic model
        P = exp(fx) / (1 + exp(fx)), fx = -12.51 + 2.053 * mag.

        :param ctx:
            Context object with attribute ``mag`` (magnitude, scalar or
            array).
        :returns:
            Probability as float (scalar) or :class:`numpy.ndarray`, same
            shape as ``ctx.mag``.
        """
        m = np.asarray(ctx.mag, dtype=float)
        fx = -12.51 + 2.053 * m
        prob = np.exp(fx) / (1.0 + np.exp(fx))
        return prob.item() if prob.shape == () else prob
