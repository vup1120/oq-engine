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
Module :mod:`openquake.fdha.primary_surf_rup.moss_ross2011` implements the
model of Moss and Ross (2011) in :class:`MossRoss2011PrimarySR`.
"""

import numpy as np
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


class MossRoss2011PrimarySR(BasePrimarySurfRup):
    """
    Principal surface-rupture probability model of Moss and Ross (2011).

    Logistic model of the probability of principal surface rupture for
    reverse-faulting events as a function of magnitude, with a single
    regression (P = 1 / (1 + exp(7.3 - 1.03*mag))).

    References
    ----------
    Moss, R.E.S., and Ross, Z.E. (2011). Probabilistic fault displacement
    hazard analysis for reverse faults. Bulletin of the Seismological
    Society of America, 101(4), 1542-1553.
    https://doi.org/10.1785/0120100248
    """

    def get_prob(self, ctx):
        """
        Compute the probability of surface rupture for ruptures with a
        reverse mechanism using the logistic model
        P = 1 / (1 + exp(7.3 - 1.03 * mag)).

        :param ctx:
            Context object with attribute ``mag`` (magnitude, scalar or
            array).
        :returns:
            Probability as float (scalar) or :class:`numpy.ndarray`, same
            shape as ``ctx.mag``.
        """
        m = np.asarray(ctx.mag, dtype=float)
        prob = 1.0 / (1.0 + np.exp(7.3 - 1.03 * m))
        return prob.item() if prob.shape == () else prob
