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
Module :mod:`openquake.fdha.primary_surf_rup.yang2021` implements the
model of Yang et al. (2021) in :class:`Yang2021PrimarySR`.

Supported Fault Styles: Reverse only

References
----------
Yang, H., Quigley, M., & King, T. (2021). Surface slip distributions and
geometric complexity of intraplate reverse-faulting earthquakes. GSA
Bulletin, 133(9-10), 1909-1929. https://doi.org/10.1130/B35809.1
"""

import numpy as np
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


class Yang2021PrimarySR(BasePrimarySurfRup):
    """
    Model of Yang et al. (2021) for the probability of surface rupture for
    reverse-faulting earthquakes in the Australian stable continental region
    (the "SCR Oz" logistic curve of their Fig. 11A, with a = 24.59 and
    b = -4.00 in P = 1/(1 + exp(a + b*M)), i.e. fx = -24.59 + 4.0*mag).

    The regression is stated by the authors to be valid only for
    4.0 <= Mw <= 6.6.
    """

    def get_prob(self, ctx):
        """
        Compute the probability of surface rupture for ruptures with a
        reverse mechanism using the logistic model
        P = exp(fx) / (1 + exp(fx)), fx = -24.59 + 4.0 * mag.

        The regression is valid for 4.0 <= Mw <= 6.6.

        :param ctx:
            Context object with attribute ``mag`` (magnitude, scalar or
            array).
        :returns:
            Probability as float (scalar) or :class:`numpy.ndarray`, same
            shape as ``ctx.mag``.
        """
        m = np.asarray(ctx.mag, dtype=float)
        fx = -24.59 + 4.0 * m
        prob = np.exp(fx) / (1.0 + np.exp(fx))
        return prob.item() if prob.shape == () else prob
