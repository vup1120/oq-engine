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
Module :mod:`openquake.fdha.primary_surf_rup.fixed` implements a fixed
probability model for primary surface rupture in :class:`FixedPrimarySR`.
"""

import numpy as np
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


class FixedPrimarySR(BasePrimarySurfRup):
    """
    Fixed probability model for primary surface rupture.

    Returns a constant P(SR) value regardless of magnitude or style. This is
    useful when the user wants to assume surface rupture is certain
    (P(SR) = 1.0) or set to any other fixed probability.

    :param value:
        The fixed probability value to return. Must be between 0 and 1.
        Default is 1.0.
    """

    def __init__(self, value=1.0):
        """
        :param value:
            The fixed probability value to return. Must be between 0 and 1.
            Default is 1.0.
        :raises ValueError:
            If ``value`` is not in the closed interval [0, 1].
        """
        self.value = float(value)
        if not 0.0 <= self.value <= 1.0:
            raise ValueError(
                "Value must be between 0 and 1, got %s" % self.value)

    def get_prob(self, ctx):
        """
        Return the fixed probability value, shaped to match ``ctx.mag``.

        :param ctx:
            Context object with attribute ``mag`` (magnitude, scalar or
            array).
        :returns:
            The fixed probability as float (scalar) or
            :class:`numpy.ndarray`, same shape as ``ctx.mag``.
        """
        m = np.asarray(ctx.mag, dtype=float)
        prob = np.full_like(m, self.value)
        return prob.item() if prob.shape == () else prob

    def __repr__(self):
        return "FixedPrimarySR(value=%s)" % self.value
