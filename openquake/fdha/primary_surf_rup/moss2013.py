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
Module :mod:`openquake.fdha.primary_surf_rup.moss2013` implements the
model of Moss et al. (2013) in :class:`Moss2013PrimarySR_Reverse` and
:class:`Moss2013PrimarySR_SS`.
"""

import numpy as np
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


class Moss2013Base(BasePrimarySurfRup):
    """
    Base class for Moss et al. (2013) logistic surface rupture probability
    models. The probability P = 1 / (1 + exp(a - b*mag)) depends on the site
    stiffness: subclasses set the stiff-soil coefficients (:attr:`coeff_a_stiff`,
    :attr:`coeff_b_stiff`, used where Vs30 > 600 m/s) and the soft-soil
    coefficients (:attr:`coeff_a_soft`, :attr:`coeff_b_soft`, used where
    Vs30 <= 600 m/s).

    References
    ----------
    Moss, R. E. S., Stanton, K. V., & Buelna, M. I. (2013). The impact of
    material stiffness on the likelihood of fault rupture propagating to the
    ground surface. Seismological Research Letters, 84(3), 485-488.
    https://doi.org/10.1785/0220110109
    """

    coeff_a_stiff = None
    coeff_b_stiff = None
    coeff_a_soft = None
    coeff_b_soft = None

    def get_prob(self, ctx):
        """
        Compute the probability of surface rupture from magnitude and site
        Vs30 using the logistic model P = 1 / (1 + exp(a - b*mag)), selecting
        the stiff-soil branch where Vs30 > 600 m/s and the soft-soil branch
        otherwise.

        :param ctx:
            Context object with attributes ``mag`` (magnitude) and ``vs30``
            (time-averaged shear-wave velocity to 30 m, m/s). Both may be
            scalar or array; ``vs30`` broadcasts against ``mag``.
        :returns:
            Probability as float (scalar) or :class:`numpy.ndarray`,
            broadcast from ``ctx.mag`` and ``ctx.vs30``.
        """
        m = np.asarray(ctx.mag, dtype=float)
        vs30 = np.asarray(ctx.vs30, dtype=float)
        prob_stiff = 1.0 / (1.0 + np.exp(
            self.coeff_a_stiff - self.coeff_b_stiff * m))
        prob_soft = 1.0 / (1.0 + np.exp(
            self.coeff_a_soft - self.coeff_b_soft * m))
        prob = np.where(vs30 > 600.0, prob_stiff, prob_soft)
        return prob.item() if prob.shape == () else prob


class Moss2013PrimarySR_Reverse(Moss2013Base):
    """
    Moss et al. (2013) primary surface rupture model for reverse faulting.
    Stiff soil (Vs30 > 600): P = 1 / (1 + exp(13.9745 - 2.1395*mag));
    soft soil (Vs30 <= 600): P = 1 / (1 + exp(6.2548 - 0.8308*mag)).
    """

    coeff_a_stiff = 13.9745
    coeff_b_stiff = 2.1395
    coeff_a_soft = 6.2548
    coeff_b_soft = 0.8308


class Moss2013PrimarySR_SS(Moss2013Base):
    """
    Moss et al. (2013) primary surface rupture model for strike-slip
    faulting. Stiff soil (Vs30 > 600): P = 1 / (1 + exp(11.4071 - 1.8465*mag));
    soft soil (Vs30 <= 600): P = 1 / (1 + exp(12.2908 - 1.9520*mag)).
    """

    coeff_a_stiff = 11.4071
    coeff_b_stiff = 1.8465
    coeff_a_soft = 12.2908
    coeff_b_soft = 1.9520
