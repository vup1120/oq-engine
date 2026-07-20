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
Package :mod:`openquake.fdha.primary_surf_rup` contains implementations of
primary (principal) surface-rupture probability models P(surface rupture | m).

The available models are discovered dynamically by scanning the package's
modules for non-abstract subclasses of
:class:`openquake.fdha.primary_surf_rup.base.BasePrimarySurfRup`, so new
models are registered simply by adding a module here (no hand-maintained
import list). Use :func:`get_available_primary_surf_rup` to obtain them.
"""
import os
import inspect
import importlib

from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


def _get_available_class(base_class):
    """
    Return an ordered dictionary with the available classes in the
    :mod:`openquake.fdha.primary_surf_rup` submodules that derive from
    ``base_class``, keyed by class name.
    """
    classes = {}  # class_name -> class
    for fname in os.listdir(os.path.dirname(__file__)):
        if fname.endswith('.py'):
            modname, _ext = os.path.splitext(fname)
            mod = importlib.import_module(
                'openquake.fdha.primary_surf_rup.' + modname)
            for cls in mod.__dict__.values():
                if inspect.isclass(cls) and issubclass(cls, base_class) \
                        and cls != base_class \
                        and not inspect.isabstract(cls):
                    classes[cls.__name__] = cls
    return dict((k, classes[k]) for k in sorted(classes))


def get_available_primary_surf_rup():
    """
    Return an ordered dictionary with the available primary surface-rupture
    probability model classes, keyed by class name.
    """
    return _get_available_class(BasePrimarySurfRup)
