# -*- coding: utf-8 -*-
"""
Tests for the primary surface rupture model registry
(:func:`openquake.fdha.primary_surf_rup.get_available_primary_surf_rup`).
"""
import unittest

from openquake.fdha.primary_surf_rup import get_available_primary_surf_rup
from openquake.fdha.primary_surf_rup.base import BasePrimarySurfRup


# Concrete, usable model classes expected to be discovered by the registry.
EXPECTED_CLASSES = {
    'WC1993PrimarySR',
    'Takao2013PrimarySR',
    'MossRoss2011PrimarySR',
    'Yang2021PrimarySR',
    'Moss2013PrimarySR_Reverse',
    'Moss2013PrimarySR_SS',
    'Pizza2023PrimarySR',
    'Pizza2023PrimarySR_Normal',
    'Pizza2023PrimarySR_Reverse',
    'Pizza2023PrimarySR_SS',
    'FixedPrimarySR',
    'Youngs2003PrimarySR_ExC',
    'Youngs2003PrimarySR_GB',
    'Youngs2003PrimarySR_nBR',
}


class RegistryTestCase(unittest.TestCase):
    """Tests for get_available_primary_surf_rup."""

    def setUp(self):
        self.registry = get_available_primary_surf_rup()

    def test_contains_all_expected(self):
        missing = EXPECTED_CLASSES - set(self.registry)
        self.assertEqual(missing, set(),
                         "registry missing: %s" % sorted(missing))

    def test_keys_match_class_names(self):
        for name, cls in self.registry.items():
            self.assertEqual(name, cls.__name__)

    def test_all_are_subclasses(self):
        for cls in self.registry.values():
            self.assertTrue(issubclass(cls, BasePrimarySurfRup))

    def test_sorted_keys(self):
        keys = list(self.registry)
        self.assertEqual(keys, sorted(keys))

    def test_base_class_excluded(self):
        self.assertNotIn('BasePrimarySurfRup', self.registry)
