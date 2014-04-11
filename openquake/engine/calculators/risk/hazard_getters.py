# -*- coding: utf-8 -*-

# Copyright (c) 2012-2014, GEM Foundation.
#
# OpenQuake is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# OpenQuake is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with OpenQuake.  If not, see <http://www.gnu.org/licenses/>.

"""
Hazard getters for Risk calculators.

A HazardGetter is responsible fo getting hazard outputs needed by a risk
calculation.
"""
import numpy

from openquake.hazardlib.imt import from_string

from openquake.engine import logs
from openquake.engine.db import models


class HazardGetter(object):
    """
    Base abstract class of an Hazard Getter.

    An Hazard Getter is used to query for the closest hazard data for
    each given asset. An Hazard Getter must be pickable such that it
    should be possible to use different strategies (e.g. distributed
    or not, using postgis or not).

    :attr hazard_outputs:
        A list of hazard output container instances (e.g. HazardCurve)
    :attr assets:
        The assets for which we wants to compute.
    :attr imt:
        The imt (in long form) for which data have to be retrieved
    """

    @property
    def assets(self):
        return [a for assets in self.site_assets.itervalues() for a in assets]

    def __init__(self, hazard_outputs, site_assets, imt):
        self.hazard_outputs = hazard_outputs
        self.site_assets = site_assets
        self.imt = imt
        self.imt_type, self.sa_period, self.sa_damping = from_string(imt)
        assert site_assets

    def __repr__(self):
        return "<%s assets=%s>" % (
            self.__class__.__name__, [a.id for a in self.assets])

    def get_data(self, hazard_output, monitor):
        """
        Subclasses must implement this.
        """
        raise NotImplementedError

    def weights(self):
        ws = []
        for hazard in self.hazard_outputs:
            h = hazard.output_container
            if hasattr(h, 'lt_realization') and h.lt_realization:
                ws.append(h.lt_realization.weight)
        return ws


class HazardCurveGetterPerAsset(HazardGetter):
    """
    Simple HazardCurve Getter that performs a spatial query for each
    asset.

    :attr imls:
        The intensity measure levels of the curves we are going to get.
    """
    def get_data(self, output_container, monitor, extra=None):
        """
        Calls ``get_by_site`` for each asset and pack the results as
        requested by the :meth:`HazardGetter.get_data` interface.
        """
        oc = output_container

        if oc.output.output_type == 'hazard_curve':
            imls = oc.imls
        elif oc.output.output_type == 'hazard_curve_multi':
            oc = models.HazardCurve.objects.get(
                output__oq_job=oc.output.oq_job,
                output__output_type='hazard_curve',
                statistics=oc.statistics,
                lt_realization=oc.lt_realization,
                imt=self.imt_type,
                sa_period=self.sa_period,
                sa_damping=self.sa_damping)
            imls = oc.imls

        all_assets, all_curves = [], []
        with monitor.copy('getting closest hazard curves'):
            for site_id, assets in self.site_assets.iteritems():
                site = models.HazardSite.objects.get(pk=site_id)
                poes = self._get_poes(site, oc.id)
                curve = zip(imls, poes)
                for asset in assets:
                    all_assets.append(asset)
                    all_curves.append(curve)
        logs.LOG.info(
            'Getting data from gmf_id=%d, %d sites, %d assets, IMT=%s',
            oc.id, len(self.site_assets), len(all_assets), self.imt)
        return all_assets, all_curves

    def _get_poes(self, site, hazard_id):
        cursor = models.getcursor('job_init')
        query = """\
        SELECT hzrdr.hazard_curve_data.poes
        FROM hzrdr.hazard_curve_data
        WHERE hazard_curve_id = %s AND location = %s
        """
        cursor.execute(query, (hazard_id, 'SRID=4326; ' + site.location.wkt))
        return cursor.fetchall()[0][0]


class ScenarioGetter(HazardGetter):
    """
    Hazard getter for loading ground motion values. It is instantiated
    with a set of assets all of the same taxonomy.
    """
    def get_gmvs(self, gmf, site_id):
        """
        :returns: gmvs and ruptures for the given site and IMT
        """
        gmvs = []
        for gmf in models.GmfData.objects.filter(
                gmf=gmf,
                site=site_id, imt=self.imt_type, sa_period=self.sa_period,
                sa_damping=self.sa_damping):
            gmvs.extend(gmf.gmvs)
            if not gmvs:
                logs.LOG.warn('No gmvs for site %s, IMT=%s', site_id, self.imt)
        return gmvs

    def get_data(self, output_container, monitor, extra=None):
        """
        :returns: a list with all the assets and the hazard data.

        For scenario computations the data is a numpy.array
        with the GMVs; for event based computations the data is
        a pair (GMVs, rupture_ids).
        """
        all_assets = []
        all_gmvs = []
        # dictionary site -> ({rupture_id: gmv}, n_assets)
        # the ordering is there only to have repeatable runs
        with monitor.copy('getting gmvs'):
            for site_id, assets in self.site_assets.iteritems():
                n_assets = len(assets)
                all_assets.extend(assets)
                gmvs = self.get_gmvs(output_container, site_id)
                if gmvs:
                    array = numpy.array(gmvs)
                    all_gmvs.extend([array] * n_assets)

        logs.LOG.info(
            'Getting data from gmf_id=%d, %d sites, %d assets, IMT=%s',
            output_container.id, len(self.site_assets), len(all_assets),
            self.imt)

        return all_assets, all_gmvs


class GroundMotionValuesGetter(ScenarioGetter):
    """
    Hazard getter for loading ground motion values. It is instantiated
    with a set of assets all of the same taxonomy.
    """

    def get_data(self, output_container, monitor, (rupture, sitecol)):
        """
        :returns: a list with all the assets and the hazard data.

        For scenario computations the data is a numpy.array
        with the GMVs; for event based computations the data is
        a pair (GMVs, rupture_ids).
        """
        with monitor.copy('associating assets'):
            indices = models.ProbabilisticRupture.objects.filter(
                rupture=rupture).site_indices
            assets = sum([self.site_assets[site_id]
                          for site_id in sitecol.sids[indices]], [])

        with monitor.copy('getting gmvs'):
            gmvs = models.GmfRupture.objects.filter(
                rupture=rupture,
                gmf=output_container,
                imt__imt_str=self.imt).ground_motion_field

        return assets, gmvs


class BCRGetter(object):
    def __init__(self, getter_orig, getter_retro):
        self.assets = getter_orig.assets
        self.getter_orig = getter_orig
        self.getter_retro = getter_retro

    def __call__(self, monitor, extra=None):
        orig_gen = self.getter_orig(monitor)
        retro_gen = self.getter_retro(monitor)
        try:
            while True:
                hid, assets, orig = orig_gen.next()
                _hid, _assets, retro = retro_gen.next()
                yield hid, assets, (orig, retro)
        except StopIteration:
            pass
