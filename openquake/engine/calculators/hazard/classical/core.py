# Copyright (c) 2010-2013, GEM Foundation.
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
Core functionality for the classical PSHA hazard calculator.
"""
import numpy
from openquake.hazardlib.imt import from_string

from openquake.engine import logs, writer
from openquake.engine.calculators.hazard import general
from openquake.engine.calculators.hazard.classical import (
    post_processing as post_proc)
from openquake.engine.input import logictree
from openquake.engine.db import models
from openquake.engine.utils import tasks as tasks
from openquake.engine.performance import EnginePerformanceMonitor


@tasks.oqtask
def generate_curves(job_id, ruptures, rlzs, ltp):
    """
    Celery task for hazard curve calculator.

    Samples logic trees, gathers site parameters, and calls the hazard curve
    calculator.

    Once hazard curve data is computed, result progress updated (within a
    transaction, to prevent race conditions) in the
    `htemp.hazard_curve_progress` table.

    :param int job_id:
        ID of the currently running job.
    :param ruptures:
        List of
        :class:`openquake.hazardlib.source.rupture.ProbabilisticRupture`
        objects
    :param rlzs:
        a list of :class:`openquake.engine.db.models.LtRealization` instances
    :param ltp:
        a :class:`openquake.engine.input.LogicTreeProcessor` instance
    """
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    curves_by_rlz = []
    for lt_rlz in rlzs:
        gsims = ltp.parse_gmpe_logictree_path(lt_rlz.gsim_lt_path)
        imts = general.im_dict_to_hazardlib(
            hc.intensity_measure_types_and_levels)
        sites = hc.site_collection
        curves = dict((imt, numpy.ones([len(sites), len(imts[imt])]))
                      for imt in imts)
        for rupture in ruptures:
            r_sites = sites
            prob = rupture.get_probability_one_or_more_occurrences()
            gsim = gsims[rupture.tectonic_region_type]
            sctx, rctx, dctx = gsim.make_contexts(r_sites, rupture)
            for imt in imts:
                poes = gsim.get_poes(sctx, rctx, dctx, imt, imts[imt],
                                     hc.truncation_level)
                curves[imt] *= r_sites.expand(
                    (1 - prob) ** poes, len(sites), placeholder=1
                )
        for imt in imts:
            curves[imt] = 1 - curves[imt]
        curves_by_imt = []
        for imt in sorted(imts):
            if (curves[imt] == 0.0).all():
                # shortcut for filtered sources giving no contribution;
                # this is essential for performance, we want to avoid
                # returning big arrays of zeros (MS)
                curves_by_imt.append(None)
            else:
                curves_by_imt.append(curves[imt])
        curves_by_rlz.append(curves_by_imt)
    return curves_by_rlz


class ClassicalHazardCalculator(general.BaseHazardCalculator):
    """
    Classical PSHA hazard calculator. Computes hazard curves for a given set of
    points.

    For each realization of the calculation, we randomly sample source models
    and GMPEs (Ground Motion Prediction Equations) from logic trees.
    """
    def execute(self):
        ltp = logictree.LogicTreeProcessor.from_hc(self.hc)
        for ltpath, sources in self.sources_per_ltpath.iteritems():
            task_args = [(self.job.id, srcs)
                         for srcs in self.block_split(sources)]
            self.initialize_percent(general.generate_ruptures, task_args)
            ruptures = tasks.map_reduce(
                general.generate_ruptures, task_args, self.add_lists, [])
            rlzs = self.rlzs_per_ltpath[ltpath]
            zeros = [[numpy.zeros((self.n_sites, len(self.imtls[imt])))
                      for imt in sorted(self.imtls)] for rlz in rlzs]
            task_args = [(self.job.id, rupts, rlzs, ltp)
                         for rupts in self.block_split(ruptures)]
            self.initialize_percent(generate_curves, task_args)
            curves = tasks.map_reduce(
                generate_curves, task_args, self.multiply_curves, zeros)
            self.save_hazard_curves(curves, rlzs)

    @EnginePerformanceMonitor.monitor
    def add_lists(self, acc, lst):
        newacc = acc + lst
        self.log_percent()
        return newacc

    @EnginePerformanceMonitor.monitor
    def multiply_curves(self, acc, curves_by_rlz):
        """
        This is used to incrementally update hazard curve results by combining
        an initial value with some new results. (Each set of new results is
        computed over only a subset of seismic sources defined in the
        calculation model.)

        :param task_result:
            A pair (curves_by_imt, ordinal) where curves_by_imt is a
            list of 2-D numpy arrays representing the new results which need
            to be combined with the current value. These should be the same
            shape as self.curves_by_rlz[i][j] where i is the realization
            ordinal and j the IMT ordinal.
        """
        for i, curves_by_imt in enumerate(curves_by_rlz):  # i is the rlz index
            for j, matrix in enumerate(curves_by_imt):  # j is the IMT index
                if matrix is not None:
                    acc[i][j] = 1. - (1. - acc[i][j]) * (1. - matrix)
        self.log_percent()
        return acc

    # this could be parallelized in the future, however in all the cases
    # I have seen until now, the serialized approach is fast enough (MS)
    @EnginePerformanceMonitor.monitor
    def save_hazard_curves(self, curves_by_rlz, rlzs):
        """
        Save the hazard curve results on the database.
        """
        for curves_by_imt, rlz in zip(curves_by_rlz, rlzs):
            # create a new `HazardCurve` 'container' record for each
            # realization (virtual container for multiple imts)
            models.HazardCurve.objects.create(
                output=models.Output.objects.create_output(
                    self.job, "hc-multi-imt-rlz-%s" % rlz.id,
                    "hazard_curve_multi"),
                lt_realization=rlz,
                imt=None,
                investigation_time=self.hc.investigation_time)

            # create a new `HazardCurve` 'container' record for each
            # realization for each intensity measure type
            for imt, curves_by_imt in zip(sorted(self.imtls), curves_by_imt):
                hc_im_type, sa_period, sa_damping = from_string(imt)

                # save output
                hco = models.Output.objects.create(
                    oq_job=self.job,
                    display_name="Hazard Curve rlz-%s" % rlz.id,
                    output_type='hazard_curve',
                )

                # save hazard_curve
                haz_curve = models.HazardCurve.objects.create(
                    output=hco,
                    lt_realization=rlz,
                    investigation_time=self.hc.investigation_time,
                    imt=hc_im_type,
                    imls=self.imtls[imt],
                    sa_period=sa_period,
                    sa_damping=sa_damping,
                )

                # save hazard_curve_data
                points = self.hc.points_to_compute()
                logs.LOG.info('saving %d hazard curves for %s, imt=%s',
                              len(points), hco, imt)
                writer.CacheInserter.saveall([
                    models.HazardCurveData(
                        hazard_curve=haz_curve,
                        poes=list(poes),
                        location='POINT(%s %s)' % (p.longitude, p.latitude),
                        weight=rlz.weight)
                    for p, poes in zip(points, curves_by_imt)])

    def post_process(self):
        """
        Optionally generates aggregate curves, hazard maps and
        uniform_hazard_spectra.
        """
        logs.LOG.debug('> starting post processing')

        # means/quantiles:
        if self.hc.mean_hazard_curves or self.hc.quantile_hazard_curves:
            self.do_aggregate_post_proc()

        # hazard maps:
        # required for computing UHS
        # if `hazard_maps` is false but `uniform_hazard_spectra` is true,
        # just don't export the maps
        if self.hc.hazard_maps or self.hc.uniform_hazard_spectra:
            self.parallelize(
                post_proc.hazard_curves_to_hazard_map_task,
                post_proc.hazard_curves_to_hazard_map_task_arg_gen(self.job),
                self.log_percent)

        if self.hc.uniform_hazard_spectra:
            post_proc.do_uhs_post_proc(self.job)

        logs.LOG.debug('< done with post processing')
