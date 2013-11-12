# -*- coding: utf-8 -*-
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
Scenario calculator core functionality. It works by splitting the
computation for blocks of realizations. The number of realizations
typically is in the range 10^3 - 10^5: therefore a block size of 1000
generates 1-100 tasks which is a reasonable amount.
Notice that is MUCH more efficient to use a large block size
when calling `openquake.hazardlib.calc.ground_motion_fields`:
this goes against parallelism, though.
"""

import math
import random
import itertools

from django.db import transaction
import numpy

from openquake.nrmllib.hazard.parsers import RuptureModelParser

# HAZARDLIB
from openquake.hazardlib.calc import ground_motion_fields
import openquake.hazardlib.gsim

from openquake.engine.calculators.hazard import general as haz_general
from openquake.engine.utils import tasks
from openquake.engine.db import models
from openquake.engine.input import source
from openquake.engine import writer
from openquake.engine.performance import EnginePerformanceMonitor

BLOCK_SIZE = 1000  # this is hard-coded, not a configuration parameter
# if it was a user-accessible parameter, users will get different
# numbers by tweaking it, since the same job would generated a
# different number of seeds

AVAILABLE_GSIMS = openquake.hazardlib.gsim.get_available_gsims()


@tasks.oqtask
def compute_gmfs(job_id, task_seeds, rupture, gmf_id):
    """
    Compute ground motion fields and store them in the db.

    :param job_id:
        ID of the currently running job.
    :param task_seeds:
        The seeds to generate each realization
    :param rupture:
        The hazardlib rupture from which we will generate
        ground motion fields.
    :param gmf_id:
        the id of a :class:`openquake.engine.db.models.Gmf` record
    """
    assert len(task_seeds) == 1, (
        'There must be only one seed, got %s' % task_seeds)
    numpy.random.seed(task_seeds[0])
    hc = models.HazardCalculation.objects.get(oqjob=job_id)
    imts = [haz_general.imt_to_hazardlib(x)
            for x in hc.intensity_measure_types]
    gsim = AVAILABLE_GSIMS[hc.gsim]()  # instantiate the GSIM class
    correlation_model = haz_general.get_correl_model(hc)

    with EnginePerformanceMonitor('computing gmfs', job_id, compute_gmfs):
        gmf_dict = ground_motion_fields(
            rupture, hc.site_collection, imts, gsim,
            hc.truncation_level, realizations=BLOCK_SIZE,
            correlation_model=correlation_model)
    with EnginePerformanceMonitor('saving gmfs', job_id, compute_gmfs):
        save_gmf(gmf_id, gmf_dict, hc.site_collection)


@transaction.commit_on_success(using='reslt_writer')
def save_gmf(gmf_id, gmf_dict, sites):
    """
    Helper method to save computed GMF data to the database.

    :param int gmf_id:
        the id of a :class:`openquake.engine.db.models.Gmf` record
    :param dict gmf_dict:
        The GMF results during the calculation
    :param sites:
        An :class:`openquake.engine.models.SiteCollection`
        object
    """
    inserter = writer.CacheInserter(models.GmfData, 100)
    # NB: GmfData may contain large arrays and the cache may become large

    for imt, gmvs in gmf_dict.iteritems():
        sa_period = None
        sa_damping = None
        if isinstance(imt, openquake.hazardlib.imt.SA):
            sa_period = imt.period
            sa_damping = imt.damping
        imt_name = imt.__class__.__name__
        for values, site in itertools.izip(gmvs, sites):
            # convert the numpy array to list of floats
            if values.shape == (1, BLOCK_SIZE):
                values = values.reshape(BLOCK_SIZE, 1)
            data = map(float, values)
            inserter.add(models.GmfData(
                gmf_id=gmf_id,
                ses_id=None,
                imt=imt_name,
                sa_period=sa_period,
                sa_damping=sa_damping,
                site_id=site.id,
                rupture_ids=None,
                gmvs=data))

    inserter.flush()


class ScenarioHazardCalculator(haz_general.BaseHazardCalculator):
    """
    Scenario hazard calculator. Computes ground motion fields.
    """

    core_calc_task = compute_gmfs
    output = None  # defined in pre_execute

    def __init__(self, *args, **kwargs):
        super(ScenarioHazardCalculator, self).__init__(*args, **kwargs)
        self.gmf = None
        self.rupture = None

    def initialize_sources(self):
        """
        Get the rupture_model file from the job.ini file, and set the
        attribute self.rupture.
        """
        nrml = RuptureModelParser(self.hc.inputs['rupture_model']).parse()
        rms = self.job.hazard_calculation.rupture_mesh_spacing
        self.rupture = source.nrml_to_hazardlib(nrml, rms, None, None)

    def pre_execute(self):
        """
        Do pre-execution work. At the moment, this work entails:
        parsing and initializing sources, parsing and initializing the
        site model (if there is one), parsing vulnerability and
        exposure files, and generating logic tree realizations. (The
        latter piece basically defines the work to be done in the
        `execute` phase.)
        """

        # Parse risk models.
        self.parse_risk_models()

        # Deal with the site model and compute site data for the calculation
        # If no site model file was specified, reference parameters are used
        # for all sites.
        self.initialize_site_model()

        # Create rupture input
        self.initialize_sources()

        # create a record in the output table
        output = models.Output.objects.create(
            oq_job=self.job,
            display_name="gmf_scenario",
            output_type="gmf_scenario")

        # create an associated gmf record
        self.gmf = models.Gmf.objects.create(output=output)

    def execute(self):
        self.parallelize(self.core_calc_task, self.task_arg_gen(BLOCK_SIZE))

    def task_arg_gen(self, block_size):
        """
        Loop through realizations and sources to generate a sequence of
        task arg tuples. Each tuple of args applies to a single task.

        Yielded results are 4-uples of the form job_id, seeds, rupture, gmf_id.
        The seeds will be used to seed numpy for temporal occurence sampling.
        """
        rnd = random.Random()
        rnd.seed(self.hc.random_seed)
        n_gmf = self.hc.number_of_ground_motion_fields
        n_tasks = int(math.ceil(float(n_gmf) / block_size))
        seeds = [rnd.randint(0, models.MAX_SINT_32) for _ in range(n_tasks)]
        for seed in seeds:
            # the oqtask decorator wants the second argument of a task to be
            # a list, this is why we are sending a single argument list
            yield self.job.id, [seed], self.rupture, self.gmf.id
