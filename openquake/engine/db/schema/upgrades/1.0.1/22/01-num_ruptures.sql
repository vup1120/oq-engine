ALTER TABLE hzrdr.ses_collection ADD COLUMN num_ruptures INTEGER[];
ALTER TABLE hzrdr.probabilistic_rupture ADD COLUMN site_indices INTEGER[];

GRANT SELECT,INSERT,UPDATE ON hzrdr.ses_collection TO oq_job_init;
