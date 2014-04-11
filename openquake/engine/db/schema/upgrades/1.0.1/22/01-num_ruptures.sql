-- imt table ----------------------------------------------------------
/*
NB: the imt_check
CHECK(imt_str = CASE
      WHEN im_type = 'SA' THEN 'SA(' || sa_period::TEXT || ')'
      ELSE im_type END)
was removed on purpose. The reason is that SA(1.0) becomes
SA(1) for postgres and the constraint cannot be satisfied.
I may consider removing all the constraints later on.
*/
CREATE TABLE hzrdi.imt(
  id SERIAL PRIMARY KEY,
  imt_str VARCHAR UNIQUE NOT NULL -- full string representation of the IMT
    CHECK(imt_str = CASE
          WHEN im_type = 'SA' THEN 'SA(' || sa_period::TEXT || ')'
          ELSE im_type END),
  im_type VARCHAR NOT NULL, -- short string for the IMT
  sa_period FLOAT CONSTRAINT imt_sa_period
        CHECK(((im_type = 'SA') AND (sa_period IS NOT NULL))
              OR ((im_type != 'SA') AND (sa_period IS NULL))),
  sa_damping FLOAT CONSTRAINT imt_sa_damping
        CHECK(((im_type = 'SA') AND (sa_damping IS NOT NULL))
            OR ((im_type != 'SA') AND (sa_damping IS NULL))),
  UNIQUE (im_type, sa_period, sa_damping)
) TABLESPACE hzrdi_ts;

ALTER TABLE hzrdi.imt OWNER TO oq_admin;
GRANT SELECT, INSERT ON hzrdi.imt TO oq_job_init;
GRANT USAGE ON hzrdi.imt_id_seq TO oq_job_init;

-- predefined intensity measure types
INSERT INTO hzrdi.imt (imt_str, im_type, sa_period, sa_damping) VALUES
('PGA', 'PGA', NULL, NULL),
('PGV', 'PGV', NULL, NULL),
('PGD', 'PGD', NULL, NULL),
('IA', 'IA', NULL, NULL),
('RSD', 'RSD', NULL, NULL),
('MMI', 'MMI', NULL, NULL),
('SA(0.1)', 'SA', 0.1, 5.0);

-- gmf_rupture table ---------------------------------------------------

CREATE TABLE hzrdr.gmf_rupture (
   id SERIAL PRIMARY KEY,
   rupture_id INTEGER NOT NULL,  -- fk to hzardr.ses_rupture
   gmf_id INTEGER NOT NULL, -- fk to hzrdr.gmf
   imt_id INTEGER NOT NULL, -- fk to hzrdi.imt
   ground_motion_field FLOAT[] NOT NULL
) TABLESPACE hzrdr_ts;

ALTER TABLE hzrdr.gmf_rupture OWNER TO oq_admin;
GRANT SELECT, INSERT ON hzrdr.gmf_rupture TO oq_job_init;
GRANT USAGE ON hzrdr.gmf_rupture_id_seq TO oq_job_init;

-- hzrdr.gmf_rupture -> hzrdi.imt FK
ALTER TABLE hzrdr.gmf_rupture
ADD CONSTRAINT hzrdr_gmf_rupture_ses_rupture_fk
FOREIGN KEY (rupture_id)
REFERENCES hzrdi.ses_rupture(id)
ON DELETE CASCADE;

-- hzrdr.gmf_rupture -> hzrdi.imt FK
ALTER TABLE hzrdr.gmf_rupture
ADD CONSTRAINT hzrdr_gmf_rupture_imt_fk
FOREIGN KEY (imt_id)
REFERENCES hzrdi.imt(id)
ON DELETE CASCADE;

-- hzrdr.gmf_rupture -> hzrdr.gmf FK
ALTER TABLE hzrdr.gmf_rupture
ADD CONSTRAINT hzrdr_gmf_rupture_gmf_fk
FOREIGN KEY (gmf_id)
REFERENCES hzrdr.gmf(id)
ON DELETE CASCADE;


-- modify probabilistic_rupture ------------------------------------------
ALTER TABLE hzrdr.probabilistic_rupture ADD COLUMN site_indices INTEGER[];

-- drop gmf_data
DROP TABLE hzrdr.gmf_data;
