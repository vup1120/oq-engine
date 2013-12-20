ALTER TABLE hzrdr.gmf_data RENAME ses_id TO task_no;
ALTER TABLE hzrdr.gmf_data DROP CONSTRAINT hzrdr_gmf_data_ses_fk;

ALTER TABLE hzrdr.ses_rupture ADD COLUMN site_indices INTEGER[];
