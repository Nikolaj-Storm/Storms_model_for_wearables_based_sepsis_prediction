# MIMIC-IV Features Appendix

## Dataset: `mimiciv_3_1_hosp`

| Features (Table.Column) | Description |
| :--- | :--- |
| **admissions.subject_id** | Each row of this table contains a unique `hadm_id`, which represents a single patient's admission to the hospital. `hadm_id` ranges from 2000000 - 2999999. It is possible for this table to have duplicate `subject_id`, indicating that a single patient had multiple admissions to the hospital. The A... (INTEGER NOT NULL) |
| **admissions.hadm_id** | Each row of this table contains a unique `hadm_id`, which represents a single patient's admission to the hospital. `hadm_id` ranges from 2000000 - 2999999. It is possible for this table to have duplicate `subject_id`, indicating that a single patient had multiple admissions to the hospital. The A... (INTEGER NOT NULL) |
| **admissions.admittime** | `admittime` provides the date and time the patient was admitted to the hospital, while `dischtime` provides the date and time the patient was discharged from the hospital. If applicable, `deathtime` provides the time of in-hospital death for the patient. Note that `deathtime` is only present if t... (TIMESTAMP NOT NULL) |
| **admissions.dischtime** | `admittime` provides the date and time the patient was admitted to the hospital, while `dischtime` provides the date and time the patient was discharged from the hospital. If applicable, `deathtime` provides the time of in-hospital death for the patient. Note that `deathtime` is only present if t... (TIMESTAMP) |
| **admissions.deathtime** | `admittime` provides the date and time the patient was admitted to the hospital, while `dischtime` provides the date and time the patient was discharged from the hospital. If applicable, `deathtime` provides the time of in-hospital death for the patient. Note that `deathtime` is only present if t... (TIMESTAMP) |
| **admissions.admission_type** | `admission_type` is useful for classifying the urgency of the admission. There are 9 possibilities: 'AMBULATORY OBSERVATION', 'DIRECT EMER.', 'DIRECT OBSERVATION', 'ELECTIVE', 'EU OBSERVATION', 'EW EMER.', 'OBSERVATION ADMIT', 'SURGICAL SAME DAY ADMISSION', 'URGENT'. (VARCHAR(40) NOT NULL) |
| **admissions.admit_provider_id** | `admit_provider_id` provides an anonymous identifier for the provider who admitted the patient. {{% include "/static/include/provider_id.md" %}} (VARCHAR(10)) |
| **admissions.admission_location** | `admission_location` provides information about the location of the patient prior to arriving at the hospital. Note that as the emergency room is technically a clinic, patients who are admitted via the emergency room usually have it as their admission location. (VARCHAR(60)) |
| **admissions.discharge_location** | `admission_location` provides information about the location of the patient prior to arriving at the hospital. Note that as the emergency room is technically a clinic, patients who are admitted via the emergency room usually have it as their admission location. (VARCHAR(60)) |
| **admissions.insurance** | The `insurance`, `language`, `marital_status`, and `ethnicity` columns provide information about patient demographics for the given hospitalization. Note that as this data is documented for each hospital admission, they may change from stay to stay. (VARCHAR(255)) |
| **admissions.language** | The `insurance`, `language`, `marital_status`, and `ethnicity` columns provide information about patient demographics for the given hospitalization. Note that as this data is documented for each hospital admission, they may change from stay to stay. (VARCHAR(10)) |
| **admissions.marital_status** | The `insurance`, `language`, `marital_status`, and `ethnicity` columns provide information about patient demographics for the given hospitalization. Note that as this data is documented for each hospital admission, they may change from stay to stay. (VARCHAR(30)) |
| **admissions.race** | Basic column identifier or value. (VARCHAR(80)) |
| **admissions.edregtime** | The date and time at which the patient was registered and discharged from the emergency department. (TIMESTAMP) |
| **admissions.edouttime** | The date and time at which the patient was registered and discharged from the emergency department. (TIMESTAMP) |
| **admissions.hospital_expire_flag** | This is a binary flag which indicates whether the patient died within the given hospitalization. `1` indicates death in the hospital, and `0` indicates survival to hospital discharge. (SMALLINT) |
| **d_hcpcs.code** | A five character code which uniquely represents the event. (CHAR(5) NOT NULL) |
| **d_hcpcs.category** | Broad classification of the code. (SMALLINT) |
| **d_hcpcs.long_description** | Textual descriptions of the `code` listed for the given row. (TEXT) |
| **d_hcpcs.short_description** | Textual descriptions of the `code` listed for the given row. (VARCHAR(180)) |
| **d_icd_diagnoses.icd_code** | `icd_code` is the International Coding Definitions (ICD) code. (CHAR(7) NOT NULL) |
| **d_icd_diagnoses.icd_version** | `icd_code` is the International Coding Definitions (ICD) code. (INTEGER NOT NULL) |
| **d_icd_diagnoses.long_title** | The `long_title` provides the meaning of the ICD code. For example, the ICD-9 code 0010 has `long_title` "Cholera due to vibrio cholerae". (VARCHAR(255)) |
| **d_icd_procedures.icd_code** | `icd_code` is the International Coding Definitions (ICD) code. (CHAR(7) NOT NULL) |
| **d_icd_procedures.icd_version** | `icd_code` is the International Coding Definitions (ICD) code. (INTEGER NOT NULL) |
| **d_icd_procedures.long_title** | Basic column identifier or value. (VARCHAR(255)) |
| **d_labitems.itemid** | A unique identifier for a laboratory concept. `itemid` is unique to each row, and can be used to identify data in labevents associated with a specific concept. (INTEGER) |
| **d_labitems.label** | The `label` column describes the concept which is represented by the `itemid`. (VARCHAR(50)) |
| **d_labitems.fluid** | `fluid` describes the substance on which the measurement was made. For example, chemistry measurements are frequently performed on blood, which is listed in this column as 'BLOOD'. Many of these measurements are also acquirable on other fluids, such as urine, and this column differentiates these ... (VARCHAR(50)) |
| **d_labitems.category** | `category` provides higher level information as to the type of measurement. For example, a category of 'ABG' indicates that the measurement is an arterial blood gas. (VARCHAR(50)) |
| **diagnoses_icd.subject_id** | {{% include "/static/include/subject_id.md" %}} (INTEGER NOT NULL) |
| **diagnoses_icd.hadm_id** | {{% include "/static/include/hadm_id.md" %}} (INTEGER NOT NULL) |
| **diagnoses_icd.seq_num** | The priority assigned to the diagnoses. The priority can be interpreted as a ranking of which diagnoses are "important", but many caveats to this broad statement exist. For example, patients who are diagnosed with sepsis must have sepsis as their *2nd* billed condition. The 1st billed condition m... (INTEGER NOT NULL) |
| **diagnoses_icd.icd_code** | `icd_code` is the International Coding Definitions (ICD) code. (VARCHAR(7)) |
| **diagnoses_icd.icd_version** | `icd_code` is the International Coding Definitions (ICD) code. (INTEGER) |
| **drgcodes.subject_id** | {{% include "/static/include/subject_id.md" %}} (INTEGER) |
| **drgcodes.hadm_id** | {{% include "/static/include/hadm_id.md" %}} (INTEGER) |
| **drgcodes.drg_type** | The specific DRG ontology used for the code. (VARCHAR(4)) |
| **drgcodes.drg_code** | The DRG code. (VARCHAR(10)) |
| **drgcodes.description** | A description for the given DRG code. (VARCHAR(195)) |
| **drgcodes.drg_severity** | Some DRG ontologies further qualify the patient severity of illness and likelihood of mortality, which are recorded here. (SMALLINT) |
| **drgcodes.drg_mortality** | Some DRG ontologies further qualify the patient severity of illness and likelihood of mortality, which are recorded here. (SMALLINT) |
| **emar.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **emar.hadm_id** | Basic column identifier or value. (INTEGER) |
| **emar.emar_id** | Basic column identifier or value. (VARCHAR(25) NOT NULL) |
| **emar.emar_seq** | Basic column identifier or value. (INTEGER NOT NULL) |
| **emar.poe_id** | Basic column identifier or value. (VARCHAR(25) NOT NULL) |
| **emar.pharmacy_id** | Basic column identifier or value. (INTEGER) |
| **emar.enter_provider_id** | Basic column identifier or value. (VARCHAR(10)) |
| **emar.charttime** | Basic column identifier or value. (TIMESTAMP NOT NULL) |
| **emar.medication** | Basic column identifier or value. (TEXT) |
| **emar.event_txt** | Basic column identifier or value. (VARCHAR(100)) |
| **emar.scheduletime** | Basic column identifier or value. (TIMESTAMP) |
| **emar.storetime** | Basic column identifier or value. (TIMESTAMP NOT NULL) |
| **emar_detail.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **emar_detail.emar_id** | Basic column identifier or value. (VARCHAR(25) NOT NULL) |
| **emar_detail.emar_seq** | Basic column identifier or value. (INTEGER NOT NULL) |
| **emar_detail.parent_field_ordinal** | Basic column identifier or value. (VARCHAR(10)) |
| **emar_detail.administration_type** | Basic column identifier or value. (VARCHAR(50)) |
| **emar_detail.pharmacy_id** | Basic column identifier or value. (INTEGER) |
| **emar_detail.barcode_type** | Basic column identifier or value. (VARCHAR(4)) |
| **emar_detail.reason_for_no_barcode** | Basic column identifier or value. (TEXT) |
| **emar_detail.complete_dose_not_given** | Basic column identifier or value. (VARCHAR(5)) |
| **emar_detail.dose_due** | Basic column identifier or value. (VARCHAR(100)) |
| **emar_detail.dose_due_unit** | Basic column identifier or value. (VARCHAR(50)) |
| **emar_detail.dose_given** | Basic column identifier or value. (VARCHAR(255)) |
| **emar_detail.dose_given_unit** | Basic column identifier or value. (VARCHAR(50)) |
| **emar_detail.will_remainder_of_dose_be_given** | Basic column identifier or value. (VARCHAR(5)) |
| **emar_detail.product_amount_given** | Basic column identifier or value. (VARCHAR(30)) |
| **emar_detail.product_unit** | Basic column identifier or value. (VARCHAR(30)) |
| **emar_detail.product_code** | Basic column identifier or value. (VARCHAR(30)) |
| **emar_detail.product_description** | Basic column identifier or value. (VARCHAR(255)) |
| **emar_detail.product_description_other** | Basic column identifier or value. (VARCHAR(255)) |
| **emar_detail.prior_infusion_rate** | Basic column identifier or value. (VARCHAR(40)) |
| **emar_detail.infusion_rate** | Basic column identifier or value. (VARCHAR(40)) |
| **emar_detail.infusion_rate_adjustment** | Basic column identifier or value. (VARCHAR(50)) |
| **emar_detail.infusion_rate_adjustment_amount** | Basic column identifier or value. (VARCHAR(30)) |
| **emar_detail.infusion_rate_unit** | Basic column identifier or value. (VARCHAR(30)) |
| **emar_detail.route** | Basic column identifier or value. (VARCHAR(10)) |
| **emar_detail.infusion_complete** | Basic column identifier or value. (VARCHAR(1)) |
| **emar_detail.completion_interval** | Basic column identifier or value. (VARCHAR(50)) |
| **emar_detail.new_iv_bag_hung** | Basic column identifier or value. (VARCHAR(1)) |
| **emar_detail.continued_infusion_in_other_location** | Basic column identifier or value. (VARCHAR(1)) |
| **emar_detail.restart_interval** | Basic column identifier or value. (TEXT) |
| **emar_detail.side** | Basic column identifier or value. (VARCHAR(10)) |
| **emar_detail.site** | Basic column identifier or value. (VARCHAR(255)) |
| **emar_detail.non_formulary_visual_verification** | Basic column identifier or value. (VARCHAR(1)) |
| **hcpcsevents.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **hcpcsevents.hadm_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **hcpcsevents.chartdate** | Basic column identifier or value. (DATE) |
| **hcpcsevents.hcpcs_cd** | Basic column identifier or value. (CHAR(5) NOT NULL) |
| **hcpcsevents.seq_num** | Basic column identifier or value. (INTEGER NOT NULL) |
| **hcpcsevents.short_description** | Basic column identifier or value. (VARCHAR(180)) |
| **labevents.labevent_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **labevents.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **labevents.hadm_id** | Basic column identifier or value. (INTEGER) |
| **labevents.specimen_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **labevents.itemid** | Basic column identifier or value. (INTEGER NOT NULL) |
| **labevents.order_provider_id** | Basic column identifier or value. (VARCHAR(10)) |
| **labevents.charttime** | Basic column identifier or value. (TIMESTAMP(0)) |
| **labevents.storetime** | Basic column identifier or value. (TIMESTAMP(0)) |
| **labevents.value** | Basic column identifier or value. (VARCHAR(200)) |
| **labevents.valuenum** | Basic column identifier or value. (DOUBLE PRECISION) |
| **labevents.valueuom** | Basic column identifier or value. (VARCHAR(20)) |
| **labevents.ref_range_lower** | Basic column identifier or value. (DOUBLE PRECISION) |
| **labevents.ref_range_upper** | Basic column identifier or value. (DOUBLE PRECISION) |
| **labevents.flag** | Basic column identifier or value. (VARCHAR(10)) |
| **labevents.priority** | Basic column identifier or value. (VARCHAR(7)) |
| **labevents.comments** | Basic column identifier or value. (TEXT) |
| **omr.subject_id** | {{% include "/static/include/subject_id.md" %}} (INTEGER NOT NULL) |
| **omr.chartdate** | Basic column identifier or value. (DATE NOT NULL) |
| **omr.seq_num** | Basic column identifier or value. (INTEGER NOT NULL) |
| **omr.result_name** | Basic column identifier or value. (VARCHAR(100) NOT NULL) |
| **omr.result_value** | `result_value` is the value associated with the given OMR observation. For example, for the `result_name` of 'Blood Pressure', the `field_value` column contains the recorded blood pressure (120/80, 130/70, and so on). (TEXT NOT NULL) |
| **patients.subject_id** | `subject_id` is a unique identifier which specifies an individual patient. Any rows associated with a single `subject_id` pertain to the same individual. As `subject_id` is the primary key for the table, it is unique for each row. (INTEGER NOT NULL) |
| **patients.gender** | `gender` is the genotypical sex of the patient. (VARCHAR(1) NOT NULL) |
| **patients.anchor_age** | These columns provide information regarding the actual patient year for the patient admission, and the patient's age at that time. (INTEGER NOT NULL) |
| **patients.anchor_year** | These columns provide information regarding the actual patient year for the patient admission, and the patient's age at that time. (INTEGER NOT NULL) |
| **patients.anchor_year_group** | These columns provide information regarding the actual patient year for the patient admission, and the patient's age at that time. (VARCHAR(255) NOT NULL) |
| **patients.dod** | The de-identified date of death for the patient. Date of death is extracted from two sources: the hospital information system and the [Massachusetts State Registry of Vital Records and Statistics](https://www.mass.gov/orgs/registry-of-vital-records-and-statistics). Individual patient records from... (TIMESTAMP(0)) |
| **pharmacy.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **pharmacy.hadm_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **pharmacy.pharmacy_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **pharmacy.poe_id** | Basic column identifier or value. (VARCHAR(25)) |
| **pharmacy.starttime** | Basic column identifier or value. (TIMESTAMP(3)) |
| **pharmacy.stoptime** | Basic column identifier or value. (TIMESTAMP(3)) |
| **pharmacy.medication** | Basic column identifier or value. (TEXT) |
| **pharmacy.proc_type** | Basic column identifier or value. (VARCHAR(50) NOT NULL) |
| **pharmacy.status** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.entertime** | Basic column identifier or value. (TIMESTAMP(3) NOT NULL) |
| **pharmacy.verifiedtime** | Basic column identifier or value. (TIMESTAMP(3)) |
| **pharmacy.route** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.frequency** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.disp_sched** | Basic column identifier or value. (VARCHAR(255)) |
| **pharmacy.infusion_type** | Basic column identifier or value. (VARCHAR(15)) |
| **pharmacy.sliding_scale** | Basic column identifier or value. (VARCHAR(1)) |
| **pharmacy.lockout_interval** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.basal_rate** | Basic column identifier or value. (REAL) |
| **pharmacy.one_hr_max** | Basic column identifier or value. (VARCHAR(10)) |
| **pharmacy.doses_per_24_hrs** | Basic column identifier or value. (REAL) |
| **pharmacy.duration** | Basic column identifier or value. (REAL) |
| **pharmacy.duration_interval** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.expiration_value** | Basic column identifier or value. (INTEGER) |
| **pharmacy.expiration_unit** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.expirationdate** | Basic column identifier or value. (TIMESTAMP(3)) |
| **pharmacy.dispensation** | Basic column identifier or value. (VARCHAR(50)) |
| **pharmacy.fill_quantity** | Basic column identifier or value. (VARCHAR(50)) |
| **poe.poe_id** | Basic column identifier or value. (VARCHAR(25) NOT NULL) |
| **poe.poe_seq** | Basic column identifier or value. (INTEGER NOT NULL) |
| **poe.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **poe.hadm_id** | Basic column identifier or value. (INTEGER) |
| **poe.ordertime** | Basic column identifier or value. (TIMESTAMP(0) NOT NULL) |
| **poe.order_type** | Basic column identifier or value. (VARCHAR(25) NOT NULL) |
| **poe.order_subtype** | Basic column identifier or value. (VARCHAR(50)) |
| **poe.transaction_type** | Basic column identifier or value. (VARCHAR(15)) |
| **poe.discontinue_of_poe_id** | Basic column identifier or value. (VARCHAR(25)) |
| **poe.discontinued_by_poe_id** | Basic column identifier or value. (VARCHAR(25)) |
| **poe.order_provider_id** | Basic column identifier or value. (VARCHAR(10)) |
| **poe.order_status** | Basic column identifier or value. (VARCHAR(15)) |
| **poe_detail.poe_id** | Basic column identifier or value. (VARCHAR(25) NOT NULL) |
| **poe_detail.poe_seq** | Basic column identifier or value. (INTEGER NOT NULL) |
| **poe_detail.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **poe_detail.field_name** | Basic column identifier or value. (VARCHAR(255) NOT NULL) |
| **poe_detail.field_value** | Basic column identifier or value. (TEXT) |
| **prescriptions.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **prescriptions.hadm_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **prescriptions.pharmacy_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **prescriptions.poe_id** | Basic column identifier or value. (VARCHAR(25)) |
| **prescriptions.poe_seq** | Basic column identifier or value. (INTEGER) |
| **prescriptions.order_provider_id** | Basic column identifier or value. (VARCHAR(10)) |
| **prescriptions.starttime** | Basic column identifier or value. (TIMESTAMP(3)) |
| **prescriptions.stoptime** | Basic column identifier or value. (TIMESTAMP(3)) |
| **prescriptions.drug_type** | Basic column identifier or value. (VARCHAR(20) NOT NULL) |
| **prescriptions.drug** | Basic column identifier or value. (VARCHAR(255) NOT NULL) |
| **prescriptions.formulary_drug_cd** | Basic column identifier or value. (VARCHAR(50)) |
| **prescriptions.gsn** | Basic column identifier or value. (VARCHAR(255)) |
| **prescriptions.ndc** | Basic column identifier or value. (VARCHAR(25)) |
| **prescriptions.prod_strength** | Basic column identifier or value. (VARCHAR(255)) |
| **prescriptions.form_rx** | Basic column identifier or value. (VARCHAR(25)) |
| **prescriptions.dose_val_rx** | Basic column identifier or value. (VARCHAR(100)) |
| **prescriptions.dose_unit_rx** | Basic column identifier or value. (VARCHAR(50)) |
| **prescriptions.form_val_disp** | Basic column identifier or value. (VARCHAR(50)) |
| **prescriptions.form_unit_disp** | Basic column identifier or value. (VARCHAR(50)) |
| **prescriptions.doses_per_24_hrs** | Basic column identifier or value. (REAL) |
| **prescriptions.route** | Basic column identifier or value. (VARCHAR(50)) |
| **procedures_icd.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **procedures_icd.hadm_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **procedures_icd.seq_num** | Basic column identifier or value. (INTEGER NOT NULL) |
| **procedures_icd.chartdate** | Basic column identifier or value. (DATE NOT NULL) |
| **procedures_icd.icd_code** | Basic column identifier or value. (VARCHAR(7)) |
| **procedures_icd.icd_version** | Basic column identifier or value. (INTEGER) |
| **provider.provider_id** | Basic column identifier or value. (VARCHAR(10) NOT NULL) |
| **services.subject_id** | Basic column identifier or value. (INT) |
| **services.hadm_id** | Basic column identifier or value. (INT) |
| **services.transfertime** | Basic column identifier or value. (TIMESTAMP(0)) |
| **services.prev_service** | Basic column identifier or value. (VARCHAR(20)) |
| **services.curr_service** | Basic column identifier or value. (VARCHAR(20)) |
| **transfers.subject_id** | Basic column identifier or value. (INTEGER NOT NULL) |
| **transfers.hadm_id** | Identifiers which specify the patient: `subject_id` is unique to a patient, `hadm_id` is unique to a patient hospital stay, and `transfer_id` is unique to a patient physical location. (INTEGER) |
| **transfers.transfer_id** | Identifiers which specify the patient: `subject_id` is unique to a patient, `hadm_id` is unique to a patient hospital stay, and `transfer_id` is unique to a patient physical location. (INTEGER NOT NULL) |
| **transfers.eventtype** | Basic column identifier or value. (VARCHAR(10)) |
| **transfers.careunit** | Basic column identifier or value. (VARCHAR(255)) |
| **transfers.intime** | Basic column identifier or value. (TIMESTAMP(0)) |
| **transfers.outtime** | Basic column identifier or value. (TIMESTAMP(0)) |


## Dataset: `mimiciv_3_1_icu`

| Features (Table.Column) | Description (including Data Type) |
|-------------------------|-----------------------------------|
| **icustays.subject_id** | Unique patient identifier (integer). [1] |
| **icustays.hadm_id** | Hospital admission ID (integer, NULL if no hospital admission). [1] |
| **icustays.stay_id** | Unique ICU stay identifier (integer). [1] |
| **icustays.intime** | ICU admission time (timestamp). [1] |
| **icustays.outtime** | ICU discharge time (timestamp). [1] |
| **icustays.los** | Length of stay in ICU (double). [1] |
| **icustays.first_careunit** | First ICU unit (varchar). [1] |
| **icustays.last_careunit** | Last ICU unit (varchar). [1] |
| **icustays.dischargelocation** | Discharge location from ICU (varchar). [1] |
| **chartevents.stay_id** | ICU stay identifier (integer). [1] |
| **chartevents.subject_id** | Patient identifier (integer). [1] |
| **chartevents.itemid** | Identifier for charted item (integer). [1] |
| **chartevents.charttime** | Time charted (timestamp). [1] |
| **chartevents.value** | Value observed (varchar). [1] |
| **chartevents.valuenum** | Numerical value (double). [1] |
| **chartevents.valueuom** | Unit of measurement (varchar). [1] |
| **chartevents.caregiver_id** | Deidentified care provider ID (integer). [1] |
| **d_items.itemid** | Item identifier (integer). [1] |
| **d_items.label** | Label/description of item (varchar). [1] |
| **d_items.abbreviation** | Abbreviation for item (varchar). [1] |
| **d_items.dhlevel** | Data hierarchy level (varchar). [1] |
| **d_items.category** | Category of item (varchar). [1] |
| **d_items.unitname** | Unit name (varchar). [1] |
| **d_items.param_type** | Parameter type (varchar). [1] |
| **inputevents.stay_id** | ICU stay identifier (integer). [1] |
| **inputevents.subject_id** | Patient identifier (integer). [1] |
| **inputevents.itemid** | Input item identifier (integer). [1] |
| **inputevents.charttime** | Time of input (timestamp). [1] |
| **inputevents.amount** | Amount input (double). [1] |
| **inputevents.amountuom** | Amount unit (varchar). [1] |
| **inputevents.rate** | Rate of input (double). [1] |
| **inputevents.rateuom** | Rate unit (varchar). [1] |
| **inputevents.totalamount** | Total amount (double). [1] |
| **inputevents.totalamountuom** | Total amount unit (varchar). [1] |
| **inputevents.caregiver_id** | Care provider ID (integer). [1] |
| **ingredientevents.stay_id** | ICU stay identifier (integer). [1] |
| **ingredientevents.subject_id** | Patient identifier (integer). [1] |
| **ingredientevents.itemid** | Ingredient item identifier (integer). [1] |
| **ingredientevents.charttime** | Time of ingredient (timestamp). [1] |
| **ingredientevents.amount** | Amount of ingredient (double). [1] |
| **ingredientevents.amountuom** | Amount unit (varchar). [1] |
| **outputevents.stay_id** | ICU stay identifier (integer). [1] |
| **outputevents.subject_id** | Patient identifier (integer). [1] |
| **outputevents.itemid** | Output item identifier (integer). [1] |
| **outputevents.charttime** | Time of output (timestamp). [1] |
| **outputevents.value** | Output value (varchar). [1] |
| **outputevents.valuenum** | Numerical output value (double). [1] |
| **outputevents.valueuom** | Output unit (varchar). [1] |
| **outputevents.isoutput** | Indicates if output (smallint). [1] |
| **outputevents.caregiver_id** | Care provider ID (integer). [1] |
| **procedureevents.stay_id** | ICU stay identifier (integer). [1] |
| **procedureevents.subject_id** | Patient identifier (integer). [1] |
| **procedureevents.itemid** | Procedure item identifier (integer). [1] |
| **procedureevents.charttime** | Time of procedure (timestamp). [1] |
| **procedureevents.value** | Procedure value (varchar). [1] |
| **procedureevents.valuenum** | Numerical value (double). [1] |
| **procedureevents.valueuom** | Unit (varchar). [1] |
| **procedureevents.caregiver_id** | Care provider ID (integer). [1] |
| **datetimeevents.stay_id** | ICU stay identifier (integer). [1] |
| **datetimeevents.subject_id** | Patient identifier (integer). [1] |
| **datetimeevents.itemid** | Datetime item identifier (integer). [1] |
| **datetimeevents.charttime** | Chart time (timestamp). [1] |
| **datetimeevents.value** | Value as datetime (timestamp). [1] |
| **datetimeevents.valueuom** | Unit (varchar). [1] |
| **datetimeevents.caregiver_id** | Care provider ID (integer). [1] |
| **caregiver.caregiver_id** | Deidentified care provider identifier (integer). [1] |
| **caregiver.label** | Description/label of caregiver role (varchar). [1]

---

## Dataset: `mimiciv_3_1_derived`

| Features (Table.Column) | Description (including Data Type) |
|--------------------------|-----------------------------------|
| **sofa.stay_id** | Unique ICU stay identifier (integer). [1] |
| **sofa.subject_id** | Unique patient identifier (integer). [1] |
| **sofa.chartdate** | Date of SOFA score calculation (date). [1] |
| **sofa.sofa_score** | Sequential Organ Failure Assessment score, composite of organ dysfunctions (integer). [1] |
| **sofa.respiration_score** | Respiratory component score (0-4, integer). [1] |
| **sofa.coagulation_score** | Coagulation component score (0-4, integer). [1] |
| **sofa.liver_score** | Liver component score (0-4, integer). [1] |
| **sofa.cardiovascular_score** | Cardiovascular component score (0-4, integer). [1] |
| **sofa.cns_score** | Central nervous system component score (0-4, integer). [1] |
| **sofa.renal_score** | Renal component score (0-4, integer). [1] |
| **sapsii.stay_id** | Unique ICU stay identifier (integer). [1] |
| **sapsii.subject_id** | Unique patient identifier (integer). [1] |
| **sapsii.chartdate** | Date of SAPS II score calculation (date). [1] |
| **sapsii.sapsii_score** | Simplified Acute Physiology Score II, predicts hospital mortality (integer, 0-163). [1] |
| **apsiii.stay_id** | Unique ICU stay identifier (integer). [1] |
| **apsiii.subject_id** | Unique patient identifier (integer). [1] |
| **apsiii.apsiii_score** | Acute Physiology Score III, assesses severity of illness (integer). [1] |
| **oasis.stay_id** | Unique ICU stay identifier (integer). [1] |
| **oasis.subject_id** | Unique patient identifier (integer). [1] |
| **oasis.oasis_score** | Organ dysfunction and/or infection score (integer). [1] |
| **lods.stay_id** | Unique ICU stay identifier (integer). [1] |
| **lods.subject_id** | Unique patient identifier (integer). [1] |
| **lods.lods_score** | Logistic Organ Dysfunction System score (integer). [1] |
| **charlson.stay_id** | Unique ICU stay identifier (integer). [1] |
| **charlson.subject_id** | Unique patient identifier (integer). [1] |
| **charlson.charlson_score** | Charlson Comorbidity Index, weighted sum of comorbidities (integer). [1] |
| **sepsis3.stay_id** | Unique ICU stay identifier (integer). [1] |
| **sepsis3.subject_id** | Unique patient identifier (integer). [1] |
| **sepsis3.sepsis3** | Binary indicator for Sepsis-3 definition met (boolean). [1] |
| **kdigo_stages.stay_id** | Unique ICU stay identifier (integer). [1] |
| **kdigo_stages.subject_id** | Unique patient identifier (integer). [1] |
| **kdigo_stages.chartdate** | Date of KDIGO stage assessment (date). [1] |
| **kdigo_stages.kdigo_stage** | Acute kidney injury stage per KDIGO criteria (integer 1-3 or null). [1] |
| **first_day_vitalsign.stay_id** | Unique ICU stay identifier (integer). [1] |
| **first_day_vitalsign.subject_id** | Unique patient identifier (integer). [1] |
| **first_day_vitalsign.chartdate** | First ICU day date (date). [1] |
| **first_day_vitalsign.heart_rate_min** | Minimum heart rate on first day (float). [1] |
| **first_day_vitalsign.heart_rate_max** | Maximum heart rate on first day (float). [1] |
| **first_day_vitalsign.sysbp_min** | Minimum systolic blood pressure (float). [1] |
| **first_day_lab.stay_id** | Unique ICU stay identifier (integer). [1] |
| **first_day_lab.subject_id** | Unique patient identifier (integer). [1] |
| **first_day_lab.chartdate** | First ICU day date (date). [1] |
| **first_day_lab.creatinine_min** | Minimum creatinine on first day (float). [1] |
| **first_day_lab.sodium_max** | Maximum sodium on first day (float). [1] |
| **first_day_gcs.stay_id** | Unique ICU stay identifier (integer). [1] |
| **first_day_gcs.subject_id** | Unique patient identifier (integer). [1] |
| **first_day_gcs.gcs_min** | Minimum Glasgow Coma Scale total on first day (integer 3-15). [1] |
| **first_day_gcs.gcs_eyes_min** | Minimum eye score (1-4). [1] |
| **first_day_urine_output.stay_id** | Unique ICU stay identifier (integer). [1] |
| **first_day_urine_output.subject_id** | Unique patient identifier (integer). [1] |
| **first_day_urine_output.urine_output** | Total urine output on first day (float, mL). [1] |
| **ventilation.stay_id** | Unique ICU stay identifier (integer). [1] |
| **ventilation.subject_id** | Unique patient identifier (integer). [1] |
| **ventilation.vent_start** | Ventilation start time (timestamp). [1] |
| **ventilation.vent_end** | Ventilation end time (timestamp). [1] |
| **rrt.stay_id** | Unique ICU stay identifier (integer). [1] |
| **rrt.subject_id** | Unique patient identifier (integer). [1] |
| **rrt.rrt_start** | Renal replacement therapy start time (timestamp). [1] |
| **crrt.stay_id** | Unique ICU stay identifier (integer). [1] |
| **crrt.subject_id** | Unique patient identifier (integer). [1] |
| **crrt.crrt_start** | Continuous RRT start time (timestamp). [1] |
| **norepinephrine_equivalent_dose.stay_id** | Unique ICU stay identifier (integer). [1] |
| **norepinephrine_equivalent_dose.subject_id** | Unique patient identifier (integer). [1] |
| **norepinephrine_equivalent_dose.chartdate** | Dose calculation date (date). [1] |
| **norepinephrine_equivalent_dose.norepinephrine_rate** | Norepinephrine equivalent dose (float, mcg/kg/min). [1] |
| **chemistry.stay_id** | Unique ICU stay identifier (integer). [1] |
| **chemistry.subject_id** | Unique patient identifier (integer). [1] |
| **chemistry.chartdate** | Lab measurement date (date). [1] |
| **chemistry.creatinine** | Serum creatinine value (float, mg/dL). [1] |
| **complete_blood_count.stay_id** | Unique ICU stay identifier (integer). [1] |
| **complete_blood_count.subject_id** | Unique patient identifier (integer). [1] |
| **complete_blood_count.chartdate** | Lab measurement date (date). [1] |
| **complete_blood_count.wbc** | White blood cell count (float, x10^3/uL). [1] |
| **coagulation.stay_id** | Unique ICU stay identifier (integer). [1] |
| **coagulation.subject_id** | Unique patient identifier (integer). [1] |
| **coagulation.chartdate** | Lab measurement date (date). [1] |
| **coagulation.pt** | Prothrombin time (float, seconds). [1] |
| **inflammation.stay_id** | Unique ICU stay identifier (integer). [1] |
| **inflammation.subject_id** | Unique patient identifier (integer). [1] |
| **inflammation.chartdate** | Lab measurement date (date). [1] |
| **inflammation.crp** | C-reactive protein level (float, mg/L). [1] |

