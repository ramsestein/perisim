# Schema — mimic

Path: `C:\Users\Ramsés\Desktop\Proyectos\perism\data\mimic4`

Status: OK

| table | file | found | missing columns | missing optional |
|---|---|---|---|---|
| admissions | hosp/admissions | yes | — | — |
| patients | hosp/patients | yes | — | — |
| icustays | icu/icustays | yes | — | — |
| transfers | hosp/transfers | yes | — | — |
| services | hosp/services | yes | — | — |
| procedures_icd | hosp/procedures_icd | yes | — | — |
| labevents | hosp/labevents | yes | — | — |
| chartevents | icu/chartevents | yes | — | — |
| inputevents | icu/inputevents | yes | — | — |
| procedureevents | icu/procedureevents | yes | — | — |
| code_status | icu/chartevents | yes | — | — |
| emar | hosp/emar | yes | — | — |
| prescriptions | hosp/prescriptions | yes | — | — |
| d_items | icu/d_items | yes | — | — |
| d_labitems | hosp/d_labitems | yes | — | — |

- **admissions** (19.9 MB): subject_id, hadm_id, admittime, dischtime, deathtime, admission_type, admit_provider_id, admission_location, discharge_location, insurance, language, marital_status, race, edregtime, edouttime, hospital_expire_flag
- **patients** (2.8 MB): subject_id, gender, anchor_age, anchor_year, anchor_year_group, dod
- **icustays** (3.3 MB): subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los
- **transfers** (46.2 MB): subject_id, hadm_id, transfer_id, eventtype, careunit, intime, outtime
- **services** (8.6 MB): subject_id, hadm_id, transfertime, prev_service, curr_service
- **procedures_icd** (7.8 MB): subject_id, hadm_id, seq_num, chartdate, icd_code, icd_version
- **labevents** (2592.9 MB): labevent_id, subject_id, hadm_id, specimen_id, itemid, order_provider_id, charttime, storetime, value, valuenum, valueuom, ref_range_lower, ref_range_upper, flag, priority, comments
- **chartevents** (3502.4 MB): subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning
- **inputevents** (401.1 MB): subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, amount, amountuom, rate, rateuom, orderid, linkorderid, ordercategoryname, secondaryordercategoryname, ordercomponenttypedescription, ordercategorydescription, patientweight, totalamount, totalamountuom, isopenbag, continueinnextdept, statusdescription, originalamount, originalrate
- **procedureevents** (24.1 MB): subject_id, hadm_id, stay_id, caregiver_id, starttime, endtime, storetime, itemid, value, valueuom, location, locationcategory, orderid, linkorderid, ordercategoryname, ordercategorydescription, patientweight, isopenbag, continueinnextdept, statusdescription, originalamount, originalrate
- **code_status** (3502.4 MB): subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning
- **emar** (811.3 MB): subject_id, hadm_id, emar_id, emar_seq, poe_id, pharmacy_id, enter_provider_id, charttime, medication, event_txt, scheduletime, storetime
- **prescriptions** (606.3 MB): subject_id, hadm_id, pharmacy_id, poe_id, poe_seq, order_provider_id, starttime, stoptime, drug_type, drug, formulary_drug_cd, gsn, ndc, prod_strength, form_rx, dose_val_rx, dose_unit_rx, form_val_disp, form_unit_disp, doses_per_24_hrs, route
- **d_items** (0.1 MB): itemid, label, abbreviation, linksto, category, unitname, param_type, lownormalvalue, highnormalvalue
- **d_labitems** (0.0 MB): itemid, label, fluid, category

## Labels of the configured itemids (MIMIC-IV)

- lab_creatinine: 50912 = Creatinine
- lab_hb: 51222 = Hemoglobin
- lab_wbc: 51301 = White Blood Cells; 51300 = WBC Count
- lab_plt: 51265 = Platelet Count
- lab_lactate: 50813 = Lactate
- lab_bilirubin: 50885 = Bilirubin, Total
- lab_albumin: 50862 = Albumin
- lab_ph: 50820 = pH
- lab_pco2: 50818 = pCO2
- lab_po2: 50821 = pO2
- vital_hr: 220045 = Heart Rate
- vital_map: 220052 = Arterial Blood Pressure mean; 220181 = Non Invasive Blood Pressure mean
- vital_rr: 220210 = Respiratory Rate
- vital_spo2: 220277 = O2 saturation pulseoxymetry
- vital_temp: 223762 = Temperature Celsius
- vital_temp_f: 223761 = Temperature Fahrenheit
- vital_fio2: 223835 = Inspired O2 Fraction
- vital_gcs: 220739 = GCS - Eye Opening; 223900 = GCS - Verbal Response; 223901 = GCS - Motor Response
- anthro_weight: 226512 = Admission Weight (Kg)
- anthro_height: 226730 = Height (cm)
- support_vent: 225792 = Invasive Ventilation
- support_vaso: 221906 = Norepinephrine; 221289 = Epinephrine; 221749 = Phenylephrine; 222315 = Vasopressin; 221662 = Dopamine
- support_rrt: 225802 = Dialysis - CRRT; 225803 = Dialysis - CVVHD; 225805 = Peritoneal Dialysis; 225809 = Dialysis - CVVHDF; 225955 = Dialysis - SCUF; 225441 = Hemodialysis

## Inventory `services.curr_service` (top 20)

MED (286528), CMED (54452), SURG (51640), OMED (33549), ORTHO (26991), NMED (26709), OBS (24923), NSURG (16795), CSURG (13421), VSURG (12385), PSYCH (9592), TRAUM (8961), GYN (7935), GU (7083), TSURG (5767), PSURG (4427), ENT (1798), DENT (59), EYE (54), (2 other values with n < 10) (None)

## Inventory `admissions.admission_type` (top 9)

EW EMER. (177459), EU OBSERVATION (119456), OBSERVATION ADMIT (84437), URGENT (54929), SURGICAL SAME DAY ADMISSION (42898), DIRECT OBSERVATION (24551), DIRECT EMER. (21973), ELECTIVE (13130), AMBULATORY OBSERVATION (7195)

## Inventory `admissions.discharge_location` (top 14)

HOME (194204), None (149818), HOME HEALTH CARE (99305), SKILLED NURSING FACILITY (52657), REHAB (13845), DIED (11721), CHRONIC/LONG TERM ACUTE CARE (8125), HOSPICE (5397), AGAINST ADVICE (3393), PSYCH FACILITY (2965), ACUTE HOSPITAL (2334), OTHER FACILITY (1592), ASSISTED LIVING (622), HEALTHCARE FACILITY (50)

## Inventory `admissions.admission_location` (top 12)

EMERGENCY ROOM (244179), PHYSICIAN REFERRAL (163228), TRANSFER FROM HOSPITAL (56227), WALK-IN/SELF REFERRAL (42365), CLINIC REFERRAL (12965), PROCEDURE SITE (8518), TRANSFER FROM SKILLED NURSING FACILITY (6317), INTERNAL TRANSFER TO OR FROM PSYCH (5837), PACU (5734), INFORMATION NOT AVAILABLE (402), AMBULATORY SURGERY TRANSFER (255), (1 other values with n < 10) (None)

## Inventory `prescriptions.route` (top 88)

IV (7964934), PO/NG (4894589), PO (3507466), SC (1355102), IV DRIP (725586), IM (420954), IH (278073), TP (179790), PR (177784), TD (94335), ORAL (92243), IVPCA (82827), BOTH EYES (60971), DWELL (56731), NU (54640), NG (48277), NEB (45304), IV BOLUS (34841), ED (34043), SL (31482), DIALYS (26220), REPLACE (25234), ID (22367), RIGHT EYE (8331), LEFT EYE (8103), None (6439), PB (6262), SUBCUT (5629), LOCK (5076), IT (4952), PERIPHNERVE (4695), PL (4013), INHALATION (3411), OU (3084), IP (2634), VG (2385), INTRAPLEURAL (1329), NAS (1182), IMPELLA (1011), OD (718), BOTH EARS (703), FLUSH (641), IVCI (635), OS (586), IN (526), LEFT EAR (518), RIGHT EAR (509), G TUBE (495), IJ (435), AD (404), IVT (377), IRR (366), IR (351), TRANSDERMAL (343), J TUBE (306), NS (239), IV INFUSION (221), LUMBAR PLEXUS (132), FEMORAL VEIN (132), IA (122), BU (118), INTRAVITREAL (116), AU (98), INTERSPACE (86), TT (86), INTRA-ARTERIALLY (83), NG/OG (79), SCPUMP (72), BUCCAL (72), IO (70), OG (68), INTRALESIONAL (61), IT IMPLANTED (57), AS (51), SUBDERMAL (51), ND (48), IVS (34), PO/OG (33), SUBCUT INFUSION (27), INFUSION VIA ON-Q PUMP (24), ET (20), IL (19), IC (18), INTRAPERICARDIAL (18), INTRACORPOREAL (14), SC INFUSION (12), ENTERAL TUBE ONLY – NOT ORAL (11), (22 other values with n < 10) (None)
