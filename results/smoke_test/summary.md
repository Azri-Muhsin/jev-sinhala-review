# Stage 2 Phase 0 Smoke Test — Gate Verification Report

**Gate Status:** PASSED
**Timestamp:** 2026-10-03T11:06:52.471671+00:00
**Total Model Decisions Logged:** 320

## Verification Checklist

- [x] Check 1: Zero runtime errors across English and Sinhala templates
- [x] Check 2: Bidirectional label mappings valid
- [x] Check 3: Raw response deserialization (Choice, Noul, Score)
- [x] Check 4: Score behavior validated on 1-5 rubrics & 3-class sentiment
- [x] Check 5: Zero Unicode/ZWJ corruption confirmed (109 interactions)

## Per-Task Smoke Results

| Task | Records Logged | Primitives Tested |
| :--- | :---: | :--- |
| `dataset_a_sentiment` | 60 | choice, noul, score |
| `dataset_b_sold` | 60 | choice, noul, score |
| `dataset_c1_nsina_categories` | 40 | choice, noul |
| `dataset_c2_nsina_media` | 20 | choice |
| `dataset_d_sinhalammlu` | 20 | choice |
| `dataset_e_salangabhava` | 60 | choice, noul, score |
| `dataset_f_cmcs` | 60 | choice, noul |
