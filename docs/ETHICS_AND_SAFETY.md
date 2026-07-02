# Ethics and Safety

## Intended role

DermaScan AI is an educational and decision-support project. It may indicate
visual similarity to supported skin-condition categories, but it does not
establish a diagnosis or replace a healthcare professional.

## Required user-facing statement

The scan result is not a diagnosis. For accurate diagnosis and treatment
recommendations, consult a qualified healthcare professional or dermatologist.

This statement appears on the landing, scan, result, API response, and PDF
surfaces.

## Implemented safeguards

- Explicit consent before analysis
- File-size, file-format, dimension, blur, and brightness validation
- Uncertain output below the configured confidence threshold
- Top-three possibilities instead of forced certainty
- Rule-based explanations that avoid diagnosis and medication advice
- Urgent-care guidance for rapid spread, severe pain, fever, bleeding, open
  wounds, or infection signs
- EXIF removal through RGB re-encoding
- Bounded in-memory image retention rather than permanent storage
- Placeholder and smoke-model status exposed in both API and UI
- Smoke checkpoints blocked from checkpoint inference by default

## Known risks

- Skin conditions can look similar in photographs.
- Lighting, camera processing, body location, and skin tone can change visual
  appearance.
- Dataset labels are differential dermatologist assessments, not uniformly
  biopsy-confirmed diagnoses.
- Current data originated in the United States and does not establish validity
  for Ghanaian users.
- Scabies and acne have small class counts.
- Confidence scores are not guaranteed to be clinically calibrated.

## Prohibited claims and behavior

- Do not say a diagnosis is confirmed.
- Do not tell a user that they have a disease.
- Do not prescribe medication.
- Do not suppress professional follow-up based on model confidence.
- Do not enable research image retention without separate informed consent,
  governance, access control, and a documented retention policy.

## Requirements before a real pilot

1. Obtain ethics and institutional review appropriate to the study context.
2. Add representative Ghanaian skin-tone and phone-camera evaluation data.
3. Increase minority-class data, especially scabies.
4. Complete full training and held-out plus external evaluation.
5. Review errors and wording with qualified dermatology professionals.
6. Calibrate confidence thresholds on validation data.
7. Conduct usability, accessibility, privacy, and security review.
