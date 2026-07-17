from backend.app.schemas import Explanation


DISCLAIMER = (
    "The scan result is not a diagnosis. For accurate diagnosis and treatment "
    "recommendations, consult a qualified healthcare professional or dermatologist."
)

URGENT_WARNING = (
    "Seek urgent medical care for a rapidly spreading rash, severe pain, fever, "
    "bleeding, an open wound, or signs of infection."
)

SUMMARIES = {
    "acne": (
        "The image has visual features that may resemble acne. Acne commonly involves blocked or inflamed pores "
        "and may show blackheads, whiteheads, pimples, oily skin, or scarring. The AI is looking at visible texture, "
        "color change, and clustered bump-like patterns, but it cannot confirm the cause from an image alone."
    ),
    "scabies": (
        "The image has visual features that may resemble scabies. Scabies commonly causes intense itching and small "
        "bumps or thin burrow-like lines, often around the fingers, wrists, waist, or other skin folds. An image alone "
        "cannot confirm an infestation because eczema, bites, and other rashes may look similar."
    ),
    "psoriasis": (
        "The image has visual features that may resemble psoriasis. Psoriasis can cause raised, scaly, itchy, "
        "or inflamed patches and may appear red, purple, brown, or silvery depending on skin tone. The AI is looking "
        "for plaque-like texture, scaling, and sharply visible patch patterns."
    ),
    "other": "The image does not closely match one supported category and may represent another skin condition.",
}

NEXT_STEPS = {
    "acne": (
        "Use gentle skin care, avoid squeezing or picking lesions, and consider professional review if acne is painful, "
        "widespread, scarring, or not improving. A clinician may recommend topical treatments such as benzoyl peroxide, "
        "retinoids, or other medicines depending on severity."
    ),
    "scabies": (
        "Avoid close skin contact or sharing clothing and bedding until a qualified clinician reviews the rash. Seek "
        "medical advice promptly if several household members are itching, symptoms are worse at night, or the rash is "
        "spreading. Effective treatment requires professional confirmation and treatment of close contacts when advised."
    ),
    "psoriasis": (
        "Avoid scratching or picking scales, note possible triggers such as stress or skin injury, and consult a clinician "
        "if patches persist, spread, become painful, or joint pain occurs. Treatment choice depends on severity and location."
    ),
}


def build_explanation(class_name: str | None, uncertain: bool) -> Explanation:
    if uncertain or class_name is None:
        summary = (
            "The screening result is uncertain. Several conditions may have similar "
            "visual features, and the image alone is not enough to distinguish them safely."
        )
        next_steps = "Try a clearer photo and arrange a professional medical review if the concern continues."
    else:
        summary = SUMMARIES.get(class_name, SUMMARIES["other"])
        next_steps = NEXT_STEPS.get(
            class_name,
            "Avoid scratching or irritating the area and consult a qualified health professional for an accurate assessment.",
        )

    return Explanation(summary=summary, next_steps=next_steps, warning=URGENT_WARNING)
