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
    "acne": "The visual pattern may be similar to acne, which commonly involves blocked or inflamed pores.",
    "eczema": "The visual pattern may be similar to eczema or dermatitis, which can involve dry, itchy, or inflamed skin.",
    "tinea": "The visual pattern may be similar to a superficial fungal skin condition such as tinea.",
    "scabies": "The visual pattern may be similar to scabies, an itchy skin condition that requires professional assessment.",
    "psoriasis": "The visual pattern may be similar to psoriasis, which can cause raised, scaly, or inflamed patches.",
    "other": "The image does not closely match one supported category and may represent another skin condition.",
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
        next_steps = (
            "Avoid scratching or irritating the area and consult a qualified health "
            "professional for an accurate assessment."
        )

    return Explanation(summary=summary, next_steps=next_steps, warning=URGENT_WARNING)
