from utils.predictor import load_class_names


_DISEASE_GUIDANCE = {
    "Apple": {
        "scab": (
            "A fungal disease that commonly creates dark, scabby lesions on leaves and fruit.",
            "Olive-green or dark lesions, leaf distortion, premature leaf drop.",
        ),
        "black_rot": (
            "A fungal disease that may affect leaves, fruit, and branches.",
            "Brown circular leaf lesions with darker borders and possible fruit decay.",
        ),
        "cedar_apple_rust": (
            "A fungal disease that requires alternate hosts such as juniper and apple.",
            "Bright yellow-orange leaf spots that may later develop small raised structures.",
        ),
    },
    "Cherry": {
        "powdery_mildew": (
            "A fungal disease that produces pale, powdery growth on leaf surfaces.",
            "White or gray powdery patches, leaf curling, and stunted new growth.",
        ),
    },
    "Corn_(maize)": {
        "cercospora_leaf_spot gray_leaf_spot": (
            "A fungal foliar disease favored by warm, humid conditions.",
            "Small rectangular gray-brown lesions restricted by leaf veins.",
        ),
        "common_rust": (
            "A fungal disease that produces rust-colored pustules on leaves.",
            "Scattered cinnamon-brown pustules, often on both leaf surfaces.",
        ),
        "northern_leaf_blight": (
            "A fungal disease associated with humid conditions and prolonged leaf moisture.",
            "Long, gray-green to brown, cigar-shaped lesions on leaves.",
        ),
    },
    "Grape": {
        "black_rot": (
            "A fungal disease that can affect grape leaves and fruit.",
            "Small circular brown leaf spots that enlarge and develop dark margins.",
        ),
        "esca_(black_measles)": (
            "A complex grapevine wood disease associated with multiple fungi.",
            "Tiger-stripe leaf discoloration and possible dark fruit spotting.",
        ),
        "leaf_blight_(isariopsis_leaf_spot)": (
            "A fungal foliar disease that can reduce canopy health.",
            "Angular or irregular brown leaf lesions and leaf yellowing.",
        ),
    },
    "Orange": {
        "haunglongbing_(citrus_greening)": (
            "A serious bacterial disease spread by insects that affects citrus trees.",
            "Blotchy yellow mottling, lopsided fruit, and canopy decline.",
        ),
    },
    "Peach": {
        "bacterial_spot": (
            "A bacterial disease affecting leaves and fruit under favorable conditions.",
            "Small dark leaf spots, yellow halos, and possible leaf tearing.",
        ),
    },
    "Pepper,_bell": {
        "bacterial_spot": (
            "A bacterial disease that commonly affects pepper foliage and fruit.",
            "Small water-soaked spots that become dark and may merge.",
        ),
    },
    "Potato": {
        "early_blight": (
            "A fungal disease that often starts on older potato leaves.",
            "Brown spots with concentric rings and surrounding yellow tissue.",
        ),
        "late_blight": (
            "A serious oomycete disease that can spread rapidly in cool, wet conditions.",
            "Water-soaked lesions that enlarge, with pale edges and possible underside growth.",
        ),
    },
    "Squash": {
        "powdery_mildew": (
            "A fungal disease that colonizes leaf surfaces.",
            "White powdery patches, yellowing leaves, and reduced plant vigor.",
        ),
    },
    "Strawberry": {
        "leaf_scorch": (
            "A fungal disease that causes irregular purple-brown leaf lesions.",
            "Dark purple spots, scorched leaf edges, and possible leaf drying.",
        ),
    },
    "Tomato": {
        "bacterial_spot": (
            "A bacterial disease that affects tomato leaves and fruit.",
            "Small water-soaked spots that become dark and may merge.",
        ),
        "early_blight": (
            "A fungal disease that often begins on lower, older leaves.",
            "Brown lesions with concentric rings and surrounding yellow tissue.",
        ),
        "late_blight": (
            "A serious oomycete disease favored by cool, moist conditions.",
            "Large water-soaked lesions, pale borders, and rapid tissue damage.",
        ),
        "leaf_mold": (
            "A fungal disease that develops most often in humid greenhouse conditions.",
            "Yellow upper-leaf patches with olive or gray growth underneath.",
        ),
        "septoria_leaf_spot": (
            "A fungal foliar disease that commonly affects lower tomato leaves first.",
            "Numerous small circular spots with dark borders and pale centers.",
        ),
        "spider_mites two-spotted_spider_mite": (
            "Damage caused by tiny feeding mites rather than a plant pathogen.",
            "Fine stippling, pale mottling, webbing, and leaf bronzing.",
        ),
        "target_spot": (
            "A fungal disease that creates spreading lesions on tomato foliage.",
            "Round brown lesions with concentric rings and possible yellow halos.",
        ),
        "tomato_yellow_leaf_curl_virus": (
            "A viral disease spread mainly by whiteflies.",
            "Upward curling leaves, yellow margins, stunting, and reduced fruiting.",
        ),
        "tomato_mosaic_virus": (
            "A viral disease that can affect leaf appearance and plant growth.",
            "Mottled light-dark green patterns, leaf distortion, and stunting.",
        ),
    },
}


_GENERAL_MANAGEMENT = [
    "Remove severely affected or fallen leaves only when it is safe and practical.",
    "Improve plant spacing and airflow, and avoid unnecessary overhead irrigation.",
    "Monitor nearby plants and isolate suspected infected material when appropriate.",
    "Use locally approved control methods only after confirming the crop and disease.",
    "Consult a local agricultural officer or certified crop adviser for diagnosis and guidance.",
]


_HEALTHY_MANAGEMENT = [
    "Continue regular inspection for new or changing symptoms.",
    "Maintain appropriate watering, balanced nutrition, and good airflow.",
    "Keep tools clean and remove plant debris as part of routine crop hygiene.",
]


def _normalized_condition(plant_name: str, disease_name: str) -> str:
    """Normalize label text for lookup without changing the source JSON."""
    normalized = disease_name.lower().strip().replace("-", " ").replace("_", " ")
    normalized = " ".join(normalized.split())

    plant_prefix = plant_name.lower().strip().replace("_", " ")
    plant_prefix = " ".join(plant_prefix.split())
    if normalized.startswith(f"{plant_prefix} "):
        normalized = normalized[len(plant_prefix):].strip()

    normalized = normalized.replace(",", "")
    normalized = normalized.replace(" ", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized


def _guidance_plant_key(plant_name: str) -> str:
    """Map display plant labels, including parenthetical variants, to guidance keys."""
    candidate = plant_name.replace(" ", "_")
    for key in _DISEASE_GUIDANCE:
        if candidate.lower() == key.lower():
            return key
    if plant_name.lower().startswith("cherry"):
        return "Cherry"
    return candidate


def parse_class_name(class_name: str) -> tuple[str, str]:
    """Convert a dataset label into display-ready plant and condition names."""
    parts = class_name.split("___", 1)
    plant_key = parts[0].replace("_", " ").strip()
    raw_condition = parts[1].strip() if len(parts) == 2 else "Unknown condition"

    plant_display = plant_key.replace("(including sour)", "(including sour)")
    condition_key = raw_condition.lower().replace(" ", "_").replace("(", "_").replace(")", "_")
    condition_key = "_".join(condition_key.split("_"))
    disease_display = raw_condition.replace("_", " ").replace(" Gray", " Gray").strip()

    if disease_display.lower() == "healthy":
        disease_display = "Healthy"

    return plant_display, disease_display


def get_disease_info(class_name: str) -> dict[str, object]:
    """Return structured, cautious guidance for a supported model class."""
    plant_name, disease_name = parse_class_name(class_name)
    is_healthy = disease_name.lower() == "healthy"

    if is_healthy:
        return {
            "plant_name": plant_name,
            "disease_name": "Healthy leaf",
            "is_healthy": True,
            "description": f"The model classified the leaf as healthy for {plant_name}.",
            "symptoms": "No disease-specific symptoms were identified by this model result.",
            "management": _HEALTHY_MANAGEMENT,
            "disclaimer": (
                "A healthy classification is a model prediction, not a guarantee of plant health."
            ),
        }

    plant_key = _guidance_plant_key(plant_name)
    condition_key = _normalized_condition(plant_name, disease_name)
    plant_key = next(
        (key for key in _DISEASE_GUIDANCE if key.lower() == plant_key.lower()),
        plant_key,
    )

    plant_guidance = _DISEASE_GUIDANCE.get(plant_key, {})
    guidance_key = next(
        (
            key
            for key in plant_guidance
            if _normalized_condition(plant_name, key) == condition_key
        ),
        None,
    )
    guidance = plant_guidance.get(guidance_key) if guidance_key is not None else None
    if guidance is None:
        description = (
            f"{disease_name} was identified as a possible condition for {plant_name}."
        )
        symptoms = "Specific disease symptoms could not be confirmed from this general result."
    else:
        description, symptoms = guidance

    return {
        "plant_name": plant_name,
        "disease_name": disease_name,
        "is_healthy": False,
        "description": description,
        "symptoms": symptoms,
        "management": _GENERAL_MANAGEMENT,
        "disclaimer": (
            "General guidance only. This result is not a laboratory diagnosis. "
            "Consult a local agricultural expert before applying treatments."
        ),
    }


def validate_disease_info_coverage() -> list[str]:
    """Return class names that do not yet have specific disease guidance."""
    missing: list[str] = []
    for class_name in load_class_names():
        if parse_class_name(class_name)[1].lower() == "healthy":
            continue
        _, disease_name = parse_class_name(class_name)
        plant_name, _ = parse_class_name(class_name)
        plant_key = _guidance_plant_key(plant_name)
        condition_key = _normalized_condition(plant_name, disease_name)
        plant_guidance = _DISEASE_GUIDANCE.get(plant_key, {})
        has_guidance = any(
            _normalized_condition(plant_name, key) == condition_key
            for key in plant_guidance
        )
        if not has_guidance:
            missing.append(class_name)
    return missing
