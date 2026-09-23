"""
classifier_tool.py — TensorFlow/Keras Skin Disease Classifier (LangChain Tool)
=================================================================================

Wraps the actual trained DermaAI MobileNetV2 model (TensorFlow Hub backbone
with Dense(8) head) into a clean LangChain tool for the agent pipeline.

Model Architecture (mirrors Backend/app/services/prediction_services.py):
    tf2-preview/mobilenet_v2/feature_vector/4 → Dense(8, softmax)

Weights File:
    Backend/models/my_model_weights.h5

Classes (8 skin diseases):
    0: Cellulitis
    1: Impetigo
    2: Athlete Foot
    3: Nail Fungus
    4: Ringworm
    5: Cutaneous Larva Migrans
    6: Chickenpox
    7: Shingles

Author  : DermaAgent Assessment Module
License : MIT
"""

import os
import sys
import logging
from typing import Optional, Dict, List, Any

import numpy as np
import cv2
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Suppress TensorFlow verbose logging (must be set before import)
# ---------------------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

# Further suppress TF warnings after import
tf.get_logger().setLevel("ERROR")

# ---------------------------------------------------------------------------
# Class Label Mapping — Actual DermaAI Training Classes (8 classes)
# ---------------------------------------------------------------------------
CLASS_LABELS: Dict[int, Dict[str, str]] = {
    0: {"abbreviation": "cell",  "name": "Cellulitis"},
    1: {"abbreviation": "imp",   "name": "Impetigo"},
    2: {"abbreviation": "af",    "name": "Athlete Foot"},
    3: {"abbreviation": "nf",    "name": "Nail Fungus"},
    4: {"abbreviation": "rw",    "name": "Ringworm"},
    5: {"abbreviation": "clm",   "name": "Cutaneous Larva Migrans"},
    6: {"abbreviation": "cp",    "name": "Chickenpox"},
    7: {"abbreviation": "sh",    "name": "Shingles"},
}

NUM_CLASSES = len(CLASS_LABELS)

# Clinical severity tiers for triage recommendations
SEVERITY_MAP: Dict[str, str] = {
    "cell": "HIGH — Bacterial skin infection. Antibiotics likely needed. Consult dermatologist.",
    "imp":  "HIGH — Highly contagious bacterial infection. Requires medical treatment.",
    "af":   "MODERATE — Fungal infection. Topical antifungal treatment recommended.",
    "nf":   "MODERATE — Fungal nail infection. Prolonged antifungal therapy may be needed.",
    "rw":   "MODERATE — Fungal infection. Topical/oral antifungal treatment recommended.",
    "clm":  "HIGH — Parasitic infection. Medical treatment required (antiparasitic agents).",
    "cp":   "HIGH — Viral infection (Varicella-zoster). Monitor for complications.",
    "sh":   "HIGH — Viral reactivation (Herpes zoster). Antiviral treatment recommended urgently.",
}

# ---------------------------------------------------------------------------
# Path resolution — locate the real trained weights
# ---------------------------------------------------------------------------
_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.dirname(_AGENT_ROOT)  # DermaAI project root

# Primary: exact path from the Backend
_DEFAULT_WEIGHTS_H5 = os.path.join(_REPO_ROOT, "Backend", "models", "my_model_weights.h5")
_DEFAULT_MODEL_KERAS = os.path.join(_REPO_ROOT, "Backend", "models", "my_model.keras")

# TF Hub feature extractor URL (same as Backend/app/services/prediction_services.py)
_TF_HUB_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

# ---------------------------------------------------------------------------
# Model Singleton
# ---------------------------------------------------------------------------
_model_singleton: Optional[tf.keras.Model] = None


def load_model(
    weights_path: Optional[str] = None,
    keras_path: Optional[str] = None,
    force_reload: bool = False,
) -> tf.keras.Model:
    """
    Load the trained DermaAI TensorFlow/Keras model.

    Attempts to load in this priority order:
        1. Full saved model from ``.keras`` file (preserves architecture)
        2. Reconstruct architecture from TF Hub + load ``.h5`` weights
        3. If neither found, reconstruct with random weights (demo fallback)

    Parameters
    ----------
    weights_path : str, optional
        Path to ``.h5`` weights file. Auto-resolved if not provided.
    keras_path : str, optional
        Path to full ``.keras`` saved model. Auto-resolved if not provided.
    force_reload : bool, optional
        If True, reload even if cached.

    Returns
    -------
    model : tf.keras.Model
        Compiled Keras model ready for inference.
    """
    global _model_singleton

    if _model_singleton is not None and not force_reload:
        logger.debug("Returning cached TF model singleton.")
        return _model_singleton

    # Resolve paths
    keras_path = keras_path or os.environ.get("DERMA_MODEL_KERAS", _DEFAULT_MODEL_KERAS)
    weights_path = weights_path or os.environ.get("DERMA_MODEL_WEIGHTS", _DEFAULT_WEIGHTS_H5)

    # ---- Strategy 1: Load full .keras saved model ----
    if keras_path and os.path.isfile(keras_path):
        try:
            logger.info("Loading full .keras model from: %s", keras_path)
            import tensorflow_hub as hub
            model = tf.keras.models.load_model(
                keras_path,
                custom_objects={"KerasLayer": hub.KerasLayer}
            )
            logger.info("✓ Full .keras model loaded successfully.")
            _model_singleton = model
            return model
        except Exception as e:
            logger.warning("Failed to load .keras model: %s. Trying .h5 weights.", e)

    # ---- Strategy 2: Reconstruct architecture + load .h5 weights ----
    if weights_path and os.path.isfile(weights_path):
        try:
            logger.info("Reconstructing model architecture from TF Hub...")
            import tensorflow_hub as hub
            import tempfile, shutil

            try:
                feature_extractor_layer = hub.KerasLayer(
                    _TF_HUB_URL,
                    input_shape=(224, 224, 3),
                    trainable=False,
                )
            except Exception as cache_err:
                logger.warning("TF Hub load failed (%s). Clearing corrupt cache and retrying...", cache_err)
                tfhub_cache = os.path.join(tempfile.gettempdir(), "tfhub_modules")
                if os.path.exists(tfhub_cache):
                    shutil.rmtree(tfhub_cache, ignore_errors=True)
                feature_extractor_layer = hub.KerasLayer(
                    _TF_HUB_URL,
                    input_shape=(224, 224, 3),
                    trainable=False,
                )

            model = tf.keras.Sequential([
                feature_extractor_layer,
                tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
            ])

            # Build the model (required before loading weights)
            model.build((None, 224, 224, 3))

            logger.info("Loading trained weights from: %s", weights_path)
            model.load_weights(weights_path)
            logger.info("✓ Trained weights loaded successfully.")

            _model_singleton = model
            return model

        except Exception as e:
            logger.error("Failed to load .h5 weights: %s", e)
            raise RuntimeError(
                f"Cannot load trained model. Weights file exists at '{weights_path}' "
                f"but loading failed: {e}"
            ) from e

    # ---- Strategy 3: No weights found — raise clear error ----
    raise FileNotFoundError(
        f"Trained model weights not found.\n"
        f"  Checked .keras path: {keras_path}\n"
        f"  Checked .h5 path:    {weights_path}\n\n"
        f"Please ensure your trained model files exist at:\n"
        f"  {_DEFAULT_MODEL_KERAS}\n"
        f"  {_DEFAULT_WEIGHTS_H5}\n"
        f"Or set DERMA_MODEL_WEIGHTS / DERMA_MODEL_KERAS env vars."
    )


# ---------------------------------------------------------------------------
# Inference Preprocessing (matches Backend/app/utils/image_utils.py)
# ---------------------------------------------------------------------------

def preprocess_image_from_path(image_path: str) -> np.ndarray:
    """
    Load and preprocess an image for inference.

    Replicates the exact preprocessing from the Backend:
        cv2.imread → BGR→RGB → resize(224,224) → scale /255.0 → expand_dims

    Parameters
    ----------
    image_path : str
        Path to the input image.

    Returns
    -------
    np.ndarray
        Preprocessed image array of shape ``(1, 224, 224, 3)``
        with pixel values in ``[0, 1]``.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    scaled = resized / 255.0
    batched = np.expand_dims(scaled, axis=0).astype(np.float32)

    logger.debug("Preprocessed image shape: %s, dtype: %s", batched.shape, batched.dtype)
    return batched


def preprocess_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess image from raw bytes (for Gradio uploads).

    Parameters
    ----------
    image_bytes : bytes
        Raw image file bytes.

    Returns
    -------
    np.ndarray
        Preprocessed image array of shape ``(1, 224, 224, 3)``.
    """
    image_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img_rgb, (224, 224))
    scaled = resized / 255.0
    return np.expand_dims(scaled, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Core Inference Function
# ---------------------------------------------------------------------------

def classify_image(
    image_path: str,
    model: Optional[tf.keras.Model] = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Classify a skin disease image using the trained DermaAI model.

    Parameters
    ----------
    image_path : str
        Path to the input skin image.
    model : tf.keras.Model, optional
        Pre-loaded model. If None, loads the singleton.
    top_k : int, optional
        Number of top predictions to return. Default is 3.

    Returns
    -------
    result : dict
        Dictionary containing:
        - ``predictions``: List of top-k prediction dicts
        - ``top_class_index``: int
        - ``top_class_name``: str
        - ``top_abbreviation``: str
        - ``top_confidence``: float (percentage)
        - ``severity``: str
        - ``all_probabilities``: List[float]
    """
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    if model is None:
        model = load_model()

    # ---- Preprocess ----
    input_data = preprocess_image_from_path(image_path)

    # ---- Inference ----
    predictions = model.predict(input_data, verbose=0)
    probs = predictions[0]  # shape: (NUM_CLASSES,)

    # ---- Extract Top-K ----
    top_k_indices = probs.argsort()[::-1][:top_k]
    pred_list: List[Dict[str, Any]] = []

    for rank, idx in enumerate(top_k_indices, start=1):
        idx = int(idx)
        label_info = CLASS_LABELS.get(idx, {"abbreviation": f"unk_{idx}", "name": f"Unknown ({idx})"})
        pred_list.append({
            "rank": rank,
            "class_index": idx,
            "class_name": label_info["name"],
            "abbreviation": label_info["abbreviation"],
            "confidence_pct": round(float(probs[idx]) * 100, 2),
        })

    top_idx = int(top_k_indices[0])
    top_label = CLASS_LABELS.get(top_idx, {"abbreviation": "unknown", "name": "Unknown"})
    severity = SEVERITY_MAP.get(top_label["abbreviation"], "UNKNOWN — Manual review needed.")

    result = {
        "predictions": pred_list,
        "top_class_index": top_idx,
        "top_class_name": top_label["name"],
        "top_abbreviation": top_label["abbreviation"],
        "top_confidence": round(float(probs[top_idx]) * 100, 2),
        "severity": severity,
        "all_probabilities": [round(float(p) * 100, 2) for p in probs],
    }

    logger.info(
        "Classification result: %s (%.1f%%) — Severity: %s",
        result["top_class_name"],
        result["top_confidence"],
        severity.split("—")[0].strip(),
    )
    return result


# ---------------------------------------------------------------------------
# LangChain Tool Definition
# ---------------------------------------------------------------------------

@tool
def skin_lesion_classifier(image_path: str) -> str:
    """Classify a skin disease image using the trained DermaAI MobileNetV2 model.
    The model was trained on 8 skin disease classes: Cellulitis, Impetigo, Athlete
    Foot, Nail Fungus, Ringworm, Cutaneous Larva Migrans, Chickenpox, and Shingles.
    Returns Top-3 predicted diseases with confidence percentages and severity assessment.

    Args:
        image_path: Absolute or relative file path to the skin image
                    (supports JPEG, PNG, BMP formats).

    Returns:
        A formatted string containing the classification results with
        Top-3 predictions, confidence scores, and severity triage.
    """
    try:
        # ---- Handle JSON-string inputs from ReAct text-based agents ----
        import json as _json
        if isinstance(image_path, str) and image_path.strip().startswith("{"):
            try:
                parsed = _json.loads(image_path)
                if isinstance(parsed, dict):
                    image_path = parsed.get("image_path", image_path)
            except (ValueError, _json.JSONDecodeError):
                pass  # Not JSON — use original value

        result = classify_image(image_path)

        lines = [
            "═══ SKIN DISEASE CLASSIFICATION REPORT ═══",
            f"Image: {os.path.basename(image_path)}",
            "",
            "Top-3 Diagnostic Predictions:",
            "─" * 45,
        ]

        for pred in result["predictions"]:
            marker = "►" if pred["rank"] == 1 else " "
            lines.append(
                f"  {marker} #{pred['rank']}: {pred['class_name']} "
                f"({pred['abbreviation']}) — {pred['confidence_pct']:.1f}% "
                f"[class_index={pred['class_index']}]"
            )

        lines.extend([
            "─" * 45,
            f"Primary Diagnosis: {result['top_class_name']}",
            f"Confidence: {result['top_confidence']:.1f}%",
            f"Class Index: {result['top_class_index']}",
            f"Clinical Severity: {result['severity']}",
            "═" * 45,
        ])

        return "\n".join(lines)

    except FileNotFoundError:
        return f"ERROR: Image file not found at '{image_path}'. Please provide a valid file path."
    except Exception as e:
        logger.exception("Classification failed for '%s'", image_path)
        return f"ERROR: Classification failed — {type(e).__name__}: {str(e)}"


# ---------------------------------------------------------------------------
# Module Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running classifier_tool self-test...")

    # Try to load the real model
    model = load_model()
    print(f"Model loaded: {type(model).__name__}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    # Create a dummy test image
    test_dir = os.path.join(_AGENT_ROOT, "outputs")
    os.makedirs(test_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "_test_dummy.png")

    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(test_image_path, dummy_img)

    result = classify_image(test_image_path)
    print("\n=== Raw Result ===")
    for k, v in result.items():
        if k != "all_probabilities":
            print(f"  {k}: {v}")

    print("\n=== LangChain Tool Output ===")
    print(skin_lesion_classifier.invoke({"image_path": test_image_path}))

    os.remove(test_image_path)
    logger.info("Self-test PASSED.")
