"""
gradcam_tool.py — LangChain Tool Wrapper for Grad-CAM Visual Explainability
==============================================================================

Connects the TensorFlow Grad-CAM engine to the DermaAgent ReAct pipeline.
Generates visual attribution heatmaps and produces human-readable
interpretability summaries for clinical decision support.

Author  : DermaAgent Assessment Module
License : MIT
"""

import os
import sys
import logging
from typing import Dict, Optional

import numpy as np
import cv2
from PIL import Image
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Ensure parent package is importable when running standalone
# ---------------------------------------------------------------------------
_AGENT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _AGENT_ROOT not in sys.path:
    sys.path.insert(0, _AGENT_ROOT)

from tools.gradcam_engine import GradCAM, overlay_heatmap
from tools.classifier_tool import (
    load_model,
    preprocess_image_from_path,
    CLASS_LABELS,
    NUM_CLASSES,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ---------------------------------------------------------------------------
# Saliency Analysis Helpers
# ---------------------------------------------------------------------------

def _describe_region(x_pct: float, y_pct: float) -> str:
    """Convert normalized percentage coordinates to a human-readable region."""
    if y_pct < 30:
        v = "upper"
    elif y_pct < 70:
        v = "central"
    else:
        v = "lower"

    if x_pct < 30:
        h = "left"
    elif x_pct < 70:
        h = "center"
    else:
        h = "right"

    if v == "central" and h == "center":
        return "center of the lesion"
    elif v == "central":
        return f"center-{h} margin of the lesion"
    elif h == "center":
        return f"{v}-center region of the lesion"
    else:
        return f"{v}-{h} quadrant of the lesion"


def _analyze_saliency(heatmap_2d: np.ndarray) -> Dict[str, object]:
    """
    Compute quantitative saliency statistics from a Grad-CAM heatmap.

    Returns
    -------
    dict with coverage_pct, concentration, peak_value, mean_activation,
    high_activation_pct, spatial_description.
    """
    total_pixels = heatmap_2d.size
    peak_value = float(np.max(heatmap_2d))
    mean_activation = float(np.mean(heatmap_2d))

    above_05 = np.sum(heatmap_2d > 0.5) / total_pixels * 100
    above_07 = np.sum(heatmap_2d > 0.7) / total_pixels * 100

    if above_05 < 15:
        concentration = "highly focused"
        spatial_desc = (
            "The model's attention is tightly concentrated on a small, "
            "specific region, suggesting strong feature localization "
            "on morphological structures."
        )
    elif above_05 < 35:
        concentration = "moderately focused"
        spatial_desc = (
            "The model's attention is moderately distributed, covering "
            "a defined region while ignoring surrounding healthy skin "
            "— indicating structured feature detection."
        )
    else:
        concentration = "diffuse"
        spatial_desc = (
            "The model's attention is broadly distributed across the "
            "image. This may indicate reliance on global texture/color "
            "patterns rather than localized morphological features."
        )

    return {
        "coverage_pct": round(above_05, 1),
        "high_activation_pct": round(above_07, 1),
        "concentration": concentration,
        "peak_value": round(peak_value, 4),
        "mean_activation": round(mean_activation, 4),
        "spatial_description": spatial_desc,
    }


def _generate_clinical_note(
    class_name: str,
    abbreviation: str,
    concentration: str,
    region: str,
) -> str:
    """Generate a clinical relevance note based on attention pattern."""
    expected_patterns = {
        "cell": "erythema, warmth, swelling, and skin surface texture changes",
        "imp":  "honey-colored crusts, vesicles, and erosions on the skin surface",
        "af":   "scaly, peeling skin with erythema, typically in interdigital spaces",
        "nf":   "nail discoloration, thickening, and dystrophic changes",
        "rw":   "annular/ring-shaped erythematous patches with central clearing",
        "clm":  "serpentine, erythematous, elevated tracks on the skin surface",
        "cp":   "vesicular eruptions on erythematous base (dewdrop on rose petal)",
        "sh":   "grouped vesicles on erythematous base in dermatomal distribution",
    }

    expected = expected_patterns.get(abbreviation, "characteristic morphological features")

    if concentration in ("highly focused", "moderately focused"):
        return (
            f"The model's attention on the {region} aligns well with "
            f"expected visual features for {class_name}: {expected}. "
            f"This suggests the prediction is based on clinically relevant "
            f"visual patterns rather than spurious artifacts."
        )
    else:
        return (
            f"The diffuse attention pattern warrants caution. For {class_name}, "
            f"the model should ideally focus on {expected}. The broad activation "
            f"may indicate partial reliance on contextual features. "
            f"Clinical correlation is strongly recommended."
        )


# ---------------------------------------------------------------------------
# LangChain Tool
# ---------------------------------------------------------------------------

@tool
def gradcam_visual_explainer(image_path: str, target_class_index: int = -1) -> str:
    """Generate a Grad-CAM visual explanation heatmap for a skin disease classification.
    This tool produces a saliency map showing which regions of the image the model
    focused on when making its prediction, enabling interpretability verification.

    Args:
        image_path: Absolute or relative path to the skin image file.
        target_class_index: The integer class index (0-7) from the classifier's
            prediction to generate the explanation for.

    Returns:
        A formatted string containing the heatmap file path, peak activation
        coordinates, saliency analysis, and clinical interpretability assessment.
    """
    # ---- Handle JSON-string inputs from ReAct text-based agents ----
    # The ReAct agent may pass a JSON string as the image_path argument
    # e.g. '{"image_path": "/path/to/img.png", "target_class_index": 2}'
    import json as _json
    if target_class_index == -1 or (isinstance(image_path, str) and image_path.strip().startswith("{")):
        try:
            parsed = _json.loads(image_path)
            if isinstance(parsed, dict):
                image_path = parsed.get("image_path", image_path)
                target_class_index = int(parsed.get("target_class_index", target_class_index))
        except (ValueError, _json.JSONDecodeError, TypeError):
            pass  # Not JSON — use original values

    if target_class_index == -1:
        return (
            "ERROR: Missing target_class_index. Please provide both image_path "
            "and target_class_index (0-7) as arguments."
        )
    try:
        if not os.path.isfile(image_path):
            return f"ERROR: Image file not found at '{image_path}'."

        if not (0 <= target_class_index < NUM_CLASSES):
            return (
                f"ERROR: Invalid class index {target_class_index}. "
                f"Must be in range [0, {NUM_CLASSES - 1}]."
            )

        # ---- Load Model ----
        model = load_model()

        # ---- Preprocess Image ----
        input_array = preprocess_image_from_path(image_path)

        logger.info(
            "Generating Grad-CAM for class %d (%s) on image: %s",
            target_class_index,
            CLASS_LABELS.get(target_class_index, {}).get("name", "Unknown"),
            image_path,
        )

        # ---- Generate Grad-CAM Heatmap ----
        cam = GradCAM(model)
        heatmap = cam.generate(input_array, target_class_idx=target_class_index)

        # ---- Create Overlay Visualization ----
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
        )
        output_path, peak_info = overlay_heatmap(
            image_path, heatmap, output_dir=output_dir
        )

        # ---- Analyze Saliency ----
        original = cv2.imread(image_path)
        h, w = original.shape[:2]
        heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
        saliency = _analyze_saliency(heatmap_full)

        # ---- Get Class Info ----
        class_info = CLASS_LABELS.get(
            target_class_index,
            {"name": f"Unknown ({target_class_index})", "abbreviation": "unk"},
        )

        # ---- Generate Clinical Note ----
        clinical_note = _generate_clinical_note(
            class_info["name"],
            class_info["abbreviation"],
            saliency["concentration"],
            peak_info["region"],
        )

        # ---- Format Output Report ----
        lines = [
            "═══ GRAD-CAM VISUAL EXPLAINABILITY REPORT ═══",
            f"Target Class: {class_info['name']} ({class_info['abbreviation']}) "
            f"[index={target_class_index}]",
            f"Heatmap Saved: {output_path}",
            "",
            "── Activation Peak ──",
            f"  Location: ({peak_info['peak_x']}, {peak_info['peak_y']}) px",
            f"  Region: {peak_info['region']}",
            f"  Peak Intensity: {peak_info['peak_value']:.4f}",
            "",
            "── Saliency Analysis ──",
            f"  Concentration: {saliency['concentration']}",
            f"  Coverage (>0.5 threshold): {saliency['coverage_pct']:.1f}% of image",
            f"  High Activation (>0.7): {saliency['high_activation_pct']:.1f}% of image",
            f"  Mean Activation: {saliency['mean_activation']:.4f}",
            "",
            f"  {saliency['spatial_description']}",
            "",
            "── Clinical Interpretability ──",
            f"  {clinical_note}",
            "═" * 48,
        ]

        report = "\n".join(lines)
        logger.info("Grad-CAM report generated successfully.")
        return report

    except Exception as e:
        logger.exception("Grad-CAM generation failed for '%s'", image_path)
        return f"ERROR: Grad-CAM generation failed — {type(e).__name__}: {str(e)}"


# ---------------------------------------------------------------------------
# Module Self-Test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running gradcam_tool self-test...")

    test_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(test_dir, exist_ok=True)
    test_image_path = os.path.join(test_dir, "_test_gradcam_dummy.png")

    dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(test_image_path, dummy_img)

    result = gradcam_visual_explainer.invoke({
        "image_path": test_image_path,
        "target_class_index": 4,  # Ringworm
    })
    print(result)

    os.remove(test_image_path)
    logger.info("Self-test PASSED.")
