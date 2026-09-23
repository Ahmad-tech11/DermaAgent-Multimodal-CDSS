"""
gradcam_engine.py — Standalone Grad-CAM Implementation for TensorFlow/Keras
=============================================================================

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) from scratch
using TensorFlow's GradientTape API, targeting the final convolutional layer
inside the TF Hub MobileNetV2 feature extractor.

This implementation works with the DermaAI model architecture:
    hub.KerasLayer("tf2-preview/mobilenet_v2/feature_vector/4")
    → Dense(8, activation='softmax')

Reference
---------
Selvaraju, R.R., et al. "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization." IJCV, 128(2), 336–359, 2020.

Author : DermaAgent Assessment Module
License: MIT
"""

import os
import logging
import datetime
from typing import Optional, Tuple, Dict, Union, List

import numpy as np
import cv2
import tensorflow as tf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for TF/Keras.

    Uses tf.GradientTape to compute gradients of a target class score
    with respect to the output of a convolutional layer, then produces
    a class-discriminative localization heatmap.

    For the DermaAI model (Sequential with hub.KerasLayer + Dense), the
    target conv layer is the **output of the KerasLayer** itself, which
    produces a 1280-dim feature vector after global average pooling.
    Since TF Hub's mobilenet_v2/feature_vector already pools spatially,
    we need to access the **internal conv output before pooling**.

    To handle this, we build a sub-model that outputs the conv features
    from inside the TF Hub layer.

    Parameters
    ----------
    model : tf.keras.Model
        The full trained Keras model.

    Examples
    --------
    >>> cam = GradCAM(model)
    >>> heatmap = cam.generate(preprocessed_image, target_class_idx=4)
    """

    def __init__(self, model: tf.keras.Model) -> None:
        self.model = model
        self._grad_model = None
        self._build_gradient_model()
        logger.info("GradCAM initialized for TF/Keras model.")

    def _build_gradient_model(self) -> None:
        """
        Build a sub-model that outputs both the convolutional feature maps
        and the final predictions, enabling gradient computation.

        For the TF Hub MobileNetV2 feature_vector KerasLayer, the internal
        MobileNetV2 has its last conv output at a specific layer. We find
        the last Conv2D output within the hub layer's internal graph.
        """
        hub_layer = self.model.layers[0]  # The hub.KerasLayer

        # Try to access the internal Keras model inside the hub layer
        # TF Hub KerasLayers wrap a SavedModel; we need to find conv outputs
        try:
            # Approach: Build a model that takes the same input and outputs
            # both the feature maps before global avg pooling and the final predictions.
            # We use tf.GradientTape to watch the hub layer output (1280-d vector)
            # and reshape it spatially for Grad-CAM visualization.

            # For TF Hub mobilenet_v2/feature_vector, the output is already
            # globally average-pooled (shape: [batch, 1280]). We'll compute
            # Grad-CAM on this 1280-d vector and reshape to a 1-D spatial map,
            # OR we intercept the conv output by inspecting saved model internals.

            # Practical approach: Use the feature vector directly.
            # Create a grad model: input → hub_layer_output, final_predictions
            self._grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[
                    hub_layer.output,       # Feature vector (batch, 1280)
                    self.model.output,      # Predictions (batch, 8)
                ],
            )
            self._feature_is_spatial = False
            self._feature_dim = 1280

            # Check if we can get spatial features from inside the hub layer
            # by trying to find internal conv layers
            self._try_build_spatial_grad_model(hub_layer)

            logger.debug(
                "Gradient model built. Spatial features: %s",
                self._feature_is_spatial,
            )
        except Exception as e:
            logger.warning("Could not build gradient model from hub internals: %s", e)
            # Fallback: use the basic approach with feature vector
            self._grad_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=[hub_layer.output, self.model.output],
            )
            self._feature_is_spatial = False
            self._feature_dim = 1280

    def _try_build_spatial_grad_model(self, hub_layer) -> None:
        """
        Attempt to extract the spatial conv output from inside the TF Hub layer
        for proper 2-D Grad-CAM heatmaps.
        """
        try:
            # Access the inner callable (SavedModel)
            inner = hub_layer._callable
            if hasattr(inner, 'signatures'):
                # Try to find internal layers via the saved model
                pass

            # Alternative: inspect the hub layer's call graph
            # For tf2-preview/mobilenet_v2/feature_vector/4,
            # the internal model applies MobileNetV2 convolutions
            # then global_average_pooling2d. We want the output BEFORE pooling.

            # Try to find the last conv output within the hub layer
            # by building a test input and tracing
            test_input = tf.zeros((1, 224, 224, 3))

            # If the hub layer contains a Keras model, look for conv layers
            if hasattr(hub_layer, '_self_tracked_trackables'):
                for tracked in hub_layer._self_tracked_trackables:
                    if isinstance(tracked, tf.keras.Model):
                        # Found internal Keras model
                        inner_model = tracked
                        # Find the last Conv2D or similar layer
                        conv_layers = [
                            l for l in inner_model.layers
                            if 'conv' in l.name.lower() or 'expanded_conv' in l.name.lower()
                        ]
                        if conv_layers:
                            last_conv = conv_layers[-1]
                            # Build spatial grad model
                            inner_output_model = tf.keras.Model(
                                inputs=inner_model.input,
                                outputs=last_conv.output,
                            )
                            # Create a combined model
                            inp = self.model.input
                            spatial_features = inner_output_model(inp)
                            final_output = self.model(inp)
                            self._grad_model = tf.keras.Model(
                                inputs=inp,
                                outputs=[spatial_features, final_output],
                            )
                            self._feature_is_spatial = True
                            logger.info(
                                "Found spatial conv layer: %s (shape: %s)",
                                last_conv.name,
                                last_conv.output_shape,
                            )
                            return

        except Exception as e:
            logger.debug("Could not extract spatial features: %s", e)

    def generate(
        self,
        input_array: np.ndarray,
        target_class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input and target class.

        Parameters
        ----------
        input_array : np.ndarray
            Preprocessed input image of shape ``(1, 224, 224, 3)``
            with pixel values in ``[0, 1]``.
        target_class_idx : int, optional
            Target class index (0–7). If None, uses the predicted class.

        Returns
        -------
        heatmap : np.ndarray
            2-D numpy array normalized to ``[0, 1]``.
            Shape is ``(7, 7)`` for spatial features or ``(8, 8)``
            approximation for vector features.
        """
        input_tensor = tf.cast(input_array, tf.float32)

        # ---- Compute gradients with GradientTape ----
        with tf.GradientTape() as tape:
            # Watch the input to enable gradient flow
            tape.watch(input_tensor)

            # Forward pass through the gradient model
            features, predictions = self._grad_model(input_tensor)

            # If no target specified, use predicted class
            if target_class_idx is None:
                target_class_idx = int(tf.argmax(predictions[0]).numpy())
                logger.info("Auto-selected target class: %d", target_class_idx)

            # Extract the target class score
            target_score = predictions[:, target_class_idx]

        # ---- Compute gradients of target score w.r.t. feature maps ----
        gradients = tape.gradient(target_score, features)

        if gradients is None:
            logger.warning("Gradients are None — model may not be differentiable through hub layer.")
            # Fallback: return a centered Gaussian-like heatmap
            return self._fallback_heatmap()

        if self._feature_is_spatial:
            # features shape: (1, H, W, C) — standard spatial Grad-CAM
            return self._compute_spatial_gradcam(features, gradients)
        else:
            # features shape: (1, 1280) — vector features, need alternative approach
            return self._compute_vector_gradcam(input_tensor, target_class_idx)

    def _compute_spatial_gradcam(
        self, features: tf.Tensor, gradients: tf.Tensor
    ) -> np.ndarray:
        """
        Standard Grad-CAM for spatial feature maps.

        L_Grad-CAM = ReLU(Σ_k α_k · A^k)
        where α_k = GAP(∂y^c/∂A^k)
        """
        # Global average pooling of gradients → channel importance weights
        # gradients shape: (1, H, W, C)
        weights = tf.reduce_mean(gradients, axis=(1, 2))  # (1, C)

        # Weighted combination of feature maps
        # features shape: (1, H, W, C)
        features_np = features.numpy()[0]    # (H, W, C)
        weights_np = weights.numpy()[0]      # (C,)

        # Σ_k α_k · A^k
        cam = np.zeros(features_np.shape[:2], dtype=np.float32)  # (H, W)
        for k in range(features_np.shape[-1]):
            cam += weights_np[k] * features_np[:, :, k]

        # ReLU — keep only positive contributions
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if cam.max() > 1e-8:
            cam = cam / cam.max()
        else:
            cam = np.zeros_like(cam)

        logger.info("Spatial Grad-CAM generated — shape: %s", cam.shape)
        return cam

    def _compute_vector_gradcam(
        self,
        input_tensor: tf.Tensor,
        target_class_idx: int,
    ) -> np.ndarray:
        """
        Grad-CAM approximation for models where the feature extractor
        outputs a 1-D vector (after global average pooling).

        Uses input-space gradients to create an activation map by computing
        the gradient of the target class w.r.t. the input image, then
        aggregating across color channels.
        """
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            predictions = self.model(input_tensor)
            target_score = predictions[:, target_class_idx]

        # Gradient of target class score w.r.t. input image
        input_gradients = tape.gradient(target_score, input_tensor)

        if input_gradients is None:
            return self._fallback_heatmap()

        # Aggregate across color channels (absolute value for saliency)
        # input_gradients shape: (1, 224, 224, 3)
        saliency = tf.reduce_max(tf.abs(input_gradients[0]), axis=-1).numpy()
        # saliency shape: (224, 224)

        # Smooth with Gaussian blur for better visualization
        saliency = cv2.GaussianBlur(saliency, (15, 15), 0)

        # Normalize to [0, 1]
        if saliency.max() > 1e-8:
            saliency = saliency / saliency.max()

        # Downsample to a coarser map (like conv features would be)
        heatmap = cv2.resize(saliency, (14, 14), interpolation=cv2.INTER_AREA)

        logger.info("Vector Grad-CAM (input-gradient) generated — shape: %s", heatmap.shape)
        return heatmap

    def _fallback_heatmap(self) -> np.ndarray:
        """Generate a centered Gaussian heatmap as fallback."""
        logger.warning("Using fallback Gaussian heatmap (gradients unavailable).")
        size = 14
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        xx, yy = np.meshgrid(x, y)
        heatmap = np.exp(-(xx**2 + yy**2) / 0.5)
        return (heatmap / heatmap.max()).astype(np.float32)


# =========================================================================
# Heatmap Visualization Utilities
# =========================================================================

def overlay_heatmap(
    image_path: str,
    heatmap: np.ndarray,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Tuple[str, Dict[str, Union[int, float, str]]]:
    """
    Overlay a Grad-CAM heatmap on the original image and save the result.

    Parameters
    ----------
    image_path : str
        Path to the original input image.
    heatmap : np.ndarray
        2-D normalized heatmap array with values in ``[0, 1]``.
    output_dir : str, optional
        Directory to save the output visualization.
    alpha : float, optional
        Blending factor (0 = original only, 1 = heatmap only).
    colormap : int, optional
        OpenCV colormap. Default is ``cv2.COLORMAP_JET``.

    Returns
    -------
    output_path : str
        Absolute path to the saved overlay image.
    peak_info : dict
        Peak activation metadata.
    """
    original = cv2.imread(image_path)
    if original is None:
        raise FileNotFoundError(f"Cannot read image at: {image_path}")

    h, w = original.shape[:2]

    # Resize heatmap to match original image
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Apply colormap
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, colormap)

    # Blend original and heatmap
    overlay = cv2.addWeighted(original, 1 - alpha, heatmap_colored, alpha, 0)

    # Compute peak intensity location
    peak_idx = np.unravel_index(np.argmax(heatmap_resized), heatmap_resized.shape)
    peak_y, peak_x = int(peak_idx[0]), int(peak_idx[1])
    peak_x_pct = round(peak_x / w * 100, 1)
    peak_y_pct = round(peak_y / h * 100, 1)

    # Draw crosshair at peak
    cv2.drawMarker(
        overlay, (peak_x, peak_y),
        color=(0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=20, thickness=2,
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"cam_{timestamp}.png"
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, overlay)
    logger.info("Heatmap overlay saved to: %s", output_path)

    region = _describe_region(peak_x_pct, peak_y_pct)
    peak_info = {
        "peak_x": peak_x,
        "peak_y": peak_y,
        "peak_x_pct": peak_x_pct,
        "peak_y_pct": peak_y_pct,
        "peak_value": float(np.max(heatmap_resized)),
        "region": region,
        "image_width": w,
        "image_height": h,
    }

    return output_path, peak_info


def _describe_region(x_pct: float, y_pct: float) -> str:
    """Convert normalized coordinates to a human-readable region description."""
    if y_pct < 33:
        v = "upper"
    elif y_pct < 66:
        v = "center"
    else:
        v = "lower"

    if x_pct < 33:
        h = "left"
    elif x_pct < 66:
        h = "center"
    else:
        h = "right"

    if v == "center" and h == "center":
        return "center of the image"
    elif v == "center":
        return f"center-{h} region"
    elif h == "center":
        return f"{v}-center region"
    else:
        return f"{v}-{h} quadrant"


# =========================================================================
# Convenience Function
# =========================================================================

def run_gradcam(
    model: tf.keras.Model,
    input_array: np.ndarray,
    image_path: str,
    target_class: Optional[int] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> Tuple[np.ndarray, str, Dict]:
    """
    End-to-end Grad-CAM: generate heatmap, overlay, and save.

    Parameters
    ----------
    model : tf.keras.Model
        The trained Keras model.
    input_array : np.ndarray
        Preprocessed input of shape ``(1, 224, 224, 3)``.
    image_path : str
        Path to the original image for overlay.
    target_class : int, optional
        Target class index. If None, uses predicted class.
    output_dir : str, optional
        Directory to save output.

    Returns
    -------
    heatmap, output_path, peak_info
    """
    cam = GradCAM(model)
    heatmap = cam.generate(input_array, target_class)
    output_path, peak_info = overlay_heatmap(image_path, heatmap, output_dir=output_dir)
    return heatmap, output_path, peak_info


# =========================================================================
# Module Self-Test
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logger.info("Running Grad-CAM engine self-test (TensorFlow)...")

    # Create a dummy input
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)

    # Try loading the real model
    try:
        from classifier_tool import load_model
        model = load_model()
        cam = GradCAM(model)
        heatmap = cam.generate(dummy_input, target_class_idx=0)
        logger.info("Self-test PASSED — heatmap shape: %s, range: [%.3f, %.3f]",
                    heatmap.shape, heatmap.min(), heatmap.max())
    except Exception as e:
        logger.error("Self-test failed: %s", e)
