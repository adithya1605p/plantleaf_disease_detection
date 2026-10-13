"""
Explainability Module: PyTorch Grad-CAM & Heatmap Visualizer.
Implements spatial activation gradient mapping and OpenCV colormap overlays.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """
    Generic PyTorch Grad-CAM implementation utilizing forward and backward hooks.
    Works seamlessly with convolutional backbones.
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Calculates normalized spatial gradient heatmap (0.0 to 1.0).
        input_tensor: shape (1, C, H, W)
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)
        
        # Handle outputs that might be tuple or YOLO tensor
        if isinstance(output, (tuple, list)):
            output = output[0]

        if class_idx is None:
            # Pick highest score across spatial/channel dimensions
            target = output.max()
        else:
            if output.dim() == 2:
                target = output[:, class_idx].sum()
            else:
                target = output.sum()

        # Backward pass for gradients
        target.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            # Fallback if layer produces no gradients
            h, w = input_tensor.shape[2], input_tensor.shape[3]
            return np.ones((h, w), dtype=np.float32) * 0.5

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Upsample to input resolution
        cam = F.interpolate(cam, size=(input_tensor.shape[2], input_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize 0 - 1
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def release(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

def overlay_heatmap(img_bgr: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlays 2D Grad-CAM mask on BGR image using OpenCV COLORMAP_JET.
    cam_mask: 2D array [0.0, 1.0] or [0, 255]
    img_bgr: uint8 BGR image
    """
    h, w = img_bgr.shape[:2]
    
    # Resize cam_mask to image dimensions if needed
    if cam_mask.shape[:2] != (h, w):
        cam_mask = cv2.resize(cam_mask, (w, h))

    # Scale to 0-255 uint8
    if cam_mask.dtype != np.uint8:
        cam_uint8 = np.uint8(255 * np.clip(cam_mask, 0.0, 1.0))
    else:
        cam_uint8 = cam_mask

    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(img_bgr, 1.0 - alpha, heatmap, alpha, 0)
    return blended

def create_synthetic_cam_from_boxes(img_shape, boxes, alpha=0.5):
    """
    High-speed edge explainability fallback:
    Creates spatial Gaussian attention heatmaps around detected pathological lesions.
    """
    h, w = img_shape[:2]
    heatmap_acc = np.zeros((h, w), dtype=np.float32)

    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        # Create 2D gaussian kernel for lesion center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sigma_x = bw / 2.0
        sigma_y = bh / 2.0

        y_coords, x_coords = np.ogrid[:h, :w]
        gaussian = np.exp(-(((x_coords - cx) ** 2) / (2 * (sigma_x ** 2) + 1e-5) +
                            ((y_coords - cy) ** 2) / (2 * (sigma_y ** 2) + 1e-5)))
        heatmap_acc = np.maximum(heatmap_acc, gaussian)

    h_min, h_max = np.min(heatmap_acc), np.max(heatmap_acc)
    if h_max > h_min:
        heatmap_acc = (heatmap_acc - h_min) / (h_max - h_min)
    
    return heatmap_acc
