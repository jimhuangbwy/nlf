import torch
import tensorrt as trt
import logging
import os
import cv2
import numpy as np
from tensorrt_bindings import int64

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TRTInference(torch.nn.Module):
    """
    Generic TensorRT inference wrapper for RT-DETR (multi-input/output).
    """

    def __init__(self, engine_path: str):
        super().__init__()
        self.engine_path = engine_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Buffer storage
        self.inputs = {}  # name -> tensor
        self.outputs = {}  # name -> tensor
        self.bindings = {}  # name -> ptr (int)

        self.stream = None
        self.context = None
        self.engine = None
        self.runtime = None

        self.load_engine()

    def load_engine(self):
        """Load TensorRT engine and identify I/O tensors."""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"Engine not found: {self.engine_path}")

        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(trt_logger)

        with open(self.engine_path, "rb") as f:
            engine_data = f.read()

        self.engine = self.runtime.deserialize_cuda_engine(engine_data)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # Inspect engine I/O
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = self.engine.get_tensor_dtype(name)
            shape = self.engine.get_tensor_shape(name)

            # Log structure
            io_type = "Input" if mode == trt.TensorIOMode.INPUT else "Output"
            logger.info(f"Found {io_type}: {name} | Shape: {shape} | Dtype: {dtype}")

    def _torch_dtype_from_trt(self, dtype):
        if dtype == trt.float32: return torch.float32
        if dtype == trt.float16: return torch.float16
        if dtype == trt.int32:   return torch.int32
        if dtype == trt.int64:   return torch.int64
        if dtype == trt.bool:    return torch.bool
        return torch.float32

    def _allocate_buffers(self, feed_dict):
        """
        Allocate buffers based on actual input shapes provided in feed_dict.
        feed_dict: {'images': tensor, 'orig_target_sizes': tensor}
        """
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)

            # 1. Handle Input Shapes
            if mode == trt.TensorIOMode.INPUT:
                if name not in feed_dict:
                    raise ValueError(f"Missing input for engine binding: {name}")

                input_tensor = feed_dict[name]
                self.context.set_input_shape(name, input_tensor.shape)

                # Reuse input tensor provided by user (zero-copy if already on GPU)
                if input_tensor.device == self.device:
                    self.inputs[name] = input_tensor
                else:
                    # If input is CPU, move to GPU
                    self.inputs[name] = input_tensor.to(self.device)

            # 2. Handle Output Shapes (Derived from Input)
            else:
                # Output shape might depend on input shape (dynamic)
                output_dims = self.context.get_tensor_shape(name)

                # Convert trt.Dims to tuple
                # Note: TensorRT 10 returns Dims object, cast to tuple
                shape = tuple(output_dims)

                # Check for dynamic dimensions (-1) that weren't resolved
                if -1 in shape:
                    raise RuntimeError(f"Output {name} has dynamic shape {shape} even after setting inputs.")

                dtype = self._torch_dtype_from_trt(self.engine.get_tensor_dtype(name))
                self.outputs[name] = torch.zeros(shape, dtype=dtype, device=self.device)

            # 3. Bind Addresses
            tensor = self.inputs[name] if mode == trt.TensorIOMode.INPUT else self.outputs[name]
            self.context.set_tensor_address(name, tensor.data_ptr())

    def forward(self, img_tensor: torch.Tensor, orig_sizes: torch.Tensor) -> dict:
        """
        Run inference.
        Args:
            img_tensor: (B, 3, H, W) normalized [0,1]
            orig_sizes: (B, 2)
        Returns:
            dict with keys 'labels', 'boxes', 'scores'
        """

        # Prepare feed dictionary
        feed_dict = {
            'images': img_tensor,
            'orig_target_sizes': orig_sizes
        }

        # Allocate or Re-allocate if shapes changed
        # Simple logic: Just allocate every time for safety with dynamic shapes,
        # or check if self.inputs matches current shapes.
        # For this script, we call it safely.
        self._allocate_buffers(feed_dict)

        # Execute Async V3
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)

        self.stream.synchronize()

        # Return a copy of outputs to avoid overwriting next frame
        return {name: tensor.clone() for name, tensor in self.outputs.items()}

# Global Helper Functions
def im_to_linear(im: torch.Tensor):
    if im.dtype == torch.uint8:
        return im.to(dtype=torch.float32).mul_(1.0 / 255.0).pow_(2.2)
    elif im.dtype == torch.uint16:
        return im.to(dtype=torch.float32).mul_(1.0 / 65504.0).nan_to_num_(posinf=1.0).pow_(2.2)
    elif im.dtype == torch.float16:
        return im ** 2.2
    else:
        return im.to(dtype=torch.float32).pow_(2.2)
# -------------------------------------------------------------------------
# Usage Example
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Init Model
    ENGINE_PATH = "RTv4-S-hgnet.engine"  # Ensure this matches your filename

    try:
        model = TRTInference(ENGINE_PATH)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        exit()

    # 2. Load and Preprocess Image
    image_path = "example_image.jpg"  # Replace with your image

    img = cv2.imread(image_path)

    h, w = img.shape[:2]

    # Preprocessing (Standard RT-DETR)
    # Resize to 640x640
    img = cv2.resize(img, (640, 640))

    # Normalize [0, 255] -> [0, 1]
    img = img.astype(np.float32) / 255.0

    # HWC -> CHW
    img = img.transpose(2, 0, 1)

    # Add Batch Dimension -> (1, 3, 640, 640)
    img_tensor = torch.from_numpy(img).unsqueeze(0).cuda()

    #img_tensor = im_to_linear(img_tensor)

    # Prepare Size Input -> (1, 2)
    # Note: Use int32 by default, but if you get errors, switch to int64
    sizes_tensor = torch.tensor([[h, w]], dtype=torch.int64).cuda()
    # sizes_tensor = torch.tensor([[h, w]], dtype=torch.int64).cuda() # Try this if int32 fails

    # 3. Inference
    logger.info("Running Inference...")
    results = model(img_tensor, sizes_tensor)

    print(results)

    # 4. Parse Results
    # RT-DETR v2/v4 usually output these names: 'labels', 'boxes', 'scores'
    # But TensorRT engine might rename them to 'output0', 'output1' etc if not named explicitly.
    # We print keys to be sure.
    logger.info(f"Output Keys: {results.keys()}")

    # Extract specific tensors (assuming standard names, adjust if yours differ)
    # The output shapes are typically:
    # scores: (B, 300) or (B, 300, 1)
    # boxes:  (B, 300, 4)
    # labels: (B, 300)

    scores = results.get('scores')
    boxes = results.get('boxes')
    labels = results.get('labels')

    if scores is not None:
        # Filter by confidence
        threshold = 0.3
        keep = scores > threshold

        # Squeeze batch dim if needed
        if keep.dim() > 1:
            keep = keep[0]
            final_boxes = boxes[0][keep]
            final_scores = scores[0][keep]
            final_labels = labels[0][keep]

        print(f"Found {len(final_boxes)} objects:")
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            print(f" - Class {int64(label)}: {score:.4f} | Box: {box.cpu().numpy()}")
    else:
        logger.warning("Could not identify output tensors automatically. Please check keys.")