import torch
import tensorrt as trt
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

class TRTInference(torch.nn.Module):
    """
    Minimal YOLO11n inference wrapper using TensorRT engine.
    Only imports: torch, tensorrt
    Fixed: Dims → tuple conversion
    """
    def __init__(self, engine_path: str):
        super().__init__()
        self.engine_path = engine_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_tensor = None
        self.output_tensor = None
        self.stream = None
        self.context = None
        self.engine = None
        self.runtime = None
        self.input_name = None
        self.output_name = None

        self.load_engine()

    def load_engine(self):
        """Load TensorRT engine from file."""
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
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.stream = torch.cuda.Stream()

        logger.info(f"Engine loaded. Input: {self.input_name}, Output: {self.output_name}")

    def _allocate_buffers(self, input_shape):
        """Allocate input/output tensors on GPU using tuple shapes."""
        device = self.device

        # Input: (1, 3, 640, 640)
        self.input_tensor = torch.zeros(input_shape, dtype=torch.float32, device=device)

        # Output: Get shape as tuple
        output_dims = self.context.get_tensor_shape(self.output_name)
        if -1 in output_dims:
            self.context.set_input_shape(self.input_name, input_shape)
            output_dims = self.context.get_tensor_shape(self.output_name)

        # Convert Dims → tuple
        output_shape = tuple(output_dims)  # Critical fix

        self.output_tensor = torch.zeros(output_shape, dtype=torch.float32, device=device)

        # Bind addresses
        self.context.set_tensor_address(self.input_name, self.input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

    def forward(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        Run inference on preprocessed image tensor.
        Expected input: torch.Tensor of shape (1, 3, 640, 640), values in [0, 1]
        """
        # expected_shape = (1, 3, 640, 640)
        # if img_tensor.shape != expected_shape:
        #     raise ValueError(f"Expected input shape {expected_shape}, got {img_tensor.shape}")

        # if img_tensor.device != self.device:
        #     img_tensor = img_tensor.to(self.device)

        # # Normalize to [0, 1]
        # if img_tensor.max() > 1.0:
        #     img_tensor = img_tensor / 255.0

        # Validate input
        # if torch.isnan(img_tensor).any() or torch.isinf(img_tensor).any():
        #     raise ValueError("Input contains NaN or Inf")

        # Allocate buffers if needed
        if self.input_tensor is None or self.input_tensor.shape != img_tensor.shape:
            self._allocate_buffers(img_tensor.shape)

        # Copy input
        self.input_tensor.copy_(img_tensor)

        # Set input shape (for dynamic models)
        self.context.set_input_shape(self.input_name, img_tensor.shape)

        # Rebind addresses
        self.context.set_tensor_address(self.input_name, self.input_tensor.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_tensor.data_ptr())

        # Execute
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()

        # Clone output
        #output = self.output_tensor.clone()

        # Clamp NaN/Inf
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     logger.warning("NaN/Inf in output, clamping...")
        #     output = torch.clamp(output, min=-1e9, max=1e9)

        #logger.info(f"Inference complete. Output shape: {output.shape}")
        return self.output_tensor

    # def cleanup(self):
    #     """Free resources."""
    #     self.input_tensor = None
    #     self.output_tensor = None
    #     self.context = None
    #     self.engine = None
    #     self.runtime = None
    #     if self.device.type == "cuda":
    #         torch.cuda.empty_cache()
    #     logger.info("Cleanup complete")