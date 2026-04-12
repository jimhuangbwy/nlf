import tensorrt as trt

def build_engine(onnx_path, engine_path, fp16=False):
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # Parse ONNX (TensorRT will also read .onnx.data automatically)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")

    # Create builder config
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8 GB
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # Create an optimization profile
    profile = builder.create_optimization_profile()

    # Get input name
    input_tensor = network.get_input(0)
    input_name = input_tensor.name

    # Define the min/opt/max shapes (all must be positive)
    # You can set a small tolerance around 284x284 if you want flexible input sizes
    profile.set_shape(
        input_name,
        min=(1, 3, 384, 384),   # smallest allowed input
        opt=(1, 3, 384, 384),   # optimal input (most common)
        max=(1, 3, 384, 384)    # largest allowed input
    )

    # Add the profile to the config
    config.add_optimization_profile(profile)

    # Build serialized network (TensorRT 9+)
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine (check shape/profile).")

    # Save to .engine file
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ TensorRT engine saved at: {engine_path}")

# Example usage
build_engine("backbone.onnx", "backbone.engine", fp16=True)