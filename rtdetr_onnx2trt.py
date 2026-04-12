import os
import argparse
import tensorrt as trt


def rtdetr_onnx2trt(onnx_path, engine_path, max_batchsize=1, opt_batchsize=1, min_batchsize=1, use_fp16=True, verbose=False) -> None:
    """ Convert ONNX model to TensorRT engine.
    Args:
        onnx_path (str): Path to the input ONNX model.
        engine_path (str): Path to save the output TensorRT engine.
        use_fp16 (bool): Whether to use FP16 precision.
        verbose (bool): Whether to enable verbose logging.
    """
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    [network.get_input(i) for i in range(network.num_inputs)]
    [network.get_output(i) for i in range(network.num_outputs)]

    parser = trt.OnnxParser(network, logger)

    if not os.path.isfile(onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"[INFO] Loading ONNX file from {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 optimization enabled.")
        else:
            print("[WARNING] FP16 not supported on this platform. Proceeding with FP32.")

    # profile = builder.create_optimization_profile()
    # profile.set_shape("images", min=(min_batchsize, 3, 640, 640), opt=(opt_batchsize, 3, 640, 640),
    #                   max=(max_batchsize, 3, 640, 640))
    # profile.set_shape("orig_target_sizes", min=(1, 2), opt=(1, 2), max=(1, 2))
    # config.add_optimization_profile(profile)

    print("[INFO] Building TensorRT engine...")
    build = builder.build_serialized_network

    if build is None:
        raise RuntimeError("Failed to build the engine.")

    print(f"[INFO] Saving engine to {engine_path}")
    with build(network, config) as engine, open(engine_path, "wb") as f:
        f.write(engine)
    print("[INFO] Engine export complete.")

rtdetr_onnx2trt('models/rtdetr.onnx', 'models/rtdetr.engine')