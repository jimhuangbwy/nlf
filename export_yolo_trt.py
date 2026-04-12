import tensorrt as trt

def build_engine(onnx_path, engine_path, fp16=True):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            for i in range(parser.num_errors):
                print("Parser error:", parser.get_error(i))
            raise RuntimeError("Failed to parse ONNX model")
        else:
            print('open onnx')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    if fp16 and builder.platform_has_fast_fp16:
        print('gpu supports fp16')
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

    else:
        print('gpu does not support fp16')

    serialized_engine = builder.build_serialized_network(network, config)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    print(f"✅ Engine rebuilt successfully → {engine_path}")

# Example:
build_engine("yolo11n.onnx", "yolo11n.engine", fp16=False)