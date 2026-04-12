import threading
import time
from collections import deque
import cv2
import torch
from pythonosc import udp_client


# --- user helper functions (copy from your original file) ---

def im_to_linear(im: torch.Tensor):
    if im.dtype == torch.uint8:
        return im.to(dtype=torch.float16).mul_(1.0 / 255.0).pow_(2.2)
    elif im.dtype == torch.uint16:
        return im.to(dtype=torch.float16).mul_(1.0 / 65504.0).nan_to_num_(posinf=1.0).pow_(2.2)
    elif im.dtype == torch.float16:
        return im ** 2.2
    else:
        return im.to(dtype=torch.float16).pow_(2.2)


def send_joint(client, address, index, enable, position, rotation, serial):
    pos_x, pos_y, pos_z = position
    rot_x, rot_y, rot_z, rot_w = rotation
    args = [index, enable, float(timeoffset), float(pos_x), float(pos_y), float(pos_z), float(rot_x), float(rot_y),
            float(rot_z), float(rot_w), serial]
    client.send_message(address, args)


# -------------------- Threaded pipeline --------------------
# Capture thread: reads camera and writes latest frame (non-blocking)
# Inference thread: reads latest frame, runs model, writes latest result
# Display thread: reads latest frame and latest result, renders and shows


def threaded_pipeline(
        model,
        client,
        device,
        input_cam=0,
        input_size=640,
        max_queue_len=1,
):
    stop_event = threading.Event()

    # shared single-slot buffers (always keep latest)
    frame_slot = deque(maxlen=max_queue_len)  # holds raw BGR numpy image
    tensor_slot = deque(maxlen=max_queue_len)  # holds preprocessed torch tensor
    result_slot = deque(maxlen=max_queue_len)  # holds model results: dict or tuple

    # synchronization
    frame_lock = threading.Lock()
    result_lock = threading.Lock()

    # extrapolation state
    last_joints = None
    current_joints = None
    last_infer_time = 0.0
    infer_interval = 0.0
    extrapolation_lock = threading.Lock()

    cap = cv2.VideoCapture(input_cam)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    # reduce internal OS buffer to prefer fresher frames
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    def capture_loop():
        # runs in background, pushes latest frame into frame_slot and tensor_slot
        nonlocal stop_event
        while not stop_event.is_set():
            ret, image = cap.read()
            if not ret:
                # brief sleep to avoid tight spinning
                time.sleep(0.005)
                continue

            image = cv2.flip(image, 1)

            # push latest BGR image
            with frame_lock:
                #frame_slot.clear()
                frame_slot.append(image)

            # prepare tensor (non-blocking) and push
            try:
                tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                tensor = im_to_linear(tensor)
            except Exception:
                tensor = None

            if tensor is not None:
                with frame_lock:
                    #tensor_slot.clear()
                    tensor_slot.append(tensor)

            # small sleep to yield CPU and avoid starving camera/driver
            time.sleep(0.001)

    def inference_loop():
        nonlocal last_joints, current_joints, last_infer_time, infer_interval
        while not stop_event.is_set():
            # pick the latest tensor, if any
            with frame_lock:
                if len(tensor_slot) == 0:
                    time.sleep(0.001)
                    continue
                tensor = tensor_slot.pop()

            # move to device once (avoid blocking capture thread)
            try:
                tensor = tensor.to(device)
            except Exception:
                tensor = tensor.cuda() if device.type == 'cuda' else tensor

            start = time.perf_counter()
            with torch.inference_mode(), torch.no_grad():
                results = model(tensor)
                # try:
                #     results = model(tensor)
                # except Exception as e:
                #     # model failure: skip and continue
                #     print("Model inference error:", e)
                #     time.sleep(0.001)
                #     continue
            end = time.perf_counter()

            # update inference timing
            now = time.perf_counter()
            interval = now - last_infer_time if last_infer_time != 0 else (end - start)
            last_infer_time = now
            infer_interval = interval

            # update extrapolation state and result slot
            # results expected: results[0]=joints_list, results[1]=boxes ... (as in your original)
            # joints = None
            # boxes = None
            try:
                joints = results[0]
                boxes = results[1]
            except Exception:
                # fallback: store raw results
                joints = results
                boxes = None

            with extrapolation_lock:
                if isinstance(joints, torch.Tensor) and len(joints) > 0:
                    # adapt to your model output shape
                    last_joints = None if current_joints is None else current_joints
                    current_joints = joints[0].cpu() #joints[0].clone().detach().cpu()
                    current_joints = current_joints-current_joints[0]
                else:
                    # no detection → reset
                    last_joints = None
                    current_joints = None
                    infer_interval = 0.0
                    last_infer_time = time.perf_counter()

            # write result into result_slot
            with result_lock:
                #result_slot.clear()
                result_slot.append((current_joints, boxes))

            # small sleep to allow display/capture to run smoothly
            time.sleep(0.001)

    def display_loop():
        nonlocal last_joints, current_joints, last_infer_time, infer_interval
        fps = 0
        fps_times = deque(maxlen=30)
        last_time = time.perf_counter()

        while not stop_event.is_set():
            # get latest image
            with frame_lock:
                if len(frame_slot) == 0:
                    time.sleep(0.001)
                    continue
                image = frame_slot[-1].copy()

            # get latest result
            with result_lock:
                res = result_slot[-1] if len(result_slot) > 0 else (None, None)

            joints_local, boxes_local = res

            # extrapolate joints if available
            if joints_local is not None:
                client.send_message("/VMT/SetRoomMatrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
                send_joint(client, "/VMT/Follow/Unity", index=24, enable=ENABLE_TRACKING_REFERENCE,
                           position=(1.5, -0.7, 0), rotation=ZEROQ, serial="HMD")  # 上半身的參考點
                with extrapolation_lock:
                    if last_joints is not None and current_joints is not None and infer_interval > 1e-6:
                        dt = time.perf_counter() - last_infer_time
                        alpha = max(0.0, min(dt / infer_interval, 1.5))
                        vel = current_joints - last_joints
                        extrapolated = current_joints + vel * alpha
                    else:
                        extrapolated = current_joints

                if extrapolated is not None:
                    # send OSC for all joints
                    for j, joint in enumerate(extrapolated):
                        send_joint(client, "/VMT/Follow/Unity", index=j, enable=ENABLE_TRACKER, position=-joint,
                                   rotation=ZEROQ, serial="VMT_24")

            # Draw bounding box if provided (boxes may be TRT format)
            if boxes_local is not None:
                try:
                    b = boxes_local[0][0]
                    if b is not None and len(b) >= 4:
                        x, y, w, h = b[:4]
                        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                except Exception:
                    pass

            # draw FPS
            now = time.perf_counter()
            fps_times.append(now - last_time)
            last_time = now
            if len(fps_times) > 29:
                avg = sum(fps_times) / len(fps_times)
                fps = 1.0 / avg if avg > 1e-6 else 0.0
                fps_times.clear()

            cv2.putText(image, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow("Webcam - Press Q to quit", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break

            # small sleep to avoid maxing CPU
            time.sleep(0.001)

        cap.release()
        cv2.destroyAllWindows()

    # --- start threads ---
    t_capture = threading.Thread(target=capture_loop, daemon=True)
    t_infer = threading.Thread(target=inference_loop, daemon=True)
    t_display = threading.Thread(target=display_loop, daemon=True)

    t_capture.start()
    t_infer.start()
    t_display.start()

    try:
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        stop_event.set()

    # t_capture.join()
    # t_infer.join()
    # t_display.join()


# -------------------- Example usage --------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load models (adapt to your environment) ---
    # Keep your original model loading code here; use the same model object as before
    from nlf.pt.multiperson import person_detector_trt
    from onnx_helper import TRTInference
    from nlf.pt.models import nlf_model_trt

    detector = person_detector_trt.PersonDetector('models/yolo12n.engine').to(device).eval().half()
    backbone = TRTInference('models/backbone.engine')
    weight_field = TRTInference('models/weight_field.engine')
    layer = TRTInference('models/layer.engine')
    model_pytorch = nlf_model_trt.NLFModel(backbone, weight_field, layer).to(device).eval().half()
    model = None
    try:
        # multiperson model wrapper
        from nlf.pt.multiperson import multiperson_model_trt

        model = multiperson_model_trt.MultipersonNLF(model_pytorch, detector, device=device).to(device).eval().half()
    except Exception as e:
        print("Failed to build multiperson model:", e)
        raise

    ip = '127.0.0.1'
    port = 39570
    client = udp_client.SimpleUDPClient(ip, port)

    # Global constants used in send_joint and display
    global ZEROQ, ENABLE_TRACKER, ENABLE_TRACKING_REFERENCE, timeoffset
    ZEROQ = (0, 0, 0, 1)
    ENABLE_TRACKER = 1
    ENABLE_TRACKING_REFERENCE = 4
    timeoffset = 0.0

    threaded_pipeline(model, client, device)
