import threading
import time
from collections import deque
import cv2
import torch
from pythonosc import udp_client

# --- Import your custom modules ---
# (We wrap these in try-except to prevent UI crashes if models are missing during development)
try:
    from nlf.pt.multiperson import person_detector_trt
    from onnx_helper import TRTInference
    from nlf.pt.models import nlf_model_trt
    from nlf.pt.multiperson import multiperson_model_trt

    MODELS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Tracking modules not found. {e}")
    MODELS_AVAILABLE = False


# Global Helper Functions
def im_to_linear(im: torch.Tensor):
    if im.dtype == torch.uint8:
        return im.to(dtype=torch.float16).mul_(1.0 / 255.0).pow_(2.2)
    elif im.dtype == torch.uint16:
        return im.to(dtype=torch.float16).mul_(1.0 / 65504.0).nan_to_num_(posinf=1.0).pow_(2.2)
    elif im.dtype == torch.float16:
        return im ** 2.2
    else:
        return im.to(dtype=torch.float16).pow_(2.2)


def send_joint(client, address, index, enable, position, rotation, serial, timeoffset=0.0):
    pos_x, pos_y, pos_z = position
    rot_x, rot_y, rot_z, rot_w = rotation
    args = [index, enable, float(timeoffset), float(pos_x), float(pos_y), float(pos_z), float(rot_x), float(rot_y),
            float(rot_z), float(rot_w), serial]
    client.send_message(address, args)


class PinoTracker:
    def __init__(self):
        self.stop_event = threading.Event()
        self.threads = []
        self.client = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # OSC Constants
        self.ZEROQ = (0, 0, 0, 1)
        self.ENABLE_TRACKER = 1
        self.ENABLE_TRACKING_REFERENCE = 4
        self.timeoffset = 0.0

    def load_models(self):
        """Loads the TensorRT engines. This is heavy, so we call it explicitly."""
        if not MODELS_AVAILABLE:
            print("Models cannot be loaded due to missing imports.")
            return

        print("Loading TensorRT Models...")
        detector = person_detector_trt.PersonDetector('models/yolo12s.engine').to(self.device).eval().half()
        backbone = TRTInference('models/backbone.engine')
        weight_field = TRTInference('models/weight_field.engine')
        layer = TRTInference('models/layer.engine')
        model_pytorch = nlf_model_trt.NLFModel(backbone, weight_field, layer).to(self.device).eval().half()

        self.model = multiperson_model_trt.MultipersonNLF(model_pytorch, detector, device=self.device).to(
            self.device).eval().half()
        print("Models Loaded.")

        # Setup OSC
        ip = '127.0.0.1'
        port = 39570
        self.client = udp_client.SimpleUDPClient(ip, port)

    def start(self, input_cam=0):
        if not self.model:
            self.load_models()

        self.stop_event.clear()

        # Start the pipeline threads
        self.threaded_pipeline(input_cam=input_cam)

    def stop(self):
        print("Stopping Tracker...")
        self.stop_event.set()
        # Wait for threads to finish
        for t in self.threads:
            t.join(timeout=1.0)
        self.threads = []
        cv2.destroyAllWindows()
        print("Tracker Stopped.")

    def threaded_pipeline(self, input_cam=0, max_queue_len=2):
        # Shared slots
        frame_slot = deque(maxlen=max_queue_len)
        tensor_slot = deque(maxlen=max_queue_len)
        result_slot = deque(maxlen=max_queue_len)

        frame_lock = threading.Lock()
        result_lock = threading.Lock()
        extrapolation_lock = threading.Lock()

        # Internal State for Inference Loop
        state = {
            "last_joints": None,
            "current_joints": None,
            "last_infer_time": 0.0,
            "infer_interval": 0.0
        }

        cap = cv2.VideoCapture(input_cam)
        if not cap.isOpened():
            print("Cannot open camera")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

        self.frame_counter = 0

        def capture_loop():
            while not self.stop_event.is_set():
                ret, image = cap.read()
                if not ret:
                    time.sleep(0.005)
                    continue

                image = cv2.flip(image, 1)

                # --- UPDATE 2: Attach ID to the frame ---
                self.frame_counter += 1
                with frame_lock:
                    # We store a tuple: (image, frame_id)
                    frame_slot.append((image, self.frame_counter))

                # ... (Tensor logic remains the same) ...
                try:
                    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
                    tensor = im_to_linear(tensor)
                    if tensor is not None:
                        with frame_lock:
                            tensor_slot.append(tensor)
                except Exception:
                    pass
                time.sleep(0.001)


        def inference_loop():
            while not self.stop_event.is_set():
                with frame_lock:
                    if len(tensor_slot) == 0:
                        time.sleep(0.001)
                        continue
                    tensor = tensor_slot.pop()

                try:
                    tensor = tensor.to(self.device)
                except:
                    pass  # handle cpu fallback if needed

                start = time.perf_counter()
                with torch.inference_mode(), torch.no_grad():
                    results = self.model(tensor)
                end = time.perf_counter()

                now = time.perf_counter()
                interval = now - state["last_infer_time"] if state["last_infer_time"] != 0 else (end - start)
                state["last_infer_time"] = now
                state["infer_interval"] = interval

                try:
                    joints = results[0]
                    boxes = results[1]
                except:
                    joints = results
                    boxes = None

                with extrapolation_lock:
                    if isinstance(joints, torch.Tensor) and len(joints) > 0:
                        state["last_joints"] = None if state["current_joints"] is None else state["current_joints"]
                        state["current_joints"] = joints[0].cpu()
                        state["current_joints"] = state["current_joints"] - state["current_joints"][0]
                    else:
                        state["last_joints"] = None
                        state["current_joints"] = None
                        state["infer_interval"] = 0.0
                        state["last_infer_time"] = time.perf_counter()

                with result_lock:
                    result_slot.append((state["current_joints"], boxes))
                time.sleep(0.001)

        def display_loop():
            fps_times = deque(maxlen=15)
            fps = 0
            last_time = time.perf_counter()
            last_rendered_id = -1

            while not self.stop_event.is_set():
                with result_lock:
                    res = result_slot[-1] if len(result_slot) > 0 else (None, None)

                joints_local, boxes_local = res

                # --- OSC LOGIC ---
                if joints_local is not None:
                    self.client.send_message("/VMT/SetRoomMatrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
                    send_joint(self.client, "/VMT/Follow/Unity", index=24, enable=self.ENABLE_TRACKING_REFERENCE,
                               position=(1.5, -0.7, 0), rotation=self.ZEROQ, serial="HMD")

                    with extrapolation_lock:
                        if (state["last_joints"] is not None and state["current_joints"] is not None
                                and state["infer_interval"] > 1e-6):
                            dt = time.perf_counter() - state["last_infer_time"]
                            alpha = max(0.0, min(dt / state["infer_interval"], 1.5))
                            vel = state["current_joints"] - state["last_joints"]
                            extrapolated = state["current_joints"] + vel * alpha
                        else:
                            extrapolated = state["current_joints"]

                    if extrapolated is not None:
                        for j, joint in enumerate(extrapolated):
                            send_joint(self.client, "/VMT/Follow/Unity", index=j, enable=self.ENABLE_TRACKER,
                                       position=-joint, rotation=self.ZEROQ, serial="VMT_24")

                    now = time.perf_counter()
                    fps_times.append(now - last_time)
                    last_time = now
                    if len(fps_times) > 14:
                        avg = sum(fps_times) / len(fps_times)
                        fps = 1.0 / avg if avg > 1e-6 else 0.0
                        fps_times.clear()

                with frame_lock:
                    if len(frame_slot) == 0:
                        time.sleep(0.001)
                        continue

                    # --- UPDATE 4: Unpack tuple ---
                    # Only grab the frame if it is NEW
                    img_data = frame_slot[-1]  # This is now (image, id)

                    # If dealing with raw image (startup), handle gracefully
                    if isinstance(img_data, tuple):
                        image, current_id = img_data
                    else:
                        # Fallback for old data in buffer
                        image = img_data
                        current_id = -1

                # --- UPDATE 5: Sync Logic ---
                # If we have already drawn this frame ID, skip and sleep
                if current_id == last_rendered_id:
                    time.sleep(0.001)
                    continue

                last_rendered_id = current_id
                # --- DRAWING ---
                if boxes_local is not None:
                    try:
                        b = boxes_local[0][0]
                        if b is not None and len(b) >= 4:
                            x, y, w, h = b[:4]
                            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    except:
                        pass

                cv2.putText(image, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.imshow("PinoFBT View - Press Q or 'Link' to stop", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop_event.set()
                    break
                time.sleep(0.001)

            cap.release()
            cv2.destroyAllWindows()

        # Start Threads
        self.threads = [
            threading.Thread(target=capture_loop, daemon=True),
            threading.Thread(target=inference_loop, daemon=True),
            threading.Thread(target=display_loop, daemon=True)
        ]
        for t in self.threads:
            t.start()

if __name__ == "__main__":
    # 1. Initialize the tracker
    tracker = PinoTracker()

    # 2. Start the background threads (Capture, Inference, Display)
    tracker.start()

    # 3. CRITICAL FIX: Keep the main thread alive!
    try:
        # Loop forever until the 'stop_event' is signaled inside the class
        # (The display_loop sets this event when you press 'q')
        while not tracker.stop_event.is_set():
            time.sleep(0.1)  # Sleep to save CPU while waiting

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nForce stopping...")

    # 4. Clean up once the loop breaks
    tracker.stop()