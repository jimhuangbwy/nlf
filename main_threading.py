import queue
import threading

import torch
import cv2
from pythonosc import udp_client
import time
import torch_tensorrt._C
from export_trt_model import export_trt_model


#from onnx_helper import ONNXClassifierWrapper

def send_joint(client, address, index, enable, position, rotation, serial):
    """
    Send joint position & rotation via OSC.

    Parameters:
        address (str): OSC address, e.g. "/joint"
        index (int): Joint index
        enable (int): Tracker enable flag
        position (tuple/list): (x, y, z) in float
        rotation (tuple/list): Quaternion (x, y, z, w)
        serial (str): Device serial
    """
    # Unpack inputs
    pos_x, pos_y, pos_z = position
    rot_x, rot_y, rot_z, rot_w = rotation

    # OSC message arguments (same order as Swift version)
    args = [
        index,
        enable,
        float(timeoffset),
        float(pos_x),
        float(pos_y),
        float(pos_z),
        float(rot_x),
        float(rot_y),
        float(rot_z),
        float(rot_w),
        serial
    ]

    client.send_message(address, args)

if __name__ == "__main__":
    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # --- Settings ---
    ZEROQ = (0, 0, 0, 1)
    ENABLE_TRACKER = 1
    ENABLE_TRACKING_REFERENCE = 4
    timeoffset = 0.0

    ###################################### Load full TorchScript model################################################
    model = export_trt_model(device)

    streams = [torch.cuda.Stream() for _ in range(2)]  # 雙 pipeline
    i = 0

    # with torch.inference_mode(), torch.device(device), torch.no_grad():
    ip = '127.0.0.1'
    port = 39570
    client = udp_client.SimpleUDPClient(ip, port)
    client.send_message("/VMT/SetRoomMatrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    send_joint(client, "/VMT/Follow/Unity", index=24, enable=ENABLE_TRACKING_REFERENCE, position=(-1.5, -0.7, 3), rotation=ZEROQ, serial="HMD") # 上半身的參考點

    cap = cv2.VideoCapture(0)
    frame_q = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=2)
    # cap.set(3, 640)  # width=1920
    # cap.set(4, 480)  # height=1080
    if not cap.isOpened():
        print("❌ Could not open webcam")
        exit()


    def reader():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if not frame_q.full():
                frame_q.put(frame)


    def inferencer():
        while True:
            frame = frame_q.get()
            frame_batch = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).to(device, non_blocking=True)
            with torch.inference_mode(), torch.no_grad():
                joints3D = model(frame_batch)
            result_q.put((frame, joints3D))
            frame_q.task_done()


    def displayer():
        prev_time = time.perf_counter()
        frame_count = 0
        while True:
            frame, joints3D = result_q.get()
            if len(joints3D) > 0:
                for j, joint3D in enumerate(joints3D[0]):
                    send_joint(client, "/VMT/Follow/Unity", index=j, enable=ENABLE_TRACKER,
                               position=-0.001 * joint3D, rotation=ZEROQ, serial="VMT_24")
            frame_count += 1
            if frame_count % 10 == 0:
                now = time.perf_counter()
                fps = 10 / (now - prev_time)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break


    threads = [
        threading.Thread(target=reader, daemon=True),
        threading.Thread(target=inferencer, daemon=True),
        threading.Thread(target=displayer, daemon=True),
    ]
    for t in threads: t.start()
    for t in threads: t.join()

        # # End timer
        # end_time = time.perf_counter()
        # loop_time = end_time - start_time
        # fps = 1.0 / loop_time
        #
        # #print('1st phase:', first_phase_time-start_time, '2nd phase:', sec_phase_time-first_phase_time, '3rd phase:', end_time-sec_phase_time )
        #
        # # Show FPS on screen
        # cv2.putText(image, f"FPS: {fps:.2f}", (20, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #
        # # Display
        # cv2.imshow("Webcam - Press Q to quit", image)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    cap.release()
    cv2.destroyAllWindows()
