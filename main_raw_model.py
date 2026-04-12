import threading
from collections import deque

import torch
import torchvision
import cv2
from pythonosc import udp_client
import time


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
    model = torch.jit.load("models/nlf_l_multi_0.3.2.torchscript").eval().to(device)

    ##################################################################################################################

    ip = '127.0.0.1'
    port = 39570
    client = udp_client.SimpleUDPClient(ip, port)
    client.send_message("/VMT/SetRoomMatrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])
    send_joint(client, "/VMT/Follow/Unity", index=24, enable=ENABLE_TRACKING_REFERENCE, position=(-1.5, -0.7, 3), rotation=ZEROQ, serial="HMD") # 上半身的參考點

    cap = cv2.VideoCapture(0)
    # cap.set(3, 640)  # width=1920
    # cap.set(4, 480)  # height=1080
    if not cap.isOpened():
        print("❌ Could not open webcam")
        exit()

    # --- 建立影像緩衝區 ---
    buffer = deque(maxlen=2)  # 雙 buffer，可改成3以增加緩衝
    stop_flag = False

    # --- 子執行緒：讀取影像 ---
    def capture_thread():
        global stop_flag
        while not stop_flag:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                stop_flag = True
                break
            buffer.append(frame)  # 加入緩衝區

    # 啟動子執行緒
    t = threading.Thread(target=capture_thread, daemon=True)
    t.start()

    while True:
        start_time = time.perf_counter()  # start timer

        # ret, image = cap.read()
        # if not ret:
        #     print("❌ Failed to grab frame")
        #     break

        if len(buffer) == 0:
            # 沒有影像可用時略過本輪
            continue

        # 取出最近影像（不會阻塞）
        image = buffer.popleft()
        frame = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device) #201

        #frame_batch = torchvision.transforms.v2.functional.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).to(device).unsqueeze(0)
        #frame_batch_linear = im_to_linear(frame_batch)
        #boxes = detector(frame_batch_linear, max_detections=1)

        with torch.inference_mode(), torch.no_grad():
            pred = model.detect_smpl_batched(frame)
            joints3D = pred['joints3d'][0]
            #print(joints3D)

        if len(joints3D) > 0:
            joints3D = joints3D[0]
            for j, joint3D in enumerate(joints3D):
                send_joint(client, "/VMT/Follow/Unity", index=j, enable=ENABLE_TRACKER, position=-0.001 * joint3D, rotation=ZEROQ, serial="VMT_24")

        # End timer
        end_time = time.perf_counter()
        loop_time = end_time - start_time
        fps = 1.0 / loop_time

        #print('1st phase:', first_phase_time-start_time, '2nd phase:', sec_phase_time-first_phase_time, '3rd phase:', end_time-sec_phase_time )

        # Show FPS on screen
        cv2.putText(image, f"FPS: {fps:.2f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("Webcam - Press Q to quit", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
