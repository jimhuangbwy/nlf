import tensorflow as tf
import os
import cv2
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

from pythonosc import udp_client

def download_model(model_type):
    model_zippath = f'C:\\Users\\12332\\.keras\\models\\'
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path

# --- Settings ---
ZEROQ = (0,0,0,1)
ENABLE_TRACKER = 1
ENABLE_TRACKING_REFERENCE = 4
timeoffset = 0.0

def sendJoint(client, address, index, enable, position, rotation, serial):
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
    print(tf.config.list_logical_devices('GPU'))
    with tf.device('/GPU:0'):
        model = tf.saved_model.load(download_model('nlf_s'))

        ip = '127.0.0.1'
        port = 39570

        client = udp_client.SimpleUDPClient(ip, port)

        client.send_message("/VMT/SetRoomMatrix", [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0])

        sendJoint(client, "/VMT/Follow/Unity", index=24, enable=ENABLE_TRACKING_REFERENCE, position=(-1.5, -0.7, 3), rotation=ZEROQ, serial="HMD") # 上半身的參考點

        cam = cv2.VideoCapture(0)

        while True:
            ret, image = cam.read()

            if not ret:
                print("Cannot grad image")
                continue

            cv2.imshow('Cam', image)
            cv2.waitKey(1)
            #skeleton = 'smpl_24' # 'smpl+head_30'
            pred = model.detect_smpl(image)

            if len(pred['joints3d'].numpy()) > 0:
                joints3D = pred['joints3d'].numpy()

                for j, joint3D in enumerate(joints3D[0]):
                    sendJoint(client, "/VMT/Follow/Unity", index=j, enable=ENABLE_TRACKER, position=-joint3D, rotation=ZEROQ , serial="VMT_24")
