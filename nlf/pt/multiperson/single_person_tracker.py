import math

def bbox_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0

    boxA_area = (boxA[2]-boxA[0]) * (boxA[3]-boxA[1])
    boxB_area = (boxB[2]-boxB[0]) * (boxB[3]-boxB[1])
    union = boxA_area + boxB_area - inter
    return inter / union


def center_distance_score(a, b, max_dist=200):
    ax = (a[0] + a[2]) / 2
    ay = (a[1] + a[3]) / 2
    bx = (b[0] + b[2]) / 2
    by = (b[1] + b[3]) / 2

    d = math.sqrt((ax - bx)**2 + (ay - by)**2)
    return max(0, 1 - d / max_dist)


def size_similarity(a, b):
    aw = a[2] - a[0]
    ah = a[3] - a[1]
    bw = b[2] - b[0]
    bh = b[3] - b[1]
    ratio = min(aw, bw) / max(aw, bw) * min(ah, bh) / max(ah, bh)
    return ratio

class SinglePersonTracker:
    def __init__(self):
        self.target_id = 1       # manually assign a stable ID
        self.target_box = None
        self.missed_frames = 0

    def init_target(self, box):
        self.target_box = box
        self.missed_frames = 0

    def update(self, detected_boxes):
        if self.target_box is None:
            return None, None  # no target yet

        # match the box with best score
        best_score = -1
        best_box = None

        for box in detected_boxes:
            iou = bbox_iou(self.target_box, box)
            cd  = center_distance_score(self.target_box, box)
            ss  = size_similarity(self.target_box, box)

            score = 0.5*iou + 0.3*cd + 0.2*ss
            if score > best_score:
                best_score = score
                best_box = box

        if best_score < 0.2:
            # No good match → treat as occluded
            self.missed_frames += 1
            if self.missed_frames < 30:
                return self.target_id, self.target_box
            else:
                return None, None
        else:
            # Good match → update target
            self.missed_frames = 0
            self.target_box = best_box
            return self.target_id, best_box
