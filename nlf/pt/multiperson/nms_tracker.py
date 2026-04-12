import torch
import torchvision
import numpy as np


class IoUTracker:
    """
    Simple IoU Tracker to maintain ID state across frames.
    """

    def __init__(self, patience=20, iou_threshold=0.3):
        self.tracks = {}  # {id: [cx, cy, w, h]}
        self.patience = patience
        self.lost_count = {}  # {id: frames_lost}
        self.id_count = 0
        self.iou_threshold = iou_threshold

    def update(self, boxes):
        # boxes: list of [cx, cy, w, h]
        # returns: list of [cx, cy, w, h, id]
        active_tracks = list(self.tracks.items())
        matched_track_indices = set()
        matched_box_indices = set()

        # 1. Calculate IoU matches
        matches = []
        for i, (track_id, track_box) in enumerate(active_tracks):
            for j, new_box in enumerate(boxes):
                iou = self._calculate_iou(track_box, new_box)
                if iou > self.iou_threshold:
                    matches.append((iou, i, j))

        matches.sort(key=lambda x: x[0], reverse=True)

        # 2. Assign matches
        for iou, track_idx, box_idx in matches:
            if track_idx in matched_track_indices or box_idx in matched_box_indices:
                continue
            track_id = active_tracks[track_idx][0]
            self.tracks[track_id] = boxes[box_idx]
            self.lost_count[track_id] = 0
            matched_track_indices.add(track_idx)
            matched_box_indices.add(box_idx)

        # 3. Handle Lost
        for i, (track_id, track_box) in enumerate(active_tracks):
            if i not in matched_track_indices:
                self.lost_count[track_id] += 1
                if self.lost_count[track_id] > self.patience:
                    del self.tracks[track_id]
                    del self.lost_count[track_id]

        # 4. Handle New
        for j, new_box in enumerate(boxes):
            if j not in matched_box_indices:
                self.tracks[self.id_count] = new_box
                self.lost_count[self.id_count] = 0
                self.id_count += 1

        # 5. Return active tracks
        results = []
        for track_id, box in self.tracks.items():
            if self.lost_count[track_id] < 5:  # Hide if lost for > 5 frames
                results.append(box + [track_id])
        return results

    def _calculate_iou(self, boxA, boxB):
        # box: cx, cy, w, h
        xA = max(boxA[0] - boxA[2] / 2, boxB[0] - boxB[2] / 2)
        yA = max(boxA[1] - boxA[3] / 2, boxB[1] - boxB[3] / 2)
        xB = min(boxA[0] + boxA[2] / 2, boxB[0] + boxB[2] / 2)
        yB = min(boxA[1] + boxA[3] / 2, boxB[1] + boxB[3] / 2)
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        denom = boxAArea + boxBArea - interArea
        if denom == 0: return 0
        return interArea / denom


def nms_and_track(
        boxes: torch.Tensor,  # (B, N, 4)
        scores: torch.Tensor,  # (B, N)
        threshold: float,
        nms_iou_threshold: float,
        max_detections: int,
        tracker: IoUTracker,  # NEW: Pass the tracker instance
        locked_id: int = None  # NEW: The ID we are currently tracking
):
    B, N, _ = boxes.shape

    # --- ORIGINAL NMS LOGIC ---
    mask = scores > threshold
    all_boxes = boxes[mask]
    all_scores = scores[mask]

    batch_idx = torch.arange(B, device=boxes.device).repeat_interleave(N)[mask.view(-1)]

    keep = torchvision.ops.batched_nms(
        all_boxes,
        all_scores,
        batch_idx,
        nms_iou_threshold,
    )

    keep = keep[all_scores[keep].argsort(descending=True)]

    # Get valid detections for the first image in batch (assuming B=1 for tracking)
    # If B > 1, you would need a loop here for every batch item
    valid_ids = keep[batch_idx[keep] == 0][:max_detections]

    det_boxes = all_boxes[valid_ids]  # (M, 4) in [x1, y1, x2, y2]
    det_scores = all_scores[valid_ids]  # (M)

    # --- NEW TRACKING LOGIC ---

    selected_boxes = []
    selected_scores = []
    new_locked_id = locked_id

    if len(det_boxes) > 0:
        # 1. Convert Tensor [x1, y1, x2, y2] -> Numpy [cx, cy, w, h] for tracker
        # We process on CPU to avoid complex tensor indexing for the tracker logic
        dets_cpu = det_boxes.detach().cpu().numpy()
        w = dets_cpu[:, 2] - dets_cpu[:, 0]
        h = dets_cpu[:, 3] - dets_cpu[:, 1]
        cx = dets_cpu[:, 0] + (w / 2)
        cy = dets_cpu[:, 1] + (h / 2)
        tracker_inputs = np.column_stack((cx, cy, w, h)).tolist()

        # 2. Update Tracker
        # returns list of [cx, cy, w, h, id]
        tracked_objects = tracker.update(tracker_inputs)

        # 3. Filter for Target
        final_box = None
        final_score = None

        # A. If we don't have a target, Lock the first one found (Highest Score)
        if new_locked_id is None and len(tracked_objects) > 0:
            # Usually the first item in 'tracker_inputs' (highest score)
            # maps to a specific ID in 'tracked_objects'.
            # Ideally we match geometrically, but for simplicity:
            new_locked_id = tracked_objects[0][4]

            # B. Find the locked ID
        for obj in tracked_objects:
            if obj[4] == new_locked_id:
                tcx, tcy, tw, th, tid = obj

                # Convert back to Tensor [x1, y1, x2, y2]
                tx1 = tcx - (tw / 2)
                ty1 = tcy - (th / 2)
                tx2 = tcx + (tw / 2)
                ty2 = tcy + (th / 2)

                # Create tensor on original device
                final_box = torch.tensor([[tx1, ty1, tx2, ty2]], device=boxes.device, dtype=boxes.dtype)

                # Use a default score or try to map back to original score.
                # Here we use 1.0 to indicate "Tracked & Confirmed"
                final_score = torch.tensor([1.0], device=scores.device, dtype=scores.dtype)
                break

        if final_box is not None:
            selected_boxes.append(final_box)
            selected_scores.append(final_score)
        else:
            # Target lost temporarily (tracker memory might still hold it, but we don't draw it)
            selected_boxes.append(torch.empty((0, 4), device=boxes.device))
            selected_scores.append(torch.empty((0), device=scores.device))

    else:
        # No detections
        selected_boxes.append(torch.empty((0, 4), device=boxes.device))
        selected_scores.append(torch.empty((0), device=scores.device))

    # Returns: boxes, scores, and the ID to save for the next frame
    return selected_boxes, selected_scores, new_locked_id