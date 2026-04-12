import time
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.ops

from onnx_helper import TRTInference

class PersonDetector(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.input_size = 640
        self.model = TRTInference(model_path) #torch.jit.load(model_path).half()
        #self.person_class_id = '0'
        #self.tracker_instance = IoUTracker(patience=15)  # Wait 15 frames if object is lost
        #self.current_target_id = None
        #tracker = SinglePersonTracker()

    def forward(
        self,
        images: torch.Tensor,
        threshold: float = 0.3,
        nms_iou_threshold: float = 0.7,
        max_detections: int = 1,
        extrinsic_matrix: Optional[torch.Tensor] = None,
        world_up_vector: Optional[torch.Tensor] = None,
        flip_aug: bool = False,
        bothflip_aug: bool = False,
        extra_boxes: Optional[List[torch.Tensor]] = None,
    ):
        device = images.device
        # extrinsic_matrix = torch.eye(4, device=device).unsqueeze(0)
        # world_up_vector = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)
        images, x_factor, y_factor, half_pad_h_float, half_pad_w_float = resize_and_pad(
            images, self.input_size
        )

        # cam_up_vector = matvec(extrinsic_matrix[:, :3, :3], world_up_vector)
        # angle = torch.atan2(cam_up_vector[:, 1], cam_up_vector[:, 0])
        # k = (torch.round(angle / (torch.pi / 2)) + 1).to(torch.int32) % 4
        # print(k)
        #images = batched_rot90(images, k)
        k = torch.zeros(1, device=device, dtype=torch.int32)

        boxes, scores = self.call_model(images, threshold)  #x1y1x2y2

        # Convert from cxcywh to xyxy (top-left-bottom-right)
        # boxes = torch.stack(
        #     [
        #         boxes[..., 0] - boxes[..., 2] / 2,
        #         boxes[..., 1] - boxes[..., 3] / 2,
        #         boxes[..., 0] + boxes[..., 2] / 2,
        #         boxes[..., 1] + boxes[..., 3] / 2,
        #     ],
        #     dim=-1,
        # )

        # scores = scores[..., 0].to(device)

        # boxes_nms, scores_nms, new_target_id = nms_and_track(
        #     boxes,
        #     scores,
        #     threshold,
        #     nms_iou_threshold,
        #     max_detections=100,
        #     tracker=self.tracker_instance,  # Pass state
        #     locked_id=self.current_target_id  # Pass current target
        # )

        # if new_target_id is not None:
        #     # 'tracks' is the dictionary inside the tracker containing active IDs
        #     if new_target_id not in self.tracker_instance.tracks:
        #         new_target_id = None

        # # 3. Update state
        # self.current_target_id = new_target_id

        # boxes_nms, scores_nms = nms_max_score(
        #     boxes, scores, threshold, nms_iou_threshold, max_detections
        # )

        return [
            scale_boxes(
                boxes,
                scores,
                half_pad_w_float,
                half_pad_h_float,
                x_factor,
                y_factor,
                k,
                self.input_size,
            )
        ]

    def call_model(self, images, threshold):
        preds = self.model(images).float()
        mask = (preds[..., 4] > threshold) & (preds[..., 5] == 0)
        preds = preds[mask]
        preds = preds[:1]
        boxes = preds[..., 0:4]
        scores = preds[..., 4]

        #preds = torch.permute(preds, [0, 2, 1])  # [batch, n_boxes, 84]
        #boxes = preds[..., :4]
        #scores = preds[..., 4:]
        return boxes, scores  #x1y1x2y2

# def nms_max_score(
#     boxes: torch.Tensor,       # (B, N, 4)
#     scores: torch.Tensor,      # (B, N)
#     threshold: float,
#     nms_iou_threshold: float,
#     max_detections: int,
# ):
#
#     B, N, _ = boxes.shape
#
#     selected_boxes = []
#     selected_scores = []
#
#     score_max = torch.argmax(scores)
#     score_max_id = score_max.item()
#
#     if score_max > threshold:
#         selected_boxes.append(boxes[0][score_max_id].unsqueeze(0))
#         selected_scores.append(scores[0][score_max_id].unsqueeze(0))
#
#     return selected_boxes, selected_scores
#
#
# def nms(
#     boxes: torch.Tensor,       # (B, N, 4)
#     scores: torch.Tensor,      # (B, N)
#     threshold: float,
#     nms_iou_threshold: float,
#     max_detections: int,
# ):
#
#     B, N, _ = boxes.shape
#
#     # mask once
#     mask = scores > threshold
#
#     # flatten
#     all_boxes  = boxes[mask]        # (M,4)
#     all_scores = scores[mask]       # (M)
#
#     # build batch idx for each
#     batch_idx = torch.arange(B, device=boxes.device).repeat_interleave(N)[mask.view(-1)]
#
#     # NMS one shot
#     keep = torchvision.ops.batched_nms(
#         all_boxes,
#         all_scores,
#         batch_idx,
#         nms_iou_threshold,
#     )
#
#     # highest score first
#     keep = keep[all_scores[keep].argsort(descending=True)]
#
#     selected_boxes = []
#     selected_scores = []
#
#     ids = keep[batch_idx[keep] == 0][:max_detections]
#     selected_boxes.append(all_boxes[ids])
#     selected_scores.append(all_scores[ids])
#
#     return selected_boxes, selected_scores


# def nms_with_extra(
#     boxes: torch.Tensor,
#     scores: torch.Tensor,
#     extra_boxes: List[torch.Tensor],
#     extra_scores: List[torch.Tensor],
#     threshold: float,
#     nms_iou_threshold: float,
#     max_detections: int,
# ):
#     selected_boxes = []
#     selected_scores = []
#     for boxes_now_, scores_now_, extra_boxes_now, extra_scores_now in zip(
#         boxes, scores, extra_boxes, extra_scores
#     ):
#         boxes_now = torch.cat([boxes_now_, extra_boxes_now], dim=0)
#         scores_now = torch.cat([scores_now_, extra_scores_now], dim=0)
#         is_above_threshold = scores_now > threshold
#         boxes_now = boxes_now[is_above_threshold]
#         scores_now = scores_now[is_above_threshold]
#         nms_indices = torchvision.ops.nms(boxes_now, scores_now, nms_iou_threshold)[
#             :max_detections
#         ]
#         selected_boxes.append(boxes_now[nms_indices])
#         selected_scores.append(scores_now[nms_indices])
#     boxes = selected_boxes
#     scores = selected_scores
#     return boxes, scores


# def batched_rot90(images, k):
#     batch_size = images.size(0)
#     rotated_images = torch.empty_like(images)
#     for i in range(batch_size):
#         rotated_images[i] = torch.rot90(images[i], k=k[i], dims=[1, 2])
#     return rotated_images


# def matvec(a, b):
#     return (a @ b.unsqueeze(-1)).squeeze(-1)


def resize_and_pad(images: torch.Tensor, input_size: int):
    h = float(images.shape[2])
    w = float(images.shape[3])
    max_side = max(h, w)
    factor = float(input_size) / max_side
    target_w = int(factor * w)
    target_h = int(factor * h)
    y_factor = h / float(target_h)
    x_factor = w / float(target_w)
    pad_h = input_size - target_h
    pad_w = input_size - target_w
    half_pad_h = pad_h // 2
    half_pad_w = pad_w // 2
    half_pad_h_float = float(half_pad_h)
    half_pad_w_float = float(half_pad_w)

    images = F.interpolate(
        images,
        (target_h, target_w),
        mode='bilinear' if factor > 1 else 'area',
        align_corners=False if factor > 1 else None,
    )
    # images = F.interpolate(
    #    images, (target_h, target_w), antialias=factor < 1)
    images **= 1 / 2.2
    images = F.pad(
        images, (half_pad_w, pad_w - half_pad_w, half_pad_h, pad_h - half_pad_h), value=0.5
    )
    return images, x_factor, y_factor, half_pad_h_float, half_pad_w_float


def scale_boxes(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    half_pad_w_float: float,
    half_pad_h_float: float,
    x_factor: float,
    y_factor: float,
    k: torch.Tensor,
    input_size: int,
):
    midpoints = (boxes[:, :2] + boxes[:, 2:]) / 2
    # midpoints = (
    #     matvec(rotmat2d(k * (torch.pi / 2)), midpoints - (input_size - 1) / 2)
    #     + (input_size - 1) / 2
    # )

    sizes = boxes[:, 2:] - boxes[:, :2]
    # if k % 2 == 1:
    #     sizes = torch.flip(sizes, [1])

    boxes_ = torch.cat([midpoints - sizes / 2, sizes], dim=1)

    return torch.stack(
        [
            (boxes_[:, 0] - half_pad_w_float) * x_factor,
            (boxes_[:, 1] - half_pad_h_float) * y_factor,
            (boxes_[:, 2]) * x_factor,
            (boxes_[:, 3]) * y_factor,
            scores,
        ],
        dim=1,
    )  # 左上xy, w, h


# def inv_scale_boxes(
#     boxes: torch.Tensor,
#     half_pad_w_float: float,
#     half_pad_h_float: float,
#     x_factor: float,
#     y_factor: float,
#     k: torch.Tensor,
#     input_size: int,
# ):
#     boxes_ = torch.stack(
#         [
#             boxes[:, 0] / x_factor + half_pad_w_float,
#             boxes[:, 1] / y_factor + half_pad_h_float,
#             boxes[:, 2] / x_factor,
#             boxes[:, 3] / y_factor,
#         ],
#         dim=1,
#     )
#
#     sizes = boxes_[:, 2:]
#     midpoints = boxes_[:, :2] + sizes / 2
#
#     if k % 2 == 1:
#         sizes = torch.flip(sizes, [1])
#
#     midpoints = (
#         matvec(rotmat2d(-k.to(torch.float32) * (torch.pi / 2)), midpoints - (input_size - 1) / 2)
#         + (input_size - 1) / 2
#     )
#
#     return torch.cat([midpoints - sizes / 2, midpoints + sizes / 2], dim=1)


# def rotmat2d(angle: torch.Tensor):
#     sin = torch.sin(angle)
#     cos = torch.cos(angle)
#     entries = [cos, -sin, sin, cos]
#     result = torch.stack(entries, dim=-1)
#     return torch.reshape(result, angle.shape + (2, 2))
