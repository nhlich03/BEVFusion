import torch
import torch.nn as nn
import numpy as np
from mmcv.runner import auto_fp16, force_fp32
import torch.nn.functional as F
from torchvision.transforms.functional import rotate

from mmdet3d.models.builder import build_backbone, build_fuser, build_head, build_neck, build_vtransform
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS
from .base import Base3DFusionModel
from mmdet3d.models.utils.bevformer_modules import SpatialCrossAttention, TemporalSelfAttention

def digit_version(version_str):
    return tuple(map(int, (version_str.split("."))))


@FUSIONMODELS.register_module()
class BEVFusion4D(Base3DFusionModel):
    def __init__(
        self,
        encoders,
        decoder,
        heads,
        num_cams=6,
        embed_dims=256,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        num_points_in_pillar=4,
        rotate_center=[100, 100],
        can_bus_norm=True,
        spatial_cross_attention=dict(
            type='SpatialCrossAttention',
            embed_dims=256,
            num_cams=6,
            deformable_attention=dict(
                type='MSDeformableAttention3D',
                embed_dims=256,
                num_levels=4)),
        temporal_self_attention=dict(
            type='TemporalSelfAttention',
            embed_dims=256,
            num_bev_queue=2,
            num_levels=1),
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.pc_range = pc_range
        self.num_points_in_pillar = num_points_in_pillar
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.rotate_center = rotate_center
        
        from mmcv.cnn.bricks.transformer import build_attention
        self.spatial_cross_attention = build_attention(spatial_cross_attention)
        self.temporal_self_attention = build_attention(temporal_self_attention)

        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d

    @force_fp32(apply_to=('reference_points', 'lidar2img'))
    def point_sampling(self, reference_points, pc_range, lidar2img, img_shape):
        allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        reference_points = reference_points.clone()
        reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
        reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
        reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1)
        reference_points = reference_points.permute(1, 0, 2, 3)
        D, B, num_query = reference_points.size()[:3]
        num_cam = lidar2img.size(1)

        reference_points = reference_points.view(D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)

        reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
        eps = 1e-5

        bev_mask = (reference_points_cam[..., 2:3] > eps)
        reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)

        reference_points_cam[..., 0] /= img_shape[1]
        reference_points_cam[..., 1] /= img_shape[0]

        bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                    & (reference_points_cam[..., 1:2] < 1.0)
                    & (reference_points_cam[..., 0:1] < 1.0)
                    & (reference_points_cam[..., 0:1] > 0.0))
        bev_mask = torch.nan_to_num(bev_mask)

        reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
        bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

        return reference_points_cam, bev_mask

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points, sensor):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders[sensor]["voxelize"](res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def extract_lidar_features(self, points, sensor="lidar") -> torch.Tensor:
        feats, coords, sizes = self.voxelize(points, sensor)
        batch_size = coords[-1, 0] + 1
        x = self.encoders[sensor]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    def extract_camera_features(self, x) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)
        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)
        if not isinstance(x, torch.Tensor):
            x = x[0]
        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        return x

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        prev_bev=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img, points, camera2ego, lidar2ego, lidar2camera, lidar2image,
                camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix,
                metas, depths, radar, gt_masks_bev, gt_bboxes_3d, gt_labels_3d, prev_bev, **kwargs
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        depths=None,
        radar=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        prev_bev=None,
        **kwargs,
    ):
        bs = len(metas)
        # 1. LiDAR BEV Extraction (serves as queries)
        lidar_bev_feat = self.extract_lidar_features(points, sensor="lidar")  # shape: (B, C, bev_h, bev_w)
        B, C, bev_h, bev_w = lidar_bev_feat.shape
        bev_queries = lidar_bev_feat.flatten(2).permute(0, 2, 1)  # (B, bev_h*bev_w, C)
        
        # 2. Camera Feature Extraction
        cam_feats = self.extract_camera_features(img)  # shape: (B, num_cam, C, fH, fW)
        
        # Spatial Shapes and Level Start Index
        B, num_cam, C_cam, fH, fW = cam_feats.shape
        spatial_shapes = torch.tensor([[fH, fW]], device=cam_feats.device)
        level_start_index = torch.tensor([0], device=cam_feats.device)
        
        cam_feats_flatten = cam_feats.flatten(3).permute(1, 0, 3, 2).flatten(2).unsqueeze(0).permute(1, 3, 2, 0).squeeze(-1)

        # 3. Reference Points and Point Sampling
        ref_3d = self.get_reference_points(
            bev_h, bev_w, self.pc_range[5] - self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bs, device=cam_feats.device, dtype=cam_feats.dtype)
        
        img_shape = metas[0]['img_shape'][0] if 'img_shape' in metas[0] else (img.shape[-2], img.shape[-1]) 

        reference_points_cam, bev_mask = self.point_sampling(
            ref_3d, self.pc_range, lidar2image, img_shape)
        
        # 4. Spatial Cross Attention
        bev_embed = self.spatial_cross_attention(
            query=bev_queries,
            key=cam_feats_flatten,
            value=cam_feats_flatten,
            reference_points=ref_3d,
            reference_points_cam=reference_points_cam,
            bev_mask=bev_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )

        # 5. Ego-Motion Calibration
        # Rotate prev_bev by ego-motion
        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(0, 2, 1) # bs, H*W, C -> bs, C, H*W
            prev_bev = prev_bev.view(bs, C, bev_h, bev_w)
            for i in range(bs):
                rotation_angle = metas[i]['can_bus'][-1]
                tmp_prev_bev = prev_bev[i]
                tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                prev_bev[i] = tmp_prev_bev
            prev_bev = prev_bev.flatten(2).permute(0, 2, 1) # bs, H*W, C

        # Inject CAN bus features into queries
        can_bus = bev_queries.new_tensor([each.get('can_bus', np.zeros(18)) for each in metas])
        can_bus = self.can_bus_mlp(can_bus)[:, None, :]
        bev_embed = bev_embed + can_bus
        
        # 6. Temporal Self Attention
        ref_2d = self.get_reference_points(bev_h, bev_w, dim='2d', bs=bs, device=cam_feats.device, dtype=cam_feats.dtype)
        
        delta_x = np.array([each.get('can_bus', np.zeros(18))[0] for each in metas])
        delta_y = np.array([each.get('can_bus', np.zeros(18))[1] for each in metas])
        ego_angle = np.array([each.get('can_bus', np.zeros(18))[-2] / np.pi * 180 for each in metas])
        
        translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        # Real resolutions
        grid_length_y = (self.pc_range[4] - self.pc_range[1]) / bev_h
        grid_length_x = (self.pc_range[3] - self.pc_range[0]) / bev_w
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift = bev_embed.new_tensor([shift_x, shift_y]).permute(1, 0)
        
        shift_ref_2d = ref_2d.clone()
        shift_ref_2d += shift[:, None, None, :]
        
        num_bev_level = ref_2d.shape[2]
        if prev_bev is not None:
             hybird_ref_2d = torch.stack([shift_ref_2d, ref_2d], 1).reshape(bs*2, bev_h*bev_w, num_bev_level, 2)
        else:
             hybird_ref_2d = torch.stack([ref_2d, ref_2d], 1).reshape(bs*2, bev_h*bev_w, num_bev_level, 2)

        # Permute for attention which uses batch_first False
        bev_embed = bev_embed.permute(1, 0, 2) # (num_query, bs, embed_dims)
        
        # For MSA Spatial Shapes of BEV
        bev_spatial_shapes = torch.tensor([[bev_h, bev_w]], device=bev_embed.device)
        bev_level_start_index = torch.tensor([0], device=bev_embed.device)

        bev_embed = self.temporal_self_attention(
            query=bev_embed,
            value=prev_bev.permute(1, 0, 2) if prev_bev is not None else None,
            identity=bev_embed,
            reference_points=hybird_ref_2d,
            spatial_shapes=bev_spatial_shapes,
            level_start_index=bev_level_start_index
        )
        
        # Return to (bs, C, bev_h, bev_w)
        bev_embed = bev_embed.permute(1, 2, 0).view(bs, self.embed_dims, bev_h, bev_w)

        # 7. Decoder and Heads
        x = self.decoder["backbone"](bev_embed)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(bs)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(bs):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
            return outputs
