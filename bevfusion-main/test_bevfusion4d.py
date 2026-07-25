import torch
import numpy as np

# Adjust imports paths
import sys
sys.path.append('.')

from mmdet3d.models import build_model
from mmcv import Config

def test_bevfusion4d():
    print("Loading config...")
    # Using a typical nuscenes config
    cfg = Config.fromfile('configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/default.yaml')
    
    # We override model type to BEVFusion4D
    cfg.model.type = 'BEVFusion4D'
    
    # Ensure camera encoder has backbone and neck, vtransform will be ignored by BEVFusion4D
    # The default configs usually inherit from components. 
    # Because cfg fromfile might have some missing base fields if it relies on MMCV inheritance, 
    # let's just make a very basic config dictionary if it fails.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Building model...")
    try:
        model = build_model(cfg.model)
        model = model.to(device)
    except Exception as e:
        print(f"Failed to build from config: {e}")
        print("Falling back to manual dictionary...")
        dummy_cfg = dict(
            type='BEVFusion4D',
            encoders=dict(
                camera=dict(
                    backbone=dict(type='ResNet', depth=50, num_stages=4, out_indices=(0, 1, 2, 3)),
                    neck=dict(type='FPN', in_channels=[256, 512, 1024, 2048], out_channels=256, num_outs=4)
                ),
                lidar=dict(
                    voxelize=dict(max_num_points=10, point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], voxel_size=[0.1, 0.1, 0.2], max_voxels=[90000, 120000]),
                    backbone=dict(
                        type='SparseEncoder',
                        in_channels=5,
                        sparse_shape=[1024, 1024, 41],
                        output_channels=128,
                        order=['conv', 'norm', 'act'],
                        encoder_channels=[[16, 16, 32], [32, 32, 64], [64, 64, 128], [128, 128]],
                        encoder_paddings=[[0, 0, 1], [0, 0, 1], [0, 0, [1, 1, 0]], [0, 0]],
                        block_type='basicblock'
                    )
                )
            ),
            decoder=dict(backbone=dict(type='SECOND', in_channels=256, out_channels=[128, 256], layer_nums=[5, 5], layer_strides=[1, 2]), neck=dict(type='SECONDFPN', in_channels=[128, 256], out_channels=[256, 256], upsample_strides=[1, 2])),
            heads=dict(
                object=dict(type='TransFusionHead', num_proposals=200, in_channels=512, out_channels=512, bbox_coder=dict(type='TransFusionBBoxCoder', pc_range=[-51.2, -51.2], post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], score_threshold=0.0, out_size_factor=8, voxel_size=[0.1, 0.1]), num_classes=10)
            )
        )
        model = build_model(dummy_cfg)
        model = model.to(device)
    
    print("Model built successfully!")
    print(f"Model class: {model.__class__.__name__}")
    
    print("Preparing dummy inputs...")
    B = 1
    num_cams = 6
    img = torch.randn(B, num_cams, 3, 256, 704).to(device)
    # Pts is list of tensors [N, 5]
    points = [torch.randn(1000, 5).to(device) for _ in range(B)]
    # Transform matrices
    camera2ego = torch.eye(4)[None, None].repeat(B, num_cams, 1, 1).to(device)
    lidar2ego = torch.eye(4)[None].repeat(B, 1, 1).to(device)
    lidar2camera = torch.eye(4)[None, None].repeat(B, num_cams, 1, 1).to(device)
    lidar2image = torch.eye(4)[None, None].repeat(B, num_cams, 1, 1).to(device)
    camera_intrinsics = torch.eye(3)[None, None].repeat(B, num_cams, 1, 1).to(device)
    camera2lidar = torch.eye(4)[None, None].repeat(B, num_cams, 1, 1).to(device)
    img_aug_matrix = torch.eye(4)[None, None].repeat(B, num_cams, 1, 1).to(device)
    lidar_aug_matrix = torch.eye(4)[None].repeat(B, 1, 1).to(device)

    metas = [
        dict(
            can_bus=np.zeros(18),
            img_shape=[(256, 704)] * num_cams
        ) for _ in range(B)
    ]
    
    print("Running forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            outputs = model(
                img=img,
                points=points,
                camera2ego=camera2ego,
                lidar2ego=lidar2ego,
                lidar2camera=lidar2camera,
                lidar2image=lidar2image,
                camera_intrinsics=camera_intrinsics,
                camera2lidar=camera2lidar,
                img_aug_matrix=img_aug_matrix,
                lidar_aug_matrix=lidar_aug_matrix,
                metas=metas,
                depths=None
            )
        print("Forward pass successful!")
        if isinstance(outputs, list):
            print(f"Output is a list of {len(outputs)} dicts")
            for k, v in outputs[0].items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: tensor of shape {v.shape}")
        else:
            print(outputs)
    except Exception as e:
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_bevfusion4d()
