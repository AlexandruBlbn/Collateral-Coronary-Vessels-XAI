"""
Top-level Gradio entrypoint for online deployment.

Usage (local or server):
    python app.py

Environment variables:
    DINO_BACKBONE_PATH   Path to best_backbone.pth
    DINO_ENCODER_NAME    timm encoder name (default: efficientnetv2_s)
    DINO_INPUT_SIZE      Input size (default: 512)
    DINO_DEVICE          auto|cpu|cuda (default: auto)
    DINO_USE_AMP         1/0 (default: 1)
    PORT                 Server port for hosted environments
    GRADIO_SHARE         1/0 enable share link in local mode (default: 1)
    PUBLIC_PROVIDER      auto|gradio|cloudflared|none (default: auto; prefers cloudflared when available)
"""

import os

from utils.dinov3_point_heatmap_app import create_demo, launch_demo


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}


def build_demo_from_env():
    backbone_path = os.getenv(
        'DINO_BACKBONE_PATH',
        'checkpoints/dinov3/dinov3_pretrain_efficientnetv2_s/best_backbone.pth',
    )
    encoder_name = os.getenv('DINO_ENCODER_NAME', 'efficientnetv2_s')
    input_size = int(os.getenv('DINO_INPUT_SIZE', '512'))
    device_name = os.getenv('DINO_DEVICE', 'auto')
    use_amp = _env_flag('DINO_USE_AMP', True)

    return create_demo(
        backbone_path=backbone_path,
        encoder_name=encoder_name,
        input_size=input_size,
        device_name=device_name,
        use_amp=use_amp,
    )


demo = build_demo_from_env()


if __name__ == '__main__':
    share = _env_flag('GRADIO_SHARE', True)
    public_provider = os.getenv('PUBLIC_PROVIDER', 'auto')
    launch_demo(
        demo=demo,
        host='0.0.0.0',
        port=int(os.getenv('PORT', '7860')),
        share=share,
        online=share,
        public_provider=public_provider,
    )
