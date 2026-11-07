from .dit import ConvDiT
from .unet import AttentionUNet


def architecture_kwargs(kind, config):
    if kind not in ("dit", "unet"):
        raise ValueError("model must be dit or unet")
    keys = (
        (
            "image_size",
            "hidden_size",
            "patch_size",
            "depth",
            "num_heads",
            "mlp_ratio",
            "dropout",
        )
        if kind == "dit"
        else (
            "image_size",
            "model_channels",
            "channel_mult",
            "num_res_blocks",
            "attention_resolutions",
            "num_heads",
            "dropout",
        )
    )
    return dict(in_channels=config["channels"], **{key: config[key] for key in keys})


def build_model(kind, config):
    return (ConvDiT if kind == "dit" else AttentionUNet)(
        **architecture_kwargs(kind, config)
    )
