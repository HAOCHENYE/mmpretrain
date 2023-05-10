from mmengine.registry import FUNCTIONS


@FUNCTIONS.register_module()
def transformer_encoder_wrap_policy(
    module,
    recurse,
    nonwrapped_numel,
) -> bool:
    from ..backbones.vision_transformer import TransformerEncoderLayer
    if recurse:
        return True  # always recurse
    return isinstance(module, TransformerEncoderLayer)
