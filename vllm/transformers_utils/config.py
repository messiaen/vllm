from transformers import AutoConfig, PretrainedConfig

from vllm.transformers_utils.configs import *  # pylint: disable=wildcard-import

_CONFIG_REGISTRY = {
    "mpt": MPTConfig,
}

_CONFIG_NVIDIA = [
    "nvgpt",
]

def get_config(model: str) -> PretrainedConfig:
    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    if config.model_type in _CONFIG_REGISTRY:
        config_class = _CONFIG_REGISTRY[config.model_type]
        config = config_class.from_pretrained(model)

    if config.model_type in _CONFIG_NVIDIA:
        config.num_hidden_layers = config.num_layers
    return config
