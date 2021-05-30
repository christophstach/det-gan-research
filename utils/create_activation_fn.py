from echoAI.Activation.Torch.eswish import Eswish
from echoAI.Activation.Torch.swish import Swish
from torch.nn import LeakyReLU, ReLU, ReLU6, PReLU, SELU, ELU, SiLU, GELU, Tanh

from activations.mish.mish import Mish
from activations.terelu.terelu import TEReLU
from layers.tprelu import TPReLU


def create_activation_fn(activation_fn: str, num_features: int):
    activation_fn_dict = {
        "lrelu": lambda: LeakyReLU(0.2, inplace=False),
        "relu": lambda: ReLU(inplace=False),
        "relu6": lambda: ReLU6(inplace=False),
        "prelu": lambda: PReLU(num_features, 0.2),
        "selu": lambda: SELU(inplace=False),
        "elu": lambda: ELU(inplace=False),
        "silu": lambda: SiLU(inplace=False),
        "gelu": lambda: GELU(),
        "mish": lambda: Mish(),
        "swish": lambda: Swish(),
        "eswish": lambda: Eswish(),
        "terelu": lambda: TEReLU(),
        "tprelu": lambda: TPReLU(num_features, 0.2),
        "tanh": lambda: Tanh()
    }

    return activation_fn_dict[activation_fn]()
