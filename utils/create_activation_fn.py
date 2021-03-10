from torch.nn import LeakyReLU, ReLU, ReLU6, PReLU, SELU, ELU, SiLU, GELU
from activations.mish.mish import Mish
from activations.terelu.terelu import TEReLU
from echoAI.Activation.Torch.eswish import Eswish
from echoAI.Activation.Torch.swish import Swish
from echoAI.Activation.Torch.sqnl import SQNL


def create_activation_fn(activation_fn: str, num_features: int):
    activation_fn_dict = {
        "lrelu": lambda: LeakyReLU(0.2, inplace=True),
        "relu": lambda: ReLU(inplace=True),
        "relu6": lambda: ReLU6(inplace=True),
        "prelu": lambda: PReLU(num_features, 0.2),
        "selu": lambda: SELU(inplace=True),
        "elu": lambda: ELU(inplace=True),
        "silu": lambda: SiLU(inplace=True),
        "gelu": lambda: GELU(),
        "mish": lambda: Mish(),
        "swish": lambda: Swish(),
        "eswish": lambda: Eswish(),
        "terelu": lambda: TEReLU()
    }

    return activation_fn_dict[activation_fn]()
