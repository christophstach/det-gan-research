import torch
from typing import Dict, Union, Sequence

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]
