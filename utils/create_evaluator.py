from facenet_pytorch import InceptionResnetV1
from torchvision.models import inception_v3


def create_evaluator(model_name: str):
    model_dict = {
        "default": (lambda: (inception_v3(pretrained=True, aux_logits=False).eval(), 299, 1000)),
        "vggface2": (lambda: (InceptionResnetV1(pretrained="vggface2", classify=True).eval(), 160, 8631)),
        "casia-webface": (lambda: (InceptionResnetV1(pretrained="casia-webface", classify=True).eval(), 160, 10575)),
    }

    if model_name in model_dict.keys():
        return model_dict[model_name]()
    else:
        return model_dict["default"]()
