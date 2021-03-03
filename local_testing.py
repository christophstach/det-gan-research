from models import MsgDiscriminator



disc = MsgDiscriminator(
    msg=True,
    pack=1,
    image_size=64,
    unary=False,
    max_filters=0,
    min_filters=0,
    activation_fn='lrelu',
    normalization='batch',
    image_channels=3,
    spectral_normalization=False,
    depth=4,
)


print(disc)