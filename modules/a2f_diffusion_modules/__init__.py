from .generator import (
    NoiseDecoder,
    ContentW2VEncoder,
)


def get_network(name: str):
    obj = globals().get(name)
    if obj is None:
        raise KeyError("Unknown Network: %s" % name)
    else:
        return obj
