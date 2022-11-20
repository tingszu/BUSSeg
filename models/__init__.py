
from models.efn.BUSSeg import BUSSeg

def get_model(n_channels, n_classes, task, *args):
    model_zoo = {

        "BUSSeg": BUSSeg(model_name='efficientnet-b2', n_channels=3, n_classes=1),

    }
    return model_zoo[task]

