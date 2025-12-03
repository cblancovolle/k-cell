from torch import Tensor
from .linear_agent import LinearAgent

__all__ = ["LinearAgent"]


class Agent:
    def __init__(self, ini_X: Tensor, ini_y: Tensor, **model_kwargs):
        raise NotImplementedError()

    @property
    def current_mem_size(self):
        raise NotImplementedError()

    def predict(self, X_test: Tensor):
        raise NotImplementedError()

    def predict_one(self, x_test: Tensor):
        raise NotImplementedError()

    def spatialization(self, eps=1e-6):
        raise NotImplementedError()

    def learn_one(
        self,
        x_new: Tensor,
        y_new: Tensor,
        update_model: bool,
        update_spatialization: bool,
    ) -> bool:
        raise NotImplementedError()
