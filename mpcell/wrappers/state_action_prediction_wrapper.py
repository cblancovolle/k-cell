import torch
from cell.trainers.online_trainer import OnlineTrainer
from numpy import ndarray


class StateActionPredictorWrapper:
    def __init__(self, trainer: OnlineTrainer):
        self.trainer = trainer

    def predict_one(self, state: ndarray, action: ndarray):
        assert len(state.shape) == 1
        assert len(action.shape) == 1
        state = torch.as_tensor(state, dtype=self.trainer.dtype).view(-1)
        action = torch.as_tensor(action, dtype=self.trainer.dtype).view(-1)
        x_test = torch.hstack((state, action))
        return self.trainer.predict_one(x_test).view(-1).numpy()

    def predict_trajectory(self, state_ini: ndarray, actions: ndarray):
        assert len(actions.shape) == 2
        assert len(state_ini.shape) == 1

        horizon, action_dim = actions.shape
        state_dim = state_ini.shape[0]

        current_state = torch.as_tensor(state_ini, dtype=self.trainer.dtype).view(
            -1, state_dim
        )
        actions = torch.as_tensor(actions, dtype=self.trainer.dtype).view(
            -1, action_dim
        )

        states = [current_state.clone()]
        for i in range(horizon):
            x_test = torch.hstack((current_state.view(-1), actions[i]))
            current_state = self.trainer.predict_one(x_test)
            states += [current_state.clone()]
        return torch.vstack(states).numpy()
