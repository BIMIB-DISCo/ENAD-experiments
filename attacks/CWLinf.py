import torch
from typing import Callable
from tqdm import tqdm


class LinfHighConf:
    def __init__(
        self,
        model: Callable,
        epsilon: float,
        learning_rate: float,
        num_steps: int,
        misclass_conf_bound: float = 1e4,
    ):

        self.model = model
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.misclass_conf_bound = misclass_conf_bound

    def attack(self, x, y_true):

        assert self.model(x.unsqueeze(dim=0)).argmax() == y_true

        adv = x.clone().to(x.device).requires_grad_(True)
        clamp_adv = x.clone().to(x.device)

        optimizer = torch.optim.Adam([adv], lr=self.learning_rate)
        loss = 0
        delta = 0
        valid_adv = False
        pbar = tqdm(range(self.num_steps), total=self.num_steps, disable=True)
        for i in pbar:

            adv_out = self.model(clamp_adv.unsqueeze(dim=0))

            valid_adv = torch.argmax(adv_out) != y_true

            optimizer.zero_grad()

            loss = self.confidence_loss(adv_out, y_true)

            loss.backward()
            optimizer.step()

            delta = torch.clamp(adv, min=0, max=1) - x
            clamp_delta = torch.clamp(delta, min=-self.epsilon, max=self.epsilon)
            clamp_adv = x + clamp_delta

        if valid_adv:
            return clamp_adv.detach().cpu(), clamp_delta.detach().cpu()
        else:
            return None

    def confidence_loss(self, adv_out: torch.Tensor, y_target_or_true: int):
        y_target_or_true_conf = adv_out[:, y_target_or_true]
        other_mask = torch.arange(adv_out.shape[1]) != y_target_or_true
        misclass_max_conf = torch.maximum(
            torch.max(adv_out[:, other_mask]), -torch.tensor(self.misclass_conf_bound)
        )

        return y_target_or_true_conf - misclass_max_conf
