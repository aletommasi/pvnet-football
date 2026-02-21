import torch
import torch.nn as nn

class WeightedMultiTaskBCE(nn.Module):
    """
    Multi-task BCEWithLogitsLoss con pos_weight separati per shot e goal.
    y shape: (N, 2)  -> [shot, goal]
    """
    def __init__(self, pos_weight_shot: float, pos_weight_goal: float, task_weights=(1.0, 1.0)):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor([pos_weight_shot, pos_weight_goal], dtype=torch.float32))
        self.task_weights = task_weights

    def forward(self, logits, y):
        # logits: (N,2), y: (N,2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, y, reduction="none", pos_weight=self.pos_weight
        )
        # loss per task
        loss_shot = loss[:, 0].mean()
        loss_goal = loss[:, 1].mean()
        return self.task_weights[0] * loss_shot + self.task_weights[1] * loss_goal
