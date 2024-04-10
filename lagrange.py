from typing import Protocol
import torch.optim as optim
import torch
import torch.nn.functional as F

class Lagrange(Protocol):
    def get_lagrangian_param(self) -> float:
        pass

    def update_lagrangian_multiplier(self, ep_cost_avg: float) -> None:
        pass        
    
class Lagrangian:
    def __init__(
        self, 
        cost_limit_in : float,
        lagrangian_init_in : float,
        lagrangian_lr_in : float,
        lagrangian_upperbound_in : float | None = None
    ) -> None:
        self.lagrangian_upperbound = lagrangian_upperbound_in
        self.cost_limit = cost_limit_in

        lagrangian_init = max(0.0, lagrangian_init_in)
        self.lagrangian_param = torch.tensor(lagrangian_init, requires_grad=True).float()
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_param], lr=lagrangian_lr_in)
        
    def get_lagrangian_param(self) -> float:
        return F.softplus(self.lagrangian_param).detach().item()
    
    def get_lambda_loss(self, ep_cost_avg) -> torch.Tensor:
        cost_deviation = ep_cost_avg - self.cost_limit
        return -self.lagrangian_param * cost_deviation
        
    def update_lagrangian_multiplier(self, ep_cost_avg : float) -> None:
        lambda_loss = self.get_lambda_loss(ep_cost_avg)
        self.lagrangian_optimizer.zero_grad()
        lambda_loss.backward()
        self.lagrangian_optimizer.step()
        self.lagrangian_param.data.clamp_(
            0.0,
            self.lagrangian_upperbound,
        )  # enforce: lambda in [0, inf]
    
class PIDLagrangian:
    def __init__(self):
        ...
        
    def get_lagrangian_param(self) -> float:
        ...
        
    def update(self, ep_cost_avg : float) -> None:
        ...