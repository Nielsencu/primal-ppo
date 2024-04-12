from typing import Protocol
import torch.optim as optim
import torch
import torch.nn.functional as F    
import enum

from alg_parameters import LagrangianParameters

class LagrangianType(int, enum.Enum):
    VANILLA=0
    PID=1

class Lagrange(Protocol):
    def get_lagrangian_param(self) -> float:
        pass

    def update_lagrangian_multiplier(self, ep_cost_avg: float) -> None:
        pass        
    
def get_lagrangian(lagrangian_type : LagrangianType, cost_limit_in) -> Lagrange:
    if lagrangian_type == LagrangianType.VANILLA:
        return Lagrangian(cost_limit_in)
    elif lagrangian_type == LagrangianType.PID:
        return PIDLagrangian(cost_limit_in)
    
class Lagrangian:
    def __init__(
        self, 
        cost_limit_in : float,
    ) -> None:
        self.cost_limit = cost_limit_in

        lagrangian_init = max(0.0, LagrangianParameters.INIT_VALUE)
        self.lagrangian_param = torch.tensor(lagrangian_init, requires_grad=True).float()
        self.lagrangian_optimizer = optim.Adam([self.lagrangian_param], lr=LagrangianParameters.LR)
        
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
            LagrangianParameters.UPPER_BOUND,
        )  # enforce: lambda in [0, inf]
    
class PIDLagrangian:
    def __init__(
        self,
        cost_limit_in : float,
    ):
        self.cost_limit = cost_limit_in
        
        self.i_term : float = max(0.0, LagrangianParameters.INIT_VALUE)
        
        self.lagrangian_param : float = 0.0
        self.delta_moving_avg : float = 0.0
        
        self.cost_moving_avg : float = 0.0
        self.cost_moving_avg_prev : float = 0.0
        
    def get_lagrangian_param(self) -> float:
        return self.lagrangian_param
        
    def update_lagrangian_multiplier(self, ep_cost_avg : float) -> None:
        delta = ep_cost_avg - self.cost_limit
        
        alpha_delta = LagrangianParameters.DELTA_MOVING_AVG_ALPHA
        self.delta_moving_avg *= alpha_delta
        self.delta_moving_avg += (1-alpha_delta) * delta
        
        alpha_cost = LagrangianParameters.COST_MOVING_AVG_ALPHA
        self.cost_moving_avg *= alpha_cost
        self.cost_moving_avg += (1-alpha_cost) * ep_cost_avg
        
        d_term = max(0.0, self.cost_moving_avg - self.cost_moving_avg_prev) 
        self.i_term = max(0.0, self.i_term + delta * LagrangianParameters.KI)
        
        pid_sum = LagrangianParameters.KP * self.delta_moving_avg + self.i_term + LagrangianParameters.KD * d_term
        self.lagrangian_param = max(0.0, pid_sum)
        self.cost_moving_avg_prev = self.cost_moving_avg
        