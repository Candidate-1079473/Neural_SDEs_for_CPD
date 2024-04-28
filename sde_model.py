"""Neural Stochastic Differential Equation for Change-Point Detection"""
import torch
import torchsde
from torch import nn
from torch.distributions import Normal


class Encoder(nn.Module):
    """Compute representation of the path for the prior"""

    def __init__(self, input_size, hidden_size, output_size):
        super(Encoder, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size)
        self.lin = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        out, _ = self.gru(inp)
        out = self.lin(out)
        return out


class LatentSDE(nn.Module):
    sde_type = "ito"
    noise_type = "diagonal"

    def __init__(self, input_size, context_size, hidden_size, output_size):
        """Initialise neural network"""
        super(LatentSDE, self).__init__()
        self.output_size = output_size
        self.encoder = Encoder(
            input_size=input_size, hidden_size=hidden_size, output_size=context_size
        )
        self.f_net = nn.Sequential(
            nn.Linear(input_size + context_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, input_size),
        )
        self.h_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, hidden_size),
            nn.Softplus(),
            nn.Linear(hidden_size, input_size),
        )
        self.g_nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(1, hidden_size),
                    nn.Softplus(),
                    nn.Linear(hidden_size, 1),
                    nn.Softplus(),
                )
                for _ in range(input_size)
            ]
        )

    def f(self, t, y):
        """Prior drift"""
        ts, context = self.context
        i = min(torch.searchsorted(ts, t, right=True), len(ts) - 1)
        return self.f_net(torch.cat((y, context[i]), dim=1))

    def h(self, t, y):
        """Approximate posterior drift"""
        return self.h_net(y)

    def g(self, t, y):
        """Diagonal diffusion drift"""
        y = torch.split(y, split_size_or_sections=1, dim=-1)
        out = [g_net_i(y_i) for (g_net_i, y_i) in zip(self.g_nets, y)]
        return torch.cat(out, dim=-1)

    def forward(self, xs, ts, dt):
        """'Evaluate' prior and posterior SDEs"""
        # contextualise for prior
        self.context = ts, self.encoder(xs.flip(dims=(0,))).flip(dims=(0,))
        # compute reconstructions for both prior and posterior
        zs, logqp_path = torchsde.sdeint(
            self, xs[0], ts, dt=dt, logqp=True, method="euler"
        )
        zs = zs[..., -self.output_size :]
        # compute distances relevant for the loss
        logqp = logqp_path.sum(dim=0).mean(dim=0)
        xs_dist = Normal(loc=zs, scale=0.01)
        log_pxs = (
            xs_dist.log_prob(xs[..., -self.output_size :]).sum(dim=(0, 2)).mean(dim=0)
        )
        return log_pxs, logqp, zs

    @torch.no_grad()
    def sample(self, ts, z0, dt):
        """Make predictions given initial condition z0"""
        zs = torchsde.sdeint(self, z0, ts, names={"drift": "h"}, dt=dt)
        return zs[..., -self.output_size :]