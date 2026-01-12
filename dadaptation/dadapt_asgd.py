# filename: dadaptation/dadapt_asgd.py

import torch
import torch.optim
import math
import logging

class DAdaptASGD(torch.optim.Optimizer):
    r"""
    Implements ASGD with D-Adaptation automatic step-sizes.
    Based on the research by Jacob Maimon and Bar Naor.
    
    The learning rate is scaled by 1/d (denominator approach) which showed
    the best convergence properties in experiments.
    """
    def __init__(self, params, 
                 lr=1.0, 
                 lambd=1e-4, 
                 alpha=0.75, 
                 t0=1e6, 
                 weight_decay=0, 
                 log_every=0,
                 d0=1e-6, 
                 growth_rate=float('inf')):

        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr,
                        lambd=lambd, 
                        alpha=alpha, 
                        t0=t0,
                        weight_decay=weight_decay, 
                        k=0,
                        log_every=log_every,
                        numerator_weighted=0.0, 
                        d=d0, 
                        growth_rate=growth_rate)
        
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        group = self.param_groups[0]
        lr = max(group['lr'] for group in self.param_groups)
        
        d = group['d']
        growth_rate = group['growth_rate']
        numerator_weighted = group['numerator_weighted']
        log_every = group['log_every']
        
        lambd = group['lambd']
        alpha = group['alpha']
        t0 = group['t0']
        weight_decay = group['weight_decay']
        k = group['k']

        sk_sq = 0.0
        
        # 1. Initial Gradient Norm Calculation (if k=0)
        if k == 0: 
            g_sq = 0.0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    grad = p.grad.data
                    if weight_decay != 0:
                        grad.add_(p.data, alpha=weight_decay)
                    g_sq += (grad * grad).sum().item()
            group['g0_norm'] = g0_norm = math.sqrt(g_sq)

        g0_norm = group['g0_norm']

        # 2. Dynamic Learning Rate Calculation (lr / d)
        if d == 0: d = 1e-6 
        dlr = lr / d 

        # 3. Update Loop
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad.data
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = torch.zeros_like(p.data)
                    state['s'] = torch.zeros_like(p.data).detach()
                    state['x0'] = torch.clone(p.data).detach()

                if weight_decay != 0:
                    grad.add_(p.data, alpha=weight_decay)

                s = state['s']
                
                if group['lr'] > 0.0:
                    numerator_weighted += dlr * torch.dot(grad.flatten(), s.flatten()).item()
                    s.data.add_(grad, alpha=dlr)
                    sk_sq += (s * s).sum().item()

                # ASGD Logic
                p.data.add_(grad, alpha=-dlr)

                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)

                state['mu'] = 1 / max(1, state['step'] - t0)
                state['step'] += 1

        d_hat = d
        if lr > 0.0 and sk_sq > 0:
            d_hat = 2 * numerator_weighted / math.sqrt(sk_sq)
            d = max(d, min(d_hat, d * growth_rate))

        if log_every > 0 and k % log_every == 0:
             print(f"Step {k}: d={d:.4f}, lr_eff={dlr:.6f}")

        for group in self.param_groups:
            group['numerator_weighted'] = numerator_weighted
            group['d'] = d
            group['k'] = k + 1

        return loss
