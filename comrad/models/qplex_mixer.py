"""
https://arxiv.org/abs/2008.01062
Ref: https://github.com/wjh720/QPLEX

"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DMAQ_SI_Weight(nn.Module):
    """
    Non-monotonic adv weights w_i = \sum_k |key_k(s)| * \sigma(agents_k(s)) * \sigma(action_k(s,a))
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        n_actions: int,
        num_kernel: int = 4,
        adv_hypernet_embed: int = 64,
        adv_hypernet_layers: int = 1
        ):
        super().__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.action_dim = n_agents * n_actions
        self.state_action_dim = self.state_dim + self.action_dim
        self.num_kernel = num_kernel

        self.key_extractors = nn.ModuleList()
        self.agents_extractors = nn.ModuleList()
        self.action_extractors = nn.ModuleList()

        for _ in range(self.num_kernel):
            if adv_hypernet_layers == 1:
                self.key_extractors.append(nn.Linear(self.state_dim, 1))
                self.agents_extractors.append(nn.Linear(self.state_dim, self.n_agents))
                self.action_extractors.append(nn.Linear(self.state_action_dim, self.n_agents))
            elif adv_hypernet_layers == 2:
                self.key_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, 1),
                ))
                self.agents_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
                self.action_extractors.append(nn.Sequential(
                    nn.Linear(self.state_action_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
            elif adv_hypernet_layers == 3:
                self.key_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, 1),
                ))
                self.agents_extractors.append(nn.Sequential(
                    nn.Linear(self.state_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
                self.action_extractors.append(nn.Sequential(
                    nn.Linear(self.state_action_dim, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, adv_hypernet_embed), nn.ReLU(),
                    nn.Linear(adv_hypernet_embed, self.n_agents),
                ))
            else:
                raise ValueError(f"adv_hypernet_layers must be 1, 2, or 3, got {adv_hypernet_layers}")

    def forward(self, states: Tensor, actions: Tensor) -> Tensor:
        """
        :param states: [B, state_dim]
        :param actions: [B, n_agents * n_actions] (flattened one-hot)
        :returns: [B, n_agents] advantage weights
        """
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        data = torch.cat([states, actions], dim=1)

        all_head_key = [k_ext(states) for k_ext in self.key_extractors]
        all_head_agents = [a_ext(states) for a_ext in self.agents_extractors]
        all_head_action = [act_ext(data) for act_ext in self.action_extractors]

        head_attend_weights = []
        for curr_head_key, curr_head_agents, curr_head_action in zip(
            all_head_key, all_head_agents, all_head_action
        ):
            x_key = torch.abs(curr_head_key).repeat(1, self.n_agents) + 1e-10
            x_agents = torch.sigmoid(curr_head_agents)
            x_action = torch.sigmoid(curr_head_action)
            weights = x_key * x_agents * x_action
            head_attend_weights.append(weights)

        head_attend = torch.stack(head_attend_weights, dim=1)
        head_attend = head_attend.view(-1, self.num_kernel, self.n_agents)
        head_attend = torch.sum(head_attend, dim=1)

        return head_attend


class DMAQer(nn.Module):
    """
    Q_tot = V_tot + A_tot
    """

    def __init__(self, cfg, n_agents: int, state_dim: int, n_actions: int):
        super().__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.action_dim = n_agents * n_actions

        hypernet_embed = getattr(cfg, 'qplex_hypernet_embed', 64)
        self.is_minus_one = getattr(cfg, 'qplex_is_minus_one', True)
        self.weighted_head = getattr(cfg, 'qplex_weighted_head', True)  # default True for dmaq

        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents),
        )
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, self.n_agents),
        )

        num_kernel = getattr(cfg, 'qplex_num_kernel', 4)
        adv_hypernet_embed = getattr(cfg, 'qplex_adv_hypernet_embed', 64)
        adv_hypernet_layers = getattr(cfg, 'qplex_adv_hypernet_layers', 3)  # default 3 for dmaq
        self.si_weight = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=num_kernel,
            adv_hypernet_embed=adv_hypernet_embed,
            adv_hypernet_layers=adv_hypernet_layers,
        )

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        return torch.sum(agent_qs, dim=-1)

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.is_minus_one:
            adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions = None, max_q_i = None, is_v = False):
        if is_v:
            return self.calc_v(agent_qs)
        else:
            return self.calc_adv(agent_qs, states, actions, max_q_i)

    def forward(self, agent_qs, states, actions = None, max_q_i = None, is_v = False):
        """
        :param agent_qs: [B, N]
        :param states: [B, state_dim]
        :param actions: [B, N * n_actions] flattened one-hot (needed for A_tot)
        :param max_q_i: [B, N] per-agent max Q (needed for A_tot)
        :param is_v: True for V_tot, False for A_tot
        :returns: (q_component [B, 1, 1], regs [])
        """
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)

        if self.weighted_head:
            if is_v:
                w_final = self.hyper_w_final(states)
                w_final = torch.abs(w_final)
                w_final = w_final.view(-1, self.n_agents) + 1e-10
                v = self.V(states)
                v = v.view(-1, self.n_agents)
            else:
                # A_tot detaches the transformed advantage immediately, so keep this
                # transform out of autograd to avoid needless activation retention.
                with torch.no_grad():
                    w_final = self.hyper_w_final(states)
                    w_final = torch.abs(w_final)
                    w_final = w_final.view(-1, self.n_agents) + 1e-10
                    v = self.V(states)
                    v = v.view(-1, self.n_agents)
            agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            if self.weighted_head:
                max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        return v_tot, []


class Qatten_Weight(nn.Module):
    """
    Q-attention value weight network.
    Multi-head scaled dot-product attention for computing V_tot weights.
    """

    def __init__(
        self,
        n_agents: int,
        state_dim: int,
        n_actions: int,
        unit_dim: int,
        embed_dim = 32,
        hypernet_embed = 64,
        n_head = 4,
        attend_reg_coef = 0.001,
        weighted_head = False,
        nonlinear = False,
        state_bias = False # Default False to prevent V(s) from absorbing TD signal, only True for many agents
        ):
        super().__init__()

        self.n_agents = n_agents
        self.state_dim = state_dim
        self.unit_dim = unit_dim
        self.n_actions = n_actions
        self.n_head = n_head
        self.embed_dim = embed_dim
        self.attend_reg_coef = attend_reg_coef
        self.weighted_head = weighted_head
        self.nonlinear = nonlinear
        self.state_bias = state_bias

        self.key_extractors = nn.ModuleList()
        self.selector_extractors = nn.ModuleList()

        for _ in range(self.n_head):
            selector_nn = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim, bias=False),
            )
            self.selector_extractors.append(selector_nn)
            if self.nonlinear:
                self.key_extractors.append(nn.Linear(self.unit_dim + 1, self.embed_dim, bias=False))
            else:
                self.key_extractors.append(nn.Linear(self.unit_dim, self.embed_dim, bias=False))

        if self.weighted_head:
            self.hyper_w_head = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.n_head),
            )

        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs: Tensor, states: Tensor, actions: Tensor = None
                ) -> Tuple[Tensor, Tensor, Tensor, list]:
        """
        :param agent_qs: [B, N]
        :param states: [B, state_dim]
        :param actions: [B, N * n_actions] (unused unless mask_dead)
        :returns: (head_attend [B, N], v [B, 1], attend_mag_regs, head_entropies)
        """
        states = states.reshape(-1, self.state_dim)

        # Extract per agent features from state
        unit_states = states[:, :self.unit_dim * self.n_agents]
        unit_states = unit_states.reshape(-1, self.n_agents, self.unit_dim)
        unit_states = unit_states.permute(1, 0, 2) # [N, B, unit_dim]

        agent_qs = agent_qs.view(-1, 1, self.n_agents) # [B, 1, N]

        if self.nonlinear:
            unit_states = torch.cat((unit_states, agent_qs.permute(2, 0, 1)), dim=2)

        all_head_selectors = [sel_ext(states) for sel_ext in self.selector_extractors]
        all_head_keys = [[k_ext(enc) for enc in unit_states] for k_ext in self.key_extractors]

        head_attend_logits = []
        head_attend_weights = []
        for curr_head_keys, curr_head_selector in zip(all_head_keys, all_head_selectors):
            attend_logits = torch.matmul(
                curr_head_selector.view(-1, 1, self.embed_dim),
                torch.stack(curr_head_keys).permute(1, 2, 0),
            ) # [B, 1, N]
            scaled_attend_logits = attend_logits / np.sqrt(self.embed_dim)
            attend_weights = F.softmax(scaled_attend_logits, dim=2)

            head_attend_logits.append(attend_logits)
            head_attend_weights.append(attend_weights)

        head_attend = torch.stack(head_attend_weights, dim=1) # [B, n_head, 1, N]
        head_attend = head_attend.view(-1, self.n_head, self.n_agents)

        v = self.V(states).view(-1, 1)

        if self.weighted_head:
            w_head = torch.abs(self.hyper_w_head(states))
            w_head = w_head.view(-1, self.n_head, 1).repeat(1, 1, self.n_agents)
            head_attend *= w_head

        head_attend = torch.sum(head_attend, dim=1)

        if not self.state_bias:
            v = v * 0. # this is float

        attend_mag_regs = self.attend_reg_coef * sum((logit ** 2).mean() for logit in head_attend_logits)
        head_entropies = [(-((probs + 1e-8).log() * probs).squeeze(1).sum(dim=1).mean()) for probs in head_attend_weights]

        return head_attend, v, attend_mag_regs, head_entropies


class DMAQ_QattenMixer(nn.Module):
    """
    Q-attention for V_tot weights
    SI-attention for A_tot weights
    """

    def __init__(self, cfg, n_agents: int, state_dim: int, n_actions: int, unit_dim: int):
        super().__init__()

        self.n_agents = n_agents
        self.n_actions = n_actions
        self.state_dim = state_dim
        self.action_dim = n_agents * n_actions

        self.is_minus_one = getattr(cfg, 'qplex_is_minus_one', True)

        embed_dim = getattr(cfg, 'qplex_embed_dim', 32)
        hypernet_embed = getattr(cfg, 'qplex_hypernet_embed', 64)
        n_head = getattr(cfg, 'qplex_n_head', 4)
        attend_reg_coef = getattr(cfg, 'qplex_attend_reg_coef', 0.001)
        weighted_head = getattr(cfg, 'qplex_weighted_head', False) # default False for qatten
        nonlinear = getattr(cfg, 'qplex_nonlinear', False)
        state_bias = getattr(cfg, 'qplex_state_bias', False) # Default False to prevent V(s) gradient starvation, as explained above

        self.attention_weight = Qatten_Weight(
            n_agents, state_dim, n_actions, unit_dim,
            embed_dim=embed_dim, hypernet_embed=hypernet_embed,
            n_head=n_head, attend_reg_coef=attend_reg_coef,
            weighted_head=weighted_head, nonlinear=nonlinear,
            state_bias=state_bias,
        )

        num_kernel = getattr(cfg, 'qplex_num_kernel', 4)
        adv_hypernet_embed = getattr(cfg, 'qplex_adv_hypernet_embed', 64)
        adv_hypernet_layers = getattr(cfg, 'qplex_adv_hypernet_layers', 1) # default 1 for qatten
        self.si_weight = DMAQ_SI_Weight(
            n_agents, state_dim, n_actions,
            num_kernel=num_kernel,
            adv_hypernet_embed=adv_hypernet_embed,
            adv_hypernet_layers=adv_hypernet_layers,
        )

    def calc_v(self, agent_qs):
        agent_qs = agent_qs.view(-1, self.n_agents)
        return torch.sum(agent_qs, dim=-1)

    def calc_adv(self, agent_qs, states, actions, max_q_i):
        states = states.reshape(-1, self.state_dim)
        actions = actions.reshape(-1, self.action_dim)
        agent_qs = agent_qs.view(-1, self.n_agents)
        max_q_i = max_q_i.view(-1, self.n_agents)

        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()

        adv_w_final = self.si_weight(states, actions)
        adv_w_final = adv_w_final.view(-1, self.n_agents)

        if self.is_minus_one:
            adv_tot = torch.sum(adv_q * (adv_w_final - 1.), dim=1)
        else:
            adv_tot = torch.sum(adv_q * adv_w_final, dim=1)
        return adv_tot

    def calc(self, agent_qs, states, actions = None, max_q_i = None, is_v = False):
        if is_v:
            return self.calc_v(agent_qs)
        else:
            return self.calc_adv(agent_qs, states, actions, max_q_i)

    def forward(self, agent_qs, states, actions = None, max_q_i = None, is_v = False):
        """
        :param agent_qs: [B, N]
        :param states: [B, state_dim]
        :param actions: [B, N * n_actions] flattened one-hot (needed for A_tot)
        :param max_q_i: [B, N] per-agent max Q (needed for A_tot)
        :param is_v: True for V_tot, False for A_tot
        :returns: (q_component [B, 1, 1], regs List[Tensor]): Tuple[Tensor, List[Tensor]]
        """
        bs = agent_qs.size(0)

        if is_v:
            w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        else:
            # A_tot only uses the transformed advantage as a detached scalar weight,
            # so the qatten path does not need its own backward graph here.
            with torch.no_grad():
                w_final, v, attend_mag_regs, head_entropies = self.attention_weight(agent_qs, states, actions)
        w_final = w_final.view(-1, self.n_agents) + 1e-10
        v = v.view(-1, 1).repeat(1, self.n_agents)
        v = v / self.n_agents

        agent_qs = agent_qs.view(-1, self.n_agents)
        agent_qs = w_final * agent_qs + v
        if not is_v:
            max_q_i = max_q_i.view(-1, self.n_agents)
            max_q_i = w_final * max_q_i + v

        y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
        v_tot = y.view(bs, -1, 1)

        if is_v:
            # Keep detached scalars only, otherwise logging can retain the full autograd
            # Entropies are computed during forward pass of mixer, so this must store detached tensor
            # Else backward() frees the graph buffers then access _last_head_entropies after backward() which is invalid
            # .cpu() is not necessary since .item() is called after, but i will just keep it
            self._last_head_entropies = [entropy.detach().cpu() for entropy in head_entropies]

        return v_tot, [attend_mag_regs]
