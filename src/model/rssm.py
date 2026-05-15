from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


RSSMState = Dict[str, torch.Tensor]


class RSSM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_mlp = nn.Sequential(
            nn.Linear(cfg.rssm_stoch_dim + cfg.action_dim, cfg.rssm_hidden_dim),
            nn.LayerNorm(cfg.rssm_hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRUCell(cfg.rssm_hidden_dim, cfg.rssm_deter_dim)

        self.prior_mlp = nn.Sequential(
            nn.Linear(cfg.rssm_deter_dim, cfg.rssm_hidden_dim),
            nn.LayerNorm(cfg.rssm_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.rssm_hidden_dim, 2 * cfg.rssm_stoch_dim),
        )
        self.post_mlp = nn.Sequential(
            nn.Linear(cfg.rssm_deter_dim + cfg.fusion_dim, cfg.rssm_hidden_dim),
            nn.LayerNorm(cfg.rssm_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.rssm_hidden_dim, 2 * cfg.rssm_stoch_dim),
        )

    def init_state(self, batch_size: int, device: torch.device) -> RSSMState:
        deter = torch.zeros(batch_size, self.cfg.rssm_deter_dim, device=device)
        stoch = torch.zeros(batch_size, self.cfg.rssm_stoch_dim, device=device)
        mean = torch.zeros(batch_size, self.cfg.rssm_stoch_dim, device=device)
        std = torch.ones(batch_size, self.cfg.rssm_stoch_dim, device=device)
        return {"deter": deter, "stoch": stoch, "mean": mean, "std": std}

    def reset_state_by_done(self, state: RSSMState, done: torch.Tensor) -> RSSMState:
        """Reset RSSM state for samples whose previous step is terminal.

        done can have shape [B] or [B, 1]. A value > 0.5 resets that batch item.
        """
        if done.ndim == 2 and done.size(-1) == 1:
            done = done.squeeze(-1)
        if done.ndim != 1:
            raise ValueError("done must have shape [B] or [B, 1].")
        reset = done.float().gt(0.5).float().view(-1, 1)
        init = self.init_state(done.size(0), done.device)
        return {k: state[k] * (1.0 - reset) + init[k] * reset for k in state.keys()}

    def _stats_to_dist(self, stats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, std_param = torch.chunk(stats, 2, dim=-1)
        std = F.softplus(std_param) + self.cfg.min_std
        return mean, std

    def _sample(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        eps = torch.randn_like(mean)
        return mean + std * eps

    def get_feat(self, state: RSSMState) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def imagine_step(self, prev_state: RSSMState, prev_action: torch.Tensor) -> RSSMState:
        x = torch.cat([prev_state["stoch"], prev_action], dim=-1)
        x = self.input_mlp(x)
        deter = self.gru(x, prev_state["deter"])
        prior_stats = self.prior_mlp(deter)
        mean, std = self._stats_to_dist(prior_stats)
        stoch = self._sample(mean, std)
        return {"deter": deter, "stoch": stoch, "mean": mean, "std": std}

    def obs_step(
        self,
        prev_state: RSSMState,
        prev_action: torch.Tensor,
        obs_embed: torch.Tensor,
    ) -> Tuple[RSSMState, RSSMState]:
        prior = self.imagine_step(prev_state, prev_action)
        post_stats = self.post_mlp(torch.cat([prior["deter"], obs_embed], dim=-1))
        mean, std = self._stats_to_dist(post_stats)
        stoch = self._sample(mean, std)
        post = {"deter": prior["deter"], "stoch": stoch, "mean": mean, "std": std}
        return prior, post

    def observe(
        self,
        obs_embeds: torch.Tensor,
        prev_actions: torch.Tensor,
        start_state: Optional[RSSMState] = None,
        prev_dones: Optional[torch.Tensor] = None,
    ) -> Tuple[RSSMState, RSSMState]:
        if obs_embeds.ndim != 3 or prev_actions.ndim != 3:
            raise ValueError("obs_embeds and prev_actions must have shape [B, T, D].")
        batch_size, seq_len, _ = obs_embeds.shape
        device = obs_embeds.device
        prev_state = start_state or self.init_state(batch_size, device)

        if prev_dones is not None:
            if prev_dones.ndim == 3 and prev_dones.size(-1) == 1:
                prev_dones = prev_dones.squeeze(-1)
            if prev_dones.shape != (batch_size, seq_len):
                raise ValueError("prev_dones must have shape [B, T] or [B, T, 1].")

        priors = {k: [] for k in ["deter", "stoch", "mean", "std"]}
        posts = {k: [] for k in ["deter", "stoch", "mean", "std"]}

        for t in range(seq_len):
            if prev_dones is not None:
                prev_state = self.reset_state_by_done(prev_state, prev_dones[:, t])
            prior, post = self.obs_step(prev_state, prev_actions[:, t], obs_embeds[:, t])
            for key in priors:
                priors[key].append(prior[key])
                posts[key].append(post[key])
            prev_state = post

        priors = {k: torch.stack(v, dim=1) for k, v in priors.items()}
        posts = {k: torch.stack(v, dim=1) for k, v in posts.items()}
        return priors, posts
