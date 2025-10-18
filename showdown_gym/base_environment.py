import copy
import logging
from typing import Any, Dict

import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env import AccountConfiguration
from poke_env.battle import AbstractBattle
from poke_env.environment.singles_env import ObsType, SinglesEnv


class BaseShowdownEnv(SinglesEnv):
    """
    Base class for PokeEnv
    """

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            account_configuration1=AccountConfiguration(account_name_one, None),
            account_configuration2=AccountConfiguration(account_name_two, None),
            strict=False,
            log_level=logging.ERROR,
            battle_format=battle_format,
            team=team,
            open_timeout=None,
            ping_interval=None,
            ping_timeout=None,
        )

        observation_size = self._observation_size()

        low = [0] * observation_size
        high = [1] * observation_size
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        act_size = self._get_action_size()
        if act_size is not None:
            self.action_spaces[self.possible_agents[0]] = Discrete(act_size)

        self.n = 1

        self._prior_battle_one: AbstractBattle  # type: ignore
        self._prior_battle_two: AbstractBattle  # type: ignore

    def render(self, mode="human"):
        return np.zeros((100, 100, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        self.n = 1

        response = super().reset(seed, options)

        self._prior_battle_one = copy.deepcopy(self.battle1)
        self._prior_battle_two = copy.deepcopy(self.battle2)

        return response

    def process_action(self, action: np.int64) -> np.int64:
        return action

    def step(self, actions: dict[str, np.int64]) -> tuple[
        dict[str, ObsType],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        self.n += 1
        self._prior_battle_one = copy.deepcopy(self.battle1)  # type: ignore
        self._prior_battle_two = copy.deepcopy(self.battle2)  # type: ignore

        actions[self.agents[0]] = self.process_action(actions[self.agents[0]])

        return super().step(actions)

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        if self.battle1 is not None:
            agent_one = self.possible_agents[0]
            info[agent_one]["win"] = self.battle1.won

        if self.battle2 is not None:
            agent_two = self.possible_agents[1]
            info[agent_two]["win"] = self.battle2.won

        return info

    def _get_prior_battle(self, battle: AbstractBattle) -> AbstractBattle | None:
        prior_battle: AbstractBattle | None = None
        if (
            self.battle1 is not None
            and self.battle1.player_username == battle.player_username
        ):
            prior_battle = self._prior_battle_one
        elif (
            self.battle2 is not None
            and self.battle2.player_username == battle.player_username
        ):
            prior_battle = self._prior_battle_two
        return prior_battle

    def _observation_size(self) -> int:
        raise NotImplementedError(
            "This method should be implemented in subclasses to define the observation spaces."
        )

    def _get_action_size(self) -> int | None:
        raise NotImplementedError(
            "This method should be implemented in subclasses to define the action spaces."
        )
