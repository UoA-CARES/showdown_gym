import os
from typing import Any, ClassVar, Dict, Iterable, List, Tuple

import numpy as np
from poke_env import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.battle import AbstractBattle
from poke_env.battle.field import Field
from poke_env.battle.move import Move
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.side_condition import SideCondition, STACKABLE_CONDITIONS
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.player.player import Player

from showdown_gym.base_environment import BaseShowdownEnv


class ShowdownEnvironment(BaseShowdownEnv):
    """Pokemon Showdown environment with structured state, reward shaping, and logging.

    The environment exposes a factored numerical state that captures board control,
    hazards, move attributes, and team health so value-based agents can reason about
    random-team battles. It is purposely designed to pair with the `RainbowDQN`
    implementation from the cares_reinforcement_learning library: Rainbow handles the
    discrete move/switch action space, prioritized replay copes with rare but decisive
    transitions, and distributional value estimation stabilises learning under the
    shaped-yet-sparse rewards. Because the observation is already vectorised, image
    based algorithms are unnecessary.
    """

    _TEAM_SIZE: ClassVar[int] = 6
    _TURN_NORMALIZER: ClassVar[float] = 50.0
    _MOVE_POWER_NORMALIZER: ClassVar[float] = 150.0
    _PRIORITY_OFFSET: ClassVar[float] = 7.0
    _PRIORITY_RANGE: ClassVar[float] = 14.0
    _STATUS_LIST: ClassVar[List[Status]] = [
        Status.BRN,
        Status.PAR,
        Status.PSN,
        Status.TOX,
        Status.SLP,
        Status.FRZ,
    ]
    _BOOST_ORDER: ClassVar[List[str]] = [
        "atk",
        "def",
        "spa",
        "spd",
        "spe",
        "accuracy",
        "evasion",
    ]
    _TYPE_LIST: ClassVar[List[PokemonType]] = [
        PokemonType.BUG,
        PokemonType.DARK,
        PokemonType.DRAGON,
        PokemonType.ELECTRIC,
        PokemonType.FAIRY,
        PokemonType.FIGHTING,
        PokemonType.FIRE,
        PokemonType.FLYING,
        PokemonType.GHOST,
        PokemonType.GRASS,
        PokemonType.GROUND,
        PokemonType.ICE,
        PokemonType.NORMAL,
        PokemonType.POISON,
        PokemonType.PSYCHIC,
        PokemonType.ROCK,
        PokemonType.STEEL,
        PokemonType.WATER,
        PokemonType.THREE_QUESTION_MARKS,
        PokemonType.STELLAR,
    ]
    _HAZARD_LIST: ClassVar[List[SideCondition]] = [
        SideCondition.STEALTH_ROCK,
        SideCondition.SPIKES,
        SideCondition.TOXIC_SPIKES,
        SideCondition.STICKY_WEB,
        SideCondition.AURORA_VEIL,
        SideCondition.LIGHT_SCREEN,
        SideCondition.REFLECT,
        SideCondition.SAFEGUARD,
        SideCondition.TAILWIND,
    ]
    _WEATHER_GROUPS: ClassVar[List[set[Weather]]] = [
        {Weather.RAINDANCE, Weather.PRIMORDIALSEA},
        {Weather.SUNNYDAY, Weather.DESOLATELAND},
        {Weather.SANDSTORM},
        {Weather.HAIL, Weather.SNOW, Weather.SNOWSCAPE},
        {Weather.DELTASTREAM},
    ]
    _TERRAIN_FIELDS: ClassVar[List[Field]] = [
        Field.ELECTRIC_TERRAIN,
        Field.GRASSY_TERRAIN,
        Field.MISTY_TERRAIN,
        Field.PSYCHIC_TERRAIN,
    ]

    def __init__(
        self,
        battle_format: str = "gen9randombattle",
        account_name_one: str = "train_one",
        account_name_two: str = "train_two",
        team: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )
        self._reward_tracker: Dict[str, Dict[str, float]] = {}

    def get_additional_info(self) -> Dict[str, Dict[str, Any]]:
        info = super().get_additional_info()

        tracked_battles = {
            self.possible_agents[0]: self.battle1,
            self.possible_agents[1]: self.battle2,
        }
        for agent, battle in tracked_battles.items():
            if battle is None:
                continue
            metrics = self._reward_tracker.get(battle.battle_tag)
            if metrics:
                info[agent].update(metrics)
                info[agent].setdefault("turn", battle.turn)
        return info

    def step(
        self, actions: Dict[str, np.int64]
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        try:
            return super().step(actions)
        except AssertionError as exc:
            if (
                self.battle1 is not None
                and self.battle2 is not None
                and self.battle1.finished
                and self.battle2.finished
            ):
                return self._finalize_finished_battles()
            raise exc

    def _finalize_finished_battles(
        self,
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict[str, Any]],
    ]:
        assert self.battle1 is not None
        assert self.battle2 is not None

        observations = {
            self.possible_agents[0]: self.embed_battle(self.battle1),
            self.possible_agents[1]: self.embed_battle(self.battle2),
        }
        rewards = {
            self.possible_agents[0]: self.calc_reward(self.battle1),
            self.possible_agents[1]: self.calc_reward(self.battle2),
        }
        term1, trunc1 = self.calc_term_trunc(self.battle1)
        term2, trunc2 = self.calc_term_trunc(self.battle2)
        terminated = {self.possible_agents[0]: term1, self.possible_agents[1]: term2}
        truncated = {self.possible_agents[0]: trunc1, self.possible_agents[1]: trunc2}
        info = self.get_additional_info()

        if hasattr(self, "agents"):
            self.agents = []

        tag1 = getattr(self.battle1, "battle_tag", None)
        tag2 = getattr(self.battle2, "battle_tag", None)
        if tag1 is not None:
            self._reward_tracker.pop(tag1, None)
        if tag2 is not None and tag2 != tag1:
            self._reward_tracker.pop(tag2, None)

        return observations, rewards, terminated, truncated, info

    def calc_reward(self, battle: AbstractBattle) -> float:
        prior_battle = self._get_prior_battle(battle)

        our_team = list(battle.team.values())
        opp_team = list(battle.opponent_team.values())
        if prior_battle is not None:
            prior_our_team = list(prior_battle.team.values())
            prior_opp_team = list(prior_battle.opponent_team.values())
        else:
            prior_our_team = our_team
            prior_opp_team = opp_team

        ally_hp_now = self._team_hp_fraction(our_team)
        opp_hp_now = self._team_hp_fraction(opp_team)
        ally_hp_prev = self._team_hp_fraction(prior_our_team)
        opp_hp_prev = self._team_hp_fraction(prior_opp_team)

        ally_fainted_now = self._fainted_fraction(our_team)
        opp_fainted_now = self._fainted_fraction(opp_team)
        ally_fainted_prev = self._fainted_fraction(prior_our_team)
        opp_fainted_prev = self._fainted_fraction(prior_opp_team)

        ally_status_now = self._status_fraction(our_team)
        opp_status_now = self._status_fraction(opp_team)
        ally_status_prev = self._status_fraction(prior_our_team)
        opp_status_prev = self._status_fraction(prior_opp_team)

        damage_component = (opp_hp_prev - opp_hp_now) * 6.0 - (ally_hp_prev - ally_hp_now) * 4.5
        faint_component = (opp_fainted_now - opp_fainted_prev) * 3.5 - (ally_fainted_now - ally_fainted_prev) * 4.5
        status_component = max(0.0, opp_status_now - opp_status_prev) * 0.6
        status_component -= max(0.0, ally_status_now - ally_status_prev) * 0.6

        switch_penalty = 0.0
        if prior_battle is not None:
            previous_active = prior_battle.active_pokemon
            current_active = battle.active_pokemon
            if (
                previous_active is not None
                and current_active is not None
                and previous_active.species is not None
                and current_active.species is not None
                and previous_active.species != current_active.species
                and not prior_battle.force_switch
            ):
                switch_penalty = -0.05

        victory_bonus = 0.0
        if battle.won:
            victory_bonus = 15.0
        elif battle.lost:
            victory_bonus = -15.0

        step_penalty = -0.01

        raw_reward = (
            damage_component
            + faint_component
            + status_component
            + switch_penalty
            + victory_bonus
            + step_penalty
        )

        scaled_reward = raw_reward / 10.0

        metrics: Dict[str, float] = {
            "ally_team_hp": float(ally_hp_now),
            "opp_team_hp": float(opp_hp_now),
            "ally_fainted_frac": float(ally_fainted_now),
            "opp_fainted_frac": float(opp_fainted_now),
            "ally_status_frac": float(ally_status_now),
            "opp_status_frac": float(opp_status_now),
            "damage_component": float(damage_component),
            "faint_component": float(faint_component),
            "status_component": float(status_component),
            "switch_penalty": float(switch_penalty),
            "victory_bonus": float(victory_bonus),
            "step_penalty": float(step_penalty),
            "reward": float(scaled_reward),
            "raw_reward": float(raw_reward),
            "turn": float(battle.turn),
        }
        self._reward_tracker[battle.battle_tag] = metrics

        return float(scaled_reward)

    def _observation_size(self) -> int:
        type_len = len(self._TYPE_LIST)
        status_len = len(self._STATUS_LIST)
        boost_len = len(self._BOOST_ORDER)
        move_block = 7 + type_len
        return (
            1
            + len(self._WEATHER_GROUPS)
            + (len(self._TERRAIN_FIELDS) + 1)
            + (1 + status_len + boost_len + type_len * 2 + 9)
            + (1 + status_len + boost_len + type_len * 2 + 2)
            + self._TEAM_SIZE * 3
            + self._TEAM_SIZE * 3
            + self._TEAM_SIZE
            + len(self._HAZARD_LIST) * 2
            + move_block * 4
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        features: List[float] = []

        turn_feature = min(1.0, battle.turn / self._TURN_NORMALIZER) if battle.turn else 0.0
        features.append(turn_feature)
        features.extend(self._weather_features(battle))
        features.extend(self._field_features(battle))

        features.extend(self._active_pokemon_features(battle))
        features.extend(self._opponent_active_features(battle))
        features.extend(self._team_summary_features(battle.team, assume_full=False))
        features.extend(self._team_summary_features(battle.opponent_team, assume_full=True))
        features.extend(self._switch_features(battle))
        features.extend(self._hazard_features(battle))
        features.extend(self._move_features(battle))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _hp_fraction(pokemon: Pokemon | None) -> float:
        if pokemon is None:
            return 1.0
        if pokemon.fainted:
            return 0.0
        if pokemon.current_hp_fraction is None:
            return 1.0
        return float(pokemon.current_hp_fraction)

    @classmethod
    def _team_hp_fraction(cls, team: Iterable[Pokemon]) -> float:
        mons = list(team)[: cls._TEAM_SIZE]
        total = sum(cls._hp_fraction(mon) for mon in mons)
        if len(mons) < cls._TEAM_SIZE:
            total += cls._TEAM_SIZE - len(mons)
        return total / cls._TEAM_SIZE

    @classmethod
    def _fainted_fraction(cls, team: Iterable[Pokemon]) -> float:
        mons = list(team)[: cls._TEAM_SIZE]
        fainted = sum(1 for mon in mons if mon.fainted)
        return fainted / cls._TEAM_SIZE

    @classmethod
    def _status_fraction(cls, team: Iterable[Pokemon]) -> float:
        mons = list(team)[: cls._TEAM_SIZE]
        statuses = sum(
            1
            for mon in mons
            if mon.status is not None and mon.status is not Status.FNT
        )
        return statuses / cls._TEAM_SIZE

    def _status_features(self, pokemon: Pokemon | None) -> List[float]:
        status = None if pokemon is None else pokemon.status
        return [1.0 if status == s else 0.0 for s in self._STATUS_LIST]

    def _boost_features(self, pokemon: Pokemon | None) -> List[float]:
        if pokemon is None:
            return [0.5] * len(self._BOOST_ORDER)
        return [
            ((pokemon.boosts.get(stat, 0) + 6) / 12.0)
            for stat in self._BOOST_ORDER
        ]

    def _type_one_hot(self, pokemon_type: PokemonType | None) -> List[float]:
        return [1.0 if pokemon_type == t else 0.0 for t in self._TYPE_LIST]

    def _pokemon_type_features(self, pokemon: Pokemon | None) -> List[float]:
        if pokemon is None:
            return [0.0] * (len(self._TYPE_LIST) * 2)
        pokemon_types = pokemon.types
        primary = pokemon_types[0] if len(pokemon_types) > 0 else None
        secondary = pokemon_types[1] if len(pokemon_types) > 1 else None
        features: List[float] = []
        features.extend(self._type_one_hot(primary))
        features.extend(self._type_one_hot(secondary))
        return features

    def _active_pokemon_features(self, battle: AbstractBattle) -> List[float]:
        active = battle.active_pokemon
        features: List[float] = [self._hp_fraction(active)]
        features.extend(self._status_features(active))
        features.extend(self._boost_features(active))
        features.extend(self._pokemon_type_features(active))
        features.append(1.0 if active and active.is_dynamaxed else 0.0)
        features.append(1.0 if active and active.is_terastallized else 0.0)
        features.append(1.0 if battle.trapped else 0.0)
        features.append(1.0 if battle.maybe_trapped else 0.0)
        features.append(1.0 if battle.force_switch else 0.0)
        features.append(1.0 if battle.can_mega_evolve else 0.0)
        features.append(1.0 if battle.can_z_move else 0.0)
        features.append(1.0 if battle.can_dynamax else 0.0)
        features.append(1.0 if battle.can_tera else 0.0)
        return features

    def _opponent_active_features(self, battle: AbstractBattle) -> List[float]:
        opponent = battle.opponent_active_pokemon
        features: List[float] = [self._hp_fraction(opponent)]
        features.extend(self._status_features(opponent))
        features.extend(self._boost_features(opponent))
        features.extend(self._pokemon_type_features(opponent))
        features.append(1.0 if opponent and opponent.is_dynamaxed else 0.0)
        features.append(1.0 if opponent and opponent.is_terastallized else 0.0)
        return features

    def _team_summary_features(self, team: Dict[str, Pokemon], assume_full: bool) -> List[float]:
        mons = list(team.values())
        padded = self._pad_pokemon_list(mons)
        features: List[float] = []
        for mon in padded:
            if mon is None:
                hp = 1.0 if assume_full else 0.0
                status_flag = 0.0
                fainted = 0.0
            else:
                hp = self._hp_fraction(mon)
                status_flag = 1.0 if mon.status not in (None, Status.FNT) else 0.0
                fainted = 1.0 if mon.fainted else 0.0
            features.extend([hp, status_flag, fainted])
        return features

    def _pad_pokemon_list(self, mons: Iterable[Pokemon]) -> List[Pokemon | None]:
        padded: List[Pokemon | None] = list(mons)[: self._TEAM_SIZE]
        if len(padded) < self._TEAM_SIZE:
            padded.extend([None] * (self._TEAM_SIZE - len(padded)))
        return padded

    def _switch_features(self, battle: AbstractBattle) -> List[float]:
        team_slots = self._pad_pokemon_list(battle.team.values())
        available = list(battle.available_switches)
        features: List[float] = []
        for mon in team_slots:
            if mon is None:
                features.append(0.0)
            else:
                features.append(1.0 if any(mon is candidate for candidate in available) else 0.0)
        return features

    def _weather_features(self, battle: AbstractBattle) -> List[float]:
        weather_state = battle.weather
        active_weather = Weather.UNKNOWN
        for weather in weather_state.keys():
            active_weather = weather
            break
        features: List[float] = []
        for group in self._WEATHER_GROUPS:
            features.append(1.0 if active_weather in group else 0.0)
        return features

    def _field_features(self, battle: AbstractBattle) -> List[float]:
        active_fields = set(battle.fields.keys())
        features: List[float] = []
        for field in self._TERRAIN_FIELDS:
            features.append(1.0 if field in active_fields else 0.0)
        features.append(1.0 if Field.TRICK_ROOM in active_fields else 0.0)
        return features

    def _hazard_features(self, battle: AbstractBattle) -> List[float]:
        return self._encode_hazards(battle.side_conditions) + self._encode_hazards(
            battle.opponent_side_conditions
        )

    def _encode_hazards(self, conditions: Dict[SideCondition, int]) -> List[float]:
        features: List[float] = []
        for hazard in self._HAZARD_LIST:
            value = conditions.get(hazard)
            if hazard in STACKABLE_CONDITIONS:
                max_stack = STACKABLE_CONDITIONS[hazard]
                normalised = (value or 0) / max_stack if max_stack else 0.0
            else:
                normalised = 1.0 if value is not None else 0.0
            features.append(float(normalised))
        return features

    def _move_features(self, battle: AbstractBattle) -> List[float]:
        active = battle.active_pokemon
        if active is None:
            move_slots: List[Move | None] = []
        else:
            move_slots = list(active.moves.values())
        move_slots = self._pad_moves(move_slots)
        available_ids = {move.id for move in battle.available_moves}
        opponent = battle.opponent_active_pokemon

        features: List[float] = []
        for move in move_slots:
            if move is None:
                features.extend([0.0] * (7 + len(self._TYPE_LIST)))
                continue
            is_available = 1.0 if move.id in available_ids else 0.0
            base_power = move.base_power or 0
            base_power = min(base_power, self._MOVE_POWER_NORMALIZER)
            power_feature = base_power / self._MOVE_POWER_NORMALIZER

            if move.accuracy is None:
                accuracy_feature = 1.0
            else:
                accuracy_feature = float(move.accuracy) / 100.0

            if move.max_pp:
                current_pp = move.current_pp if move.current_pp is not None else move.max_pp
                pp_feature = max(0.0, min(1.0, current_pp / move.max_pp))
            else:
                pp_feature = 1.0

            if opponent is not None:
                effectiveness = opponent.damage_multiplier(move)
            else:
                effectiveness = 1.0
            effectiveness_feature = min(effectiveness, 4.0) / 4.0

            status_flag = 1.0 if move.category == MoveCategory.STATUS else 0.0
            priority = (move.priority + self._PRIORITY_OFFSET) / self._PRIORITY_RANGE
            priority_feature = max(0.0, min(1.0, priority))

            features.extend(
                [
                    is_available,
                    power_feature,
                    accuracy_feature,
                    pp_feature,
                    effectiveness_feature,
                    status_flag,
                    priority_feature,
                ]
            )
            features.extend(self._type_one_hot(move.type))
        return features

    def _pad_moves(self, moves: Iterable[Move | None]) -> List[Move | None]:
        padded: List[Move | None] = list(moves)[:4]
        if len(padded) < 4:
            padded.extend([None] * (4 - len(padded)))
        return padded########################################
# DO NOT EDIT THE CODE BELOW THIS LINE #
########################################


class SingleShowdownWrapper(SingleAgentWrapper):
    """
    A wrapper class for the PokeEnvironment that simplifies the setup of single-agent
    reinforcement learning tasks in a Pokémon battle environment.

    This class initializes the environment with a specified battle format, opponent type,
    and evaluation mode. It also handles the creation of opponent players and account names
    for the environment.

    Do NOT edit this class!

    Attributes:
        battle_format (str): The format of the Pokémon battle (e.g., "gen9randombattle").
        opponent_type (str): The type of opponent player to use ("simple", "max", "random").
        evaluation (bool): Whether the environment is in evaluation mode.
    Raises:
        ValueError: If an unknown opponent type is provided.
    """

    def __init__(
        self,
        team_type: str = "random",
        opponent_type: str = "random",
        evaluation: bool = False,
    ):
        opponent: Player
        if opponent_type == "simple":
            opponent = SimpleHeuristicsPlayer()
        elif opponent_type == "max":
            opponent = MaxBasePowerPlayer()
        elif opponent_type == "random":
            opponent = RandomPlayer()
        else:
            raise ValueError(f"Unknown opponent type: {opponent_type}")

        account_name_one: str = "train_one" if not evaluation else "eval_one"
        account_name_two: str = "train_two" if not evaluation else "eval_two"

        account_name_one = f"{account_name_one}_{opponent_type}"
        account_name_two = f"{account_name_two}_{opponent_type}"

        team = self._load_team(team_type)

        battle_fomat = "gen9randombattle" if team is None else "gen9ubers"

        primary_env = ShowdownEnvironment(
            battle_format=battle_fomat,
            account_name_one=account_name_one,
            account_name_two=account_name_two,
            team=team,
        )

        super().__init__(env=primary_env, opponent=opponent)

    def _load_team(self, team_type: str) -> str | None:
        bot_teams_folders = os.path.join(os.path.dirname(__file__), "teams")

        bot_teams = {}

        for team_file in os.listdir(bot_teams_folders):
            if team_file.endswith(".txt"):
                with open(
                    os.path.join(bot_teams_folders, team_file), "r", encoding="utf-8"
                ) as file:
                    bot_teams[team_file[:-4]] = file.read()

        if team_type in bot_teams:
            return bot_teams[team_type]

        return None





