  # Transit_mdp.py.
"""
MDP формулировка для Transit Network Design Problem
Реализует точную MDP формулировку из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"

State: st = (Rt, rt, extendt)
Actions: shortest paths для расширения маршрута или continue/halt
Reward: только в финальном состоянии Rt = -C(C, Rt)
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional, Set, Union, Any
from dataclasses import dataclass
import random
import logging
from enum import Enum
from copy import deepcopy

from config import config
from cost_functions import TransitCostCalculator

logger = logging.getLogger(__name__)

class ExtendMode(Enum):
    """Режимы MDP: extend или halt"""
    EXTEND = True
    HALT = False

@dataclass
class MDPState:
    """
    Состояние MDP для планирования транзитных сетей
    st = (Rt, rt, extendt) из статьи [[12]]
    """
    completed_routes: List[List[int]]  # Rt - завершенные маршруты.
    current_route: List[int]  # Rt - текущий маршрут в процессе.
    extend_mode: ExtendMode  # Extendt - режим (extend/halt).
    city_graph: nx.Graph  # C - граф города (не изменяется).
    od_matrix: np.ndarray  # Матрица спроса (не изменяется).

    def __post_init__(self):
        """Валидация состояния"""
        if self.completed_routes is None:
            self.completed_routes = []
        if self.current_route is None:
            self.current_route = []

    def is_terminal(self) -> bool:
        """Проверить, является ли состояние терминальным"""
        total_routes = len(self.completed_routes)
        if len(self.current_route) > 0:
            total_routes += 1
        return total_routes >= config.mdp.num_routes

    def get_all_routes(self) -> List[List[int]]:
        """Получить все маршруты (завершенные + текущий если есть)"""
        routes = self.completed_routes.copy()
        if len(self.current_route) > 0:
            routes.append(self.current_route.copy())
        return routes

    def copy(self) -> 'MDPState':
        """Создать копию состояния"""
        return MDPState(
            completed_routes=deepcopy(self.completed_routes),
            current_route=self.current_route.copy(),
            extend_mode=self.extend_mode,
            city_graph=self.city_graph,  # Граф не копируем, он неизменный.
            od_matrix=self.od_matrix  # Матрица не копируется, она неизменна.
        )

class TransitAction:
    """Базовый класс для действий в MDP"""
    pass

@dataclass
class ExtendAction(TransitAction):
    """Действие расширения маршрута кратчайшим путем"""
    path: List[int]  # Кратчайший путь для добавления.
    append_to_end: bool  # Добавить в конец (True) или начало (False) маршрута.

    def __str__(self):
        direction = "end" if self.append_to_end else "start"
        return f"Extend({self.path} to {direction})"

@dataclass
class HaltAction(TransitAction):
    """Действие остановки текущего маршрута"""
    decision: str  # "continue" или "halt".

    def __str__(self):
        return f"Halt({self.decision})"

class TransitMDP:
    """
    MDP для проектирования транзитных сетей
    Реализует формулировку из статьи [[12]]
    """

    def __init__(self,
                 city_graph: nx.Graph,
                 od_matrix: np.ndarray,
                 cost_calculator: Optional[TransitCostCalculator] = None,
                 alpha: float = 0.5):
        """
        Args:
            city_graph: граф дорожной сети города
            od_matrix: матрица Origin-Destination спроса
            cost_calculator: калькулятор стоимости транзитной сети
            alpha: вес пассажирской стоимости [0,1]
        """
        self.city_graph = city_graph
        self.od_matrix = od_matrix
        self.alpha = alpha

        if cost_calculator is None:
            self.cost_calculator = TransitCostCalculator(alpha=alpha)
        else:
            self.cost_calculator = cost_calculator

  # Предвычисляем все кратчайшие пути [[12]].
  # Logger.info("Предвычисление кратчайших путей для MDP...").
        self.shortest_paths = self._compute_shortest_paths()

  # Создаем пространство действий расширения.
        self.extend_actions = self._create_extend_action_space()

  # Logger.info(f"MDP инициализирован: {len(self.city_graph.nodes())} узлов, ".
  # F"{len(self.extend_actions)} возможных расширений").

    def _compute_shortest_paths(self) -> Dict[Tuple[int, int], List[int]]:
        """Предвычислить все кратчайшие пути между парами узлов"""
        try:
  # Получаем все кратчайшие пути (без weight параметра).
            all_paths = dict(nx.all_pairs_shortest_path(self.city_graph))

            shortest_paths = {}
            for source in all_paths:
                for target in all_paths[source]:
                    if source != target:
                        path = all_paths[source][target]
                        shortest_paths[(source, target)] = path

  # Logger.info(f"Предвычислено {len(shortest_paths)} кратчайших путей").
            return shortest_paths

        except Exception as e:
            logger.error(f"Ошибка предвычисления кратчайших путей: {e}")
            return {}

    def _create_extend_action_space(self) -> List[List[int]]:
        """
        Создать пространство действий расширения (SP из статьи)
        Возвращает список всех кратчайших путей длиной ≤ MAX
        """
        extend_actions = []

        for (source, target), path in self.shortest_paths.items():
            path_length = len(path)
            if path_length <= config.mdp.max_route_length:
                extend_actions.append(path)

  # Удаляем дубликаты.
        unique_actions = []
        seen_paths = set()

        for path in extend_actions:
            path_tuple = tuple(path)
            if path_tuple not in seen_paths:
                seen_paths.add(path_tuple)
                unique_actions.append(path)

  # Logger.info(f"Создано {len(unique_actions)} уникальных действий расширения").
        return unique_actions

    def get_initial_state(self) -> MDPState:
        """Получить начальное состояние MDP: s0 = ({}, [], True)"""
        return MDPState(
            completed_routes=[],
            current_route=[],
            extend_mode=ExtendMode.EXTEND,
            city_graph=self.city_graph,
            od_matrix=self.od_matrix
        )

    def get_valid_actions(self, state: MDPState) -> List[TransitAction]:
        """
        Получить валидные действия в данном состоянии
        Реализует логику из статьи [[12]]
        """
        if state.is_terminal():
            return []

        if state.extend_mode == ExtendMode.EXTEND:
            return self._get_extend_actions(state)
        else:
            return self._get_halt_actions(state)

    def _get_extend_actions(self, state: MDPState) -> List[ExtendAction]:
        """
        Получить действия расширения маршрута
        Условия из статьи:
        - если rt = [], то At = {a | a ∈ SP, |a| ≤ MAX}
        - иначе проверяем совместимость с текущим маршрутом
        """
        valid_actions = []
        current_route = state.current_route

        if len(current_route) == 0:
  # Если маршрут пустой, можем начать с любого пути.
            for path in self.extend_actions:
                if len(path) <= config.mdp.max_route_length:
                    valid_actions.append(ExtendAction(path=path, append_to_end=True))
        else:
  # Проверяем совместимость с текущим маршрутом.
            route_start = current_route[0]
            route_end = current_route[-1]
            used_nodes = set(current_route)

            for path in self.extend_actions:
                path_start = path[0]
                path_end = path[-1]
                path_nodes = set(path)

  # Проверяем условия из статьи:.
  # 1. Есть ребро между концом маршрута и началом пути (или наоборот).
  # 2. Нет общих узлов.
  # 3. Длина не превышает лимит.

                can_append_to_end = (
                        self.city_graph.has_edge(route_end, path_start) and
                        len(path_nodes.intersection(used_nodes)) == 0 and
                        len(current_route) + len(path) - 1 <= config.mdp.max_route_length
                )

                can_append_to_start = (
                        self.city_graph.has_edge(route_start, path_end) and
                        len(path_nodes.intersection(used_nodes)) == 0 and
                        len(current_route) + len(path) - 1 <= config.mdp.max_route_length
                )

                if can_append_to_end:
                    valid_actions.append(ExtendAction(path=path, append_to_end=True))

                if can_append_to_start:
                    valid_actions.append(ExtendAction(path=path, append_to_end=False))

        return valid_actions

    def _get_halt_actions(self, state: MDPState) -> List[HaltAction]:
        """
        Получить действия остановки/продолжения маршрута
        Логика из статьи [[12]]:
        - {continue} если |r| < MIN
        - {halt} если |r| = MAX
        - {continue, halt} иначе
        """
        route_length = len(state.current_route)

        if route_length < config.mdp.min_route_length:
            return [HaltAction("continue")]
        elif route_length >= config.mdp.max_route_length:
            return [HaltAction("halt")]
        else:
            return [HaltAction("continue"), HaltAction("halt")]

    def step(self, state: MDPState, action: TransitAction) -> Tuple[MDPState, float, bool]:
        """
        Выполнить шаг в MDP

        Args:
            state: текущее состояние
            action: выбранное действие

        Returns:
            (next_state, reward, done)
        """
        if state.is_terminal():
            return state, 0.0, True

        next_state = state.copy()
        reward = 0.0
        done = False

        if state.extend_mode == ExtendMode.EXTEND:
  # Режим расширения маршрута.
            if isinstance(action, ExtendAction):
                next_state = self._apply_extend_action(next_state, action)
                next_state.extend_mode = ExtendMode.HALT
            else:
                raise ValueError(f"Ожидалось ExtendAction в режиме EXTEND, получено {type(action)}")

        else:
  # Режим halt/continue.
            if isinstance(action, HaltAction):
                next_state, done = self._apply_halt_action(next_state, action)
                if not done:
                    next_state.extend_mode = ExtendMode.EXTEND
            else:
                raise ValueError(f"Ожидалось HaltAction в режиме HALT, получено {type(action)}")

  # Вычисляем награду только в терминальном состоянии [[12]].
        if done or next_state.is_terminal():
            reward = self._compute_terminal_reward(next_state)
            done = True

        return next_state, reward, done

    def _apply_extend_action(self, state: MDPState, action: ExtendAction) -> MDPState:
        """Применить действие расширения маршрута"""
        path = action.path

        if len(state.current_route) == 0:
  # Начинаем новый маршрут.
            state.current_route = path.copy()
        else:
  # Добавляем к существующему маршруту.
            if action.append_to_end:
  # Добавляем в конец (исключаем первый узел пути, чтобы избежать дублирования).
                state.current_route.extend(path[1:])
            else:
  # Добавляем в начало (исключаем последний узел пути).
                state.current_route = path[:-1] + state.current_route

        return state

    def _apply_halt_action(self, state: MDPState, action: HaltAction) -> Tuple[MDPState, bool]:
        """Применить действие остановки/продолжения"""
        if action.decision == "halt":
  # Завершаем текущий маршрут.
            if len(state.current_route) > 0:
                state.completed_routes.append(state.current_route.copy())
                state.current_route = []

  # Проверяем, достигли ли мы нужного количества маршрутов.
            done = len(state.completed_routes) >= config.mdp.num_routes
            return state, done

        elif action.decision == "continue":
  # Продолжаем строить текущий маршрут.
            return state, False

        else:
            raise ValueError(f"Неизвестное решение halt: {action.decision}")

    def _compute_terminal_reward(self, state: MDPState) -> float:
        """
        Вычислить терминальную награду: Rt = -C(C, Rt)
        Награда равна отрицательной стоимости сети
        """
        all_routes = state.get_all_routes()

        if len(all_routes) == 0:
            return -float('inf')  # Штраф за отсутствие маршрутов.

  # Вычисляем стоимость сети.
        cost = self.cost_calculator.calculate_cost(
            self.city_graph, all_routes, self.od_matrix, self.alpha
        )

        return -cost  # Награда = отрицательная стоимость.

    def rollout_episode(self,
                        policy_func: callable,
                        max_steps: int = 1000) -> Tuple[List[MDPState], List[TransitAction], List[float]]:
        """
        Выполнить полный эпизод с использованием политики

        Args:
            policy_func: функция политики (state) -> action
            max_steps: максимальное количество шагов

        Returns:
            (states, actions, rewards) - траектория эпизода
        """
        states = []
        actions = []
        rewards = []

        current_state = self.get_initial_state()
        step_count = 0

        while not current_state.is_terminal() and step_count < max_steps:
            states.append(current_state.copy())

  # Получаем действие от политики.
            valid_actions = self.get_valid_actions(current_state)
            if not valid_actions:
                logger.warning("Нет валидных действий в состоянии")
                break

            action = policy_func(current_state, valid_actions)
            actions.append(action)

  # Выполняем шаг.
            next_state, reward, done = self.step(current_state, action)
            rewards.append(reward)

            current_state = next_state
            step_count += 1

  # Добавляем финальное состояние.
        if current_state.is_terminal():
            states.append(current_state)

        logger.debug(f"Эпизод завершен за {step_count} шагов")
        return states, actions, rewards

    def random_policy(self, state: MDPState, valid_actions: List[TransitAction]) -> TransitAction:
        """Случайная политика для тестирования"""
        return random.choice(valid_actions)

    def validate_state_action(self, state: MDPState, action: TransitAction) -> bool:
        """Проверить валидность пары состояние-действие"""
        valid_actions = self.get_valid_actions(state)

  # Проверяем тип действия.
        action_matches = False
        for valid_action in valid_actions:
            if type(action) == type(valid_action):
                if isinstance(action, ExtendAction):
                    if (action.path == valid_action.path and
                            action.append_to_end == valid_action.append_to_end):
                        action_matches = True
                        break
                elif isinstance(action, HaltAction):
                    if action.decision == valid_action.decision:
                        action_matches = True
                        break

        return action_matches

    def get_state_features(self, state: MDPState) -> Dict[str, Any]:
        """
        Извлечь признаки состояния для нейронной сети

        Returns:
            Словарь с признаками состояния
        """
        all_routes = state.get_all_routes()

  # Базовые признаки.
        features = {
            'num_completed_routes': len(state.completed_routes),
            'current_route_length': len(state.current_route),
            'extend_mode': 1 if state.extend_mode == ExtendMode.EXTEND else 0,
            'total_routes': len(all_routes),
            'is_terminal': 1 if state.is_terminal() else 0
        }

  # Признаки покрытия узлов.
        covered_nodes = set()
        for route in all_routes:
            covered_nodes.update(route)

        features['covered_nodes_count'] = len(covered_nodes)
        features['coverage_ratio'] = len(covered_nodes) / len(self.city_graph.nodes())

  # Признаки текущего маршрута.
        if len(state.current_route) > 0:
            features['current_route_start'] = state.current_route[0]
            features['current_route_end'] = state.current_route[-1]
        else:
            features['current_route_start'] = -1
            features['current_route_end'] = -1

  # Статистики маршрутов.
        if all_routes:
            route_lengths = [len(route) for route in all_routes]
            features['avg_route_length'] = np.mean(route_lengths)
            features['min_route_length'] = min(route_lengths)
            features['max_route_length'] = max(route_lengths)
        else:
            features['avg_route_length'] = 0.0
            features['min_route_length'] = 0
            features['max_route_length'] = 0

        return features

def create_random_episode(city_graph: nx.Graph,
                          od_matrix: np.ndarray,
                          alpha: float = 0.5,
                          max_steps: int = 1000) -> Tuple[List[List[int]], float]:
    """
    Создать случайный эпизод для тестирования

    Returns:
        (routes, cost) - сгенерированные маршруты и их стоимость
    """
    mdp = TransitMDP(city_graph, od_matrix, alpha=alpha)

    states, actions, rewards = mdp.rollout_episode(mdp.random_policy, max_steps)

    if states:
        final_state = states[-1]
        routes = final_state.get_all_routes()

  # Вычисляем финальную стоимость.
        cost = mdp.cost_calculator.calculate_cost(city_graph, routes, od_matrix, alpha)

        return routes, cost
    else:
        return [], float('inf')

if __name__ == "__main__":
  # Демонстрация и тестирование MDP.
    logger.info("Тестирование MDP для транзитного планирования...")

  # Создаем простой тестовый граф.
    G = nx.grid_2d_graph(4, 4)
    G = nx.convert_node_labels_to_integers(G)

  # Добавляем атрибуты времени поездки.
    for u, v in G.edges():
        G[u][v]['travel_time'] = 60.0  # 1 минута на сегмент.

  # Создаем простую OD матрицу.
    n_nodes = len(G.nodes())
    od_matrix = np.random.randint(10, 100, size=(n_nodes, n_nodes))
    np.fill_diagonal(od_matrix, 0)

    logger.info(f"Тестовый граф: {n_nodes} узлов, {len(G.edges())} ребер")

  # Создаем MDP.
    mdp = TransitMDP(G, od_matrix, alpha=0.5)

    logger.info(f"MDP создан: {len(mdp.extend_actions)} возможных расширений")

  # Тестируем начальное состояние.
    initial_state = mdp.get_initial_state()
    logger.info(f"Начальное состояние: режим={initial_state.extend_mode.name}")

  # Тестируем действия.
    valid_actions = mdp.get_valid_actions(initial_state)
    logger.info(f"Валидных действий в начальном состоянии: {len(valid_actions)}")

  # Выполняем несколько случайных эпизодов.
    logger.info("\nТестирование случайных эпизодов:")
    for i in range(3):
        routes, cost = create_random_episode(G, od_matrix, alpha=0.5)
        logger.info(f"Эпизод {i + 1}: {len(routes)} маршрутов, стоимость={cost:.3f}")

  # Выводим маршруты.
        for j, route in enumerate(routes):
            logger.info(f"  Маршрут {j + 1}: {route} (длина {len(route)})")

  # Тестируем валидацию.
    logger.info("\nТестирование валидации состояний...")
    test_state = mdp.get_initial_state()
    test_actions = mdp.get_valid_actions(test_state)

    for action in test_actions[:3]:  # Тестируем первые 3 действия.
        is_valid = mdp.validate_state_action(test_state, action)
        logger.info(f"Действие {action} валидно: {is_valid}")

  # Тестируем признаки состояния.
    logger.info("\nПризнаки начального состояния:")
    features = mdp.get_state_features(initial_state)
    for key, value in features.items():
        logger.info(f"  {key}: {value}")

    logger.info("Тестирование MDP завершено успешно!")
