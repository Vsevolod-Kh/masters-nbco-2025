  # Neural_bco.py.
"""
Neural Bee Colony Optimization для Transit Network Design
Реализует гибридный алгоритм из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"

Ключевая идея: замена Type-1 пчел на "neural bees", которые используют обученную нейронную политику
для замены целых маршрутов, сохраняя Type-2 пчел для разнообразия поиска.

Улучшения: до 20% по сравнению с чистой нейронной сетью, до 53% по сравнению с классическим BCO
"""

import numpy as np
import networkx as nx
import random
import time
import logging
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass
import pickle
import torch
import os

from config import config
from cost_functions import TransitCostCalculator, CostComponents
from bco_algorithm import BeeAgent, BCOSolution, BCOStatistics, Type2Bee, BeeColonyOptimization
from neural_planner import NeuralPlannerTrainer, NeuralPlanner
from transit_mdp import TransitMDP, MDPState, ExtendMode
from utils import RouteUtils

logger = logging.getLogger(__name__)

@dataclass
class NeuralBCOStatistics(BCOStatistics):
    """Расширенная статистика для Neural BCO"""
    neural_bee_improvements: int = 0
    neural_bee_calls: int = 0
    neural_policy_time: float = 0.0
    route_replacement_success_rate: float = 0.0
    diversity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict.update({
            'neural_bee_improvements': self.neural_bee_improvements,
            'neural_bee_calls': self.neural_bee_calls,
            'neural_policy_time': self.neural_policy_time,
            'route_replacement_success_rate': self.route_replacement_success_rate,
            'diversity_score': self.diversity_score
        })
        return base_dict

class NeuralBee(BeeAgent):
    """
    Neural Bee: заменяет Type-1 пчел в BCO алгоритме
    Использует обученную нейронную политику для замены целых маршрутов
    """

    def __init__(self,
                 bee_id: int,
                 city_graph: nx.Graph,
                 cost_calculator: TransitCostCalculator,
                 od_matrix: np.ndarray,
                 neural_planner: NeuralPlanner,
                 alpha: float = 0.5):
        super().__init__(bee_id, city_graph, cost_calculator, od_matrix)

        self.neural_planner = neural_planner
        self.alpha = alpha
        self.type_name = "Neural"

  # Статистика работы neural bee.
        self.successful_replacements = 0
        self.total_replacement_attempts = 0
        self.policy_call_time = 0.0

        logger.debug(f"Neural Bee {bee_id} инициализирована с α={alpha}")

    def explore(self, current_routes: List[List[int]], num_cycles: int = 2) -> BCOSolution:
        """Выполнить исследование пространства решений с probabilistic acceptance"""
        best_solution = self.evaluate_solution(current_routes)
        current_solution = best_solution.copy()

        for cycle in range(num_cycles):
  # Модифицируем решение с помощью нейронной политики.
            modified_routes = self._neural_route_replacement(current_solution.routes)
            new_solution = self.evaluate_solution(modified_routes)

  # Probabilistic acceptance для exploration.
            if new_solution.cost < current_solution.cost:
  # Принимаем улучшение.
                current_solution = new_solution
                self.successful_replacements += 1
            elif self._should_accept_worse_solution(current_solution.cost, new_solution.cost, cycle):
  # Иногда принимаем худшие решения для exploration.
                current_solution = new_solution
                self.successful_replacements += 1

  # Обновляем лучшее решение.
            if current_solution.cost < best_solution.cost:
                best_solution = current_solution

            self.total_replacement_attempts += 1

        return best_solution

    def _should_accept_worse_solution(self, current_cost: float, new_cost: float,
                                      cycle: int, initial_temperature: float = 0.2) -> bool:
        """Simulated annealing для принятия худших решений"""
        if new_cost <= current_cost:
            return True

  # Адаптивная температура - уменьшается с циклами.
        temperature = initial_temperature * (0.8 ** cycle)
        cost_diff = new_cost - current_cost
        accept_prob = np.exp(-cost_diff / temperature)
        return np.random.random() < accept_prob

    def _calculate_exploration_bonus(self, new_route: List[int], existing_routes: List[List[int]]) -> float:
        """
        Вычислить бонус за исследование новых областей.
        Бонус основан на новизне маршрута относительно существующих маршрутов.
        """
        if not new_route or not existing_routes:
            return 0.0  # Если нет маршрутов, бонус равен 0.

        novelty_score = 0.0
        new_route_set = set(new_route)

        for existing_route in existing_routes:
            existing_set = set(existing_route)
            overlap = len(new_route_set.intersection(existing_set))
            max_length = max(len(new_route), len(existing_route))

  # Чем меньше пересечений, тем выше бонус.
            if max_length > 0:
                novelty_score += 1.0 - (overlap / max_length)

  # Средний бонус по всем существующим маршрутам.
        return novelty_score / len(existing_routes) if existing_routes else 0.0

    def _neural_route_replacement(self, routes: List[List[int]]) -> List[List[int]]:
        """Заменить один маршрут с использованием нейронной политики"""
        if not routes:
            return routes

        start_time = time.time()

        try:
  # Выбираем маршрут для замены.
            probabilities = self.get_route_selection_probabilities(routes)
            route_idx = np.random.choice(len(routes), p=probabilities)

  # Создаем копию маршрутов без выбранного маршрута.
            partial_routes = routes.copy()
            selected_route = partial_routes.pop(route_idx)

  # Используем нейронную политику для создания нового маршрута.
            new_route = self._generate_neural_route(partial_routes)

  # ИСПРАВЛЕНИЕ: Применяем exploration bonus.
            exploration_bonus = self._calculate_exploration_bonus(new_route, partial_routes)

  # Вставляем новый маршрут на место старого.
            modified_routes = partial_routes.copy()
            modified_routes.insert(route_idx, new_route)

  # Сохраняем bonus для применения в evaluate_solution.
            self._current_exploration_bonus = exploration_bonus * 0.05  # Небольшой вес.

            self.policy_call_time += time.time() - start_time
            return modified_routes

        except Exception as e:
            logger.warning(f"Ошибка в neural route replacement: {e}")
            self.policy_call_time += time.time() - start_time
            return routes

    def evaluate_solution(self, routes: List[List[int]]) -> BCOSolution:
        """Оценить качество решения с учетом exploration bonus"""
  # Базовая оценка.
        base_solution = super().evaluate_solution(routes)

  # Применяем exploration bonus если есть.
        if hasattr(self, '_current_exploration_bonus'):
            base_solution.cost -= self._current_exploration_bonus
            delattr(self, '_current_exploration_bonus')  # Убираем после использования.

        return base_solution

    def _generate_neural_route(self, existing_routes: List[List[int]]) -> List[int]:
        """
        Генерировать новый маршрут с использованием нейронной политики

        Args:
            existing_routes: уже существующие маршруты

        Returns:
            Новый маршрут
        """
        try:
  # Создаем MDP для планирования одного маршрута.
            mdp = TransitMDP(self.city_graph, self.od_matrix, self.cost_calculator, self.alpha)

  # Начальное состояние с существующими маршрутами.
            initial_state = MDPState(
                completed_routes=existing_routes.copy(),
                current_route=[],
                extend_mode=ExtendMode.EXTEND,
                city_graph=self.city_graph,
                od_matrix=self.od_matrix
            )

  # Генерируем маршрут с помощью нейронной политики.
            new_route = self._rollout_single_route(mdp, initial_state)

  # Проверяем валидность сгенерированного маршрута.
            if self._validate_route(new_route):
                return new_route
            else:
  # Если маршрут невалидный, создаем простой fallback маршрут.
                return self._create_fallback_route(existing_routes)

        except Exception as e:
            logger.warning(f"Ошибка генерации neural route: {e}")
            return self._create_fallback_route(existing_routes)

    def _rollout_single_route(self, mdp: TransitMDP, initial_state: MDPState, max_steps: int = 50) -> List[int]:
        """
        Выполнить rollout для создания одного маршрута

        Args:
            mdp: MDP экземпляр
            initial_state: начальное состояние
            max_steps: максимальное количество шагов

        Returns:
            Сгенерированный маршрут
        """
        current_state = initial_state
        step_count = 0

        self.neural_planner.eval()
        with torch.no_grad():
            while step_count < max_steps:
  # Проверяем условие остановки.
                if (len(current_state.current_route) >= config.mdp.max_route_length or
                        len(current_state.current_route) >= config.mdp.min_route_length and random.random() < 0.3):
                    break

                valid_actions = mdp.get_valid_actions(current_state)
                if not valid_actions:
                    break

                if current_state.extend_mode == ExtendMode.EXTEND:
  # Режим расширения маршрута.
                    action = self._select_extend_action(current_state, valid_actions)
                else:
  # Режим halt/continue.
                    action = self._select_halt_action(current_state, valid_actions)

  # Выполняем шаг.
                current_state, reward, done = mdp.step(current_state, action)
                step_count += 1

  # Если создан достаточно длинный маршрут, можем остановиться.
                if (len(current_state.current_route) >= config.mdp.min_route_length and
                        current_state.extend_mode == ExtendMode.HALT):
                    break

        return current_state.current_route if current_state.current_route else []

    def _select_extend_action(self, state: MDPState, valid_actions: List[Any]) -> Any:
        """Более робастный выбор действия с fallback стратегиями"""
        if not valid_actions:
            return None

  # Стратегия 1: Neural network (основная).
        try:
            city_features = self._extract_city_features()
            route_mask = self._create_route_mask(state.current_route)
            action_features = self._create_action_features(valid_actions)

            if action_features.size(0) > 0:
                node_features, edge_features = city_features
                action_logits = self.neural_planner.forward_extend(
                    node_features.unsqueeze(0),
                    edge_features.unsqueeze(0),
                    route_mask.unsqueeze(0),
                    action_features.unsqueeze(0)
                )[0]

  # Добавляем exploration noise.
                noise = torch.randn_like(action_logits) * 0.1
                action_logits = action_logits + noise

                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action_idx = action_dist.sample()

                return valid_actions[action_idx.item()]

        except Exception as e:
            logger.debug(f"Neural policy failed: {e}")

  # Стратегия 2: Heuristic-based selection (вместо random).
        try:
            return self._demand_based_action_selection(state, valid_actions)
        except Exception as e:
            logger.debug(f"Heuristic selection failed: {e}")

  # Стратегия 3: Random (последний резерв).
        return random.choice(valid_actions)

    def _demand_based_action_selection(self, state: MDPState, valid_actions: List[Any]) -> Any:
        """Эвристический выбор на основе спроса"""
        action_scores = []

        for action in valid_actions:
            score = 0.0

  # Оцениваем действие по спросу.
            for node_id in action.path:
                node_list = list(self.city_graph.nodes())
                if node_id in node_list:
                    node_idx = node_list.index(node_id)
                    if node_idx < self.od_matrix.shape[0]:
  # Суммируем входящий и исходящий спрос.
                        demand = np.sum(self.od_matrix[node_idx, :]) + np.sum(self.od_matrix[:, node_idx])
                        score += demand

  # Нормализуем по длине пути.
            if len(action.path) > 0:
                score /= len(action.path)

            action_scores.append(score)

  # Выбираем действие с наибольшим спросом с некоторой случайностью.
        if action_scores:
            scores_array = np.array(action_scores)
            if np.max(scores_array) > 0:
                scores_array = scores_array / np.max(scores_array)  # Нормализация.
                probs = np.exp(scores_array) / np.sum(np.exp(scores_array))
                selected_idx = np.random.choice(len(valid_actions), p=probs)
                return valid_actions[selected_idx]

        return random.choice(valid_actions)

    def _select_halt_action(self, state: MDPState, valid_actions: List[Any]) -> Any:
        """Выбрать действие halt/continue с использованием нейронной политики"""
        try:
  # Извлекаем признаки города.
            city_features = self._extract_city_features()

  # Создаем маску текущего маршрута.
            route_mask = self._create_route_mask(state.current_route)
            route_length = torch.tensor([len(state.current_route)], dtype=torch.float32)

  # Forward pass через нейронную сеть.
            node_features, edge_features = city_features
            decision_logits = self.neural_planner.forward_halt(
                node_features.unsqueeze(0),
                edge_features.unsqueeze(0),
                route_mask.unsqueeze(0),
                route_length
            )[0]

  # Фильтруем доступные решения.
            available_decisions = [action.decision for action in valid_actions]
            decision_to_idx = {'continue': 0, 'halt': 1}

            available_logits = []
            available_actions = []
            for action in valid_actions:
                idx = decision_to_idx[action.decision]
                available_logits.append(decision_logits[idx])
                available_actions.append(action)

            if available_logits:
                available_logits = torch.stack(available_logits)
                decision_probs = torch.softmax(available_logits, dim=-1)
                decision_dist = torch.distributions.Categorical(decision_probs)
                decision_idx = decision_dist.sample()

                return available_actions[decision_idx.item()]
            else:
                return random.choice(valid_actions)

        except Exception as e:
            logger.debug(f"Ошибка в select_halt_action: {e}")
            return random.choice(valid_actions)

    def _extract_city_features(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Расширенные признаки города для нейронной сети"""
        num_nodes = len(self.city_graph.nodes())
        node_list = list(self.city_graph.nodes())

  # Предвычисляем метрики центральности.
        degree_centrality = nx.degree_centrality(self.city_graph)
        betweenness_centrality = nx.betweenness_centrality(self.city_graph)
        closeness_centrality = nx.closeness_centrality(self.city_graph)

  # Анализируем OD матрицу.
        node_demand_in = np.sum(self.od_matrix, axis=0)  # Входящий спрос.
        node_demand_out = np.sum(self.od_matrix, axis=1)  # Исходящий спрос.
        max_demand = max(np.max(node_demand_in), np.max(node_demand_out), 1.0)

  # Улучшенные node features.
        node_features = []
        for i, node_id in enumerate(node_list):
            node_data = self.city_graph.nodes[node_id]

  # Нормализованные координаты.
            x_norm = node_data.get('x', 0.0) / 1000.0
            y_norm = node_data.get('y', 0.0) / 1000.0

  # Метрики центральности.
            degree_cent = degree_centrality.get(node_id, 0.0)
            between_cent = betweenness_centrality.get(node_id, 0.0)
            close_cent = closeness_centrality.get(node_id, 0.0)

  # Спрос (нормализованный).
            demand_in = node_demand_in[i] / max_demand if i < len(node_demand_in) else 0.0
            demand_out = node_demand_out[i] / max_demand if i < len(node_demand_out) else 0.0

            features = [
                x_norm, y_norm,
                demand_in, demand_out,  # Реальные данные спроса.
                degree_cent, between_cent, close_cent,
                len(list(self.city_graph.neighbors(node_id))) / num_nodes  # Локальная связность.
            ]
            node_features.append(features)

        node_features = torch.tensor(node_features, dtype=torch.float32)

  # Улучшенные edge features.
        edge_features = torch.zeros(num_nodes, num_nodes, config.network.edge_feature_dim)

        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                if i != j:
  # Спрос между узлами.
                    demand = self.od_matrix[i, j] if i < self.od_matrix.shape[0] and j < self.od_matrix.shape[
                        1] else 0.0

  # Геометрическое расстояние.
                    xi, yi = self.city_graph.nodes[node_i]['x'], self.city_graph.nodes[node_i]['y']
                    xj, yj = self.city_graph.nodes[node_j]['x'], self.city_graph.nodes[node_j]['y']
                    distance = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2) / 1000.0

  # Есть ли прямое ребро.
                    has_edge = 1.0 if self.city_graph.has_edge(node_i, node_j) else 0.0

  # Время поездки.
                    try:
                        travel_time = nx.shortest_path_length(self.city_graph, node_i, node_j,
                                                              weight='travel_time') / 60.0
                    except:
                        travel_time = distance * 4

  # Нормализованный спрос.
                    norm_demand = demand / max_demand if max_demand > 0 else 0.0

  # Совместная важность узлов.
                    joint_importance = degree_centrality.get(node_i, 0) * degree_centrality.get(node_j, 0)

                    edge_features[i, j] = torch.tensor([
                        norm_demand,
                        distance,
                        0.0,  # Existing_transit.
                        has_edge,
                        travel_time,
                        joint_importance  # Новый признак.
                    ])

        return node_features, edge_features

    def _create_route_mask(self, route: List[int]) -> torch.Tensor:
        """Создать маску для текущего маршрута"""
        num_nodes = len(self.city_graph.nodes())
        mask = torch.zeros(num_nodes)

        node_list = list(self.city_graph.nodes())
        for node_id in route:
            if node_id in node_list:
                mask[node_list.index(node_id)] = 1.0

        return mask

    def _create_action_features(self, valid_actions: List[Any]) -> torch.Tensor:
        """Создать features для действий расширения"""
        if not valid_actions:
            return torch.zeros(0, config.network.hidden_dim)

        action_features = []
        node_list = list(self.city_graph.nodes())

        for action in valid_actions:
  # Простые features действия - среднее положение узлов в пути.
            path_coords = []
            for node_id in action.path:
                if node_id in node_list:
                    node_data = self.city_graph.nodes[node_id]
                    path_coords.append([node_data.get('x', 0.0), node_data.get('y', 0.0)])

            if path_coords:
                mean_coord = np.mean(path_coords, axis=0)
  # Создаем простой feature vector.
                feature = torch.zeros(config.network.hidden_dim)
                feature[0] = mean_coord[0] / 1000.0  # Нормализованная x координата.
                feature[1] = mean_coord[1] / 1000.0  # Нормализованная y координата.
                feature[2] = len(action.path) / config.mdp.max_route_length  # Нормализованная длина пути.
                action_features.append(feature)
            else:
                action_features.append(torch.zeros(config.network.hidden_dim))

        return torch.stack(action_features)

    def _validate_route(self, route: List[int]) -> bool:
        """Проверить валидность маршрута"""
        if not route:
            return False

  # Проверяем длину.
        if len(route) < config.mdp.min_route_length or len(route) > config.mdp.max_route_length:
            return False

  # Проверяем на циклы.
        if len(set(route)) != len(route):
            return False

  # Проверяем, что все узлы существуют.
        for node in route:
            if node not in self.city_graph.nodes():
                return False

        return True

    def _create_fallback_route(self, existing_routes: List[List[int]]) -> List[int]:
        """Создать простой fallback маршрут"""
        try:
  # Получаем узлы, не используемые в существующих маршрутах.
            used_nodes = set()
            for route in existing_routes:
                used_nodes.update(route)

            available_nodes = [n for n in self.city_graph.nodes() if n not in used_nodes]

            if len(available_nodes) >= config.mdp.min_route_length:
  # Создаем маршрут из доступных узлов.
                selected_nodes = random.sample(available_nodes,
                                               min(config.mdp.min_route_length, len(available_nodes)))
                return selected_nodes
            else:
  # Если мало доступных узлов, создаем маршрут из любых узлов.
                all_nodes = list(self.city_graph.nodes())
                return random.sample(all_nodes, min(config.mdp.min_route_length, len(all_nodes)))

        except Exception as e:
            logger.warning(f"Ошибка создания fallback route: {e}")
            return list(self.city_graph.nodes())[:config.mdp.min_route_length]

    def set_exploration_schedule(self, iteration: int, max_iterations: int):
        """Адаптивное расписание exploration в зависимости от прогресса"""
  # Высокая exploration в начале, низкая в конце.
        progress = iteration / max_iterations
        temperature = 1.5 * (1.0 - progress) + 0.8 * progress  # От 1.5 до 0.8.

  # Устанавливаем температуру в нейронной политике.
        if hasattr(self.neural_planner, 'extend_head'):
            self.neural_planner.extend_head.set_exploration_temperature(temperature)

        logger.debug(f"Neural Bee {self.bee_id}: установлена температура {temperature:.2f}")

    def get_statistics(self) -> Dict[str, float]:
        """Получить статистику работы neural bee"""
        return {
            'successful_replacements': self.successful_replacements,
            'total_replacement_attempts': self.total_replacement_attempts,
            'success_rate': (self.successful_replacements / self.total_replacement_attempts
                             if self.total_replacement_attempts > 0 else 0.0),
            'policy_call_time': self.policy_call_time
        }

class NeuralBeeColonyOptimization(BeeColonyOptimization):
    """
    Neural Bee Colony Optimization - гибридный алгоритм
    Заменяет Type-1 пчел на Neural Bees, сохраняя Type-2 пчел для разнообразия
    """

    def __init__(self,
                 city_graph: nx.Graph,
                 od_matrix: np.ndarray,
                 neural_planner: NeuralPlanner,
                 cost_calculator: Optional[TransitCostCalculator] = None,
                 num_bees: int = None,
                 num_cycles: int = None,
                 num_passes: int = None,
                 num_iterations: int = None,
                 alpha: float = 0.5,
                 neural_bee_ratio: float = None):
        """
        Args:
            city_graph: граф дорожной сети города
            od_matrix: матрица Origin-Destination спроса
            neural_planner: обученная нейронная сеть
            cost_calculator: калькулятор стоимости
            num_bees: количество пчел (B в статье)
            num_cycles: количество циклов (NC в статье)
            num_passes: количество проходов (NP в статье)
            num_iterations: количество итераций (I в статье)
            alpha: параметр стоимостной функции
            neural_bee_ratio: доля neural bees от общего числа пчел
        """
  # Вызываем конструктор родительского класса без создания пчел.
        self.city_graph = city_graph
        self.od_matrix = od_matrix
        self.alpha = alpha

  # Параметры алгоритма.
        self.num_bees = num_bees or config.bco.num_bees
        self.num_cycles = num_cycles or config.bco.num_cycles
        self.num_passes = num_passes or config.bco.num_passes
        self.num_iterations = num_iterations or config.bco.num_iterations
        self.neural_bee_ratio = neural_bee_ratio or config.bco.neural_bee_ratio

  # Калькулятор стоимости.
        if cost_calculator is None:
            self.cost_calculator = TransitCostCalculator(alpha=alpha)
        else:
            self.cost_calculator = cost_calculator

  # Нейронная сеть.
        self.neural_planner = neural_planner

  # Создаем гибридный рой пчел.
        self.bees = self._create_hybrid_bees()

  # Статистика.
        self.statistics: List[NeuralBCOStatistics] = []
        self.best_solution: Optional[BCOSolution] = None

  # НОВОЕ: Детальный сбор данных для визуализации.
        self.iteration_history = []
        self.neural_performance_history = []
        self.swarm_diversity_history = []

  # ДОБАВИТЬ ЭТО:.
        self.diversity_history = []

  # Логируем режим работы.
        mode = "Neural-only" if not config.evaluation.enable_classical_comparison else "Comparison"
        logger.info(f"Neural BCO инициализирован ({mode} режим): {self.num_bees} пчел, "
                    f"{len([b for b in self.bees if isinstance(b, NeuralBee)])} neural bees, "
                    f"{len([b for b in self.bees if isinstance(b, Type2Bee)])} type-2 bees")

    def log_iteration_data(self, iteration):
        """НОВОЕ: Сбор реальных данных каждой итерации"""
  # Состояние роя.
        swarm_costs = [bee.current_solution.cost for bee in self.bees if bee.current_solution]
        swarm_state = {
            'iteration': iteration,
            'best_cost': min(swarm_costs) if swarm_costs else float('inf'),
            'average_cost': np.mean(swarm_costs) if swarm_costs else float('inf'),
            'worst_cost': max(swarm_costs) if swarm_costs else float('inf'),
            'diversity': np.std(swarm_costs) / np.mean(swarm_costs) if swarm_costs and np.mean(swarm_costs) > 0 else 0
        }

  # Neural Bees статистика.
        neural_bees = [bee for bee in self.bees if isinstance(bee, NeuralBee)]
        neural_metrics = {
            'iteration': iteration,
            'total_calls': sum(bee.total_replacement_attempts for bee in neural_bees),
            'total_successes': sum(bee.successful_replacements for bee in neural_bees),
            'success_rate': sum(bee.successful_replacements for bee in neural_bees) / max(
                sum(bee.total_replacement_attempts for bee in neural_bees), 1),
            'policy_time': sum(bee.policy_call_time for bee in neural_bees)
        }

  # Анализ лучшего решения.
        best_solution_metrics = {}
        if self.best_solution and self.best_solution.routes:
            covered_nodes = set()
            for route in self.best_solution.routes:
                covered_nodes.update(route)

            best_solution_metrics = {
                'iteration': iteration,
                'cost': self.best_solution.cost,
                'num_routes': len(self.best_solution.routes),
                'avg_route_length': np.mean([len(route) for route in self.best_solution.routes]),
                'coverage_nodes': len(covered_nodes),
                'coverage_ratio': len(covered_nodes) / len(self.city_graph.nodes()) if len(
                    self.city_graph.nodes()) > 0 else 0,
                'is_feasible': self.best_solution.is_feasible
            }

  # Сохраняем все данные.
        self.iteration_history.append(swarm_state)
        self.neural_performance_history.append(neural_metrics)

  # Вычисляем разнообразие решений.
        diversity = self._calculate_solution_diversity()
        self.swarm_diversity_history.append({
            'iteration': iteration,
            'diversity': diversity
        })

        return {
            'swarm_state': swarm_state,
            'neural_metrics': neural_metrics,
            'best_solution_metrics': best_solution_metrics,
            'diversity': diversity
        }

    def _create_hybrid_bees(self) -> List[BeeAgent]:
        """Создать гибридный рой: Neural Bees + Type-2 Bees"""
        bees = []

        num_neural = int(self.num_bees * self.neural_bee_ratio)
        num_type2 = self.num_bees - num_neural

  # Neural Bees (заменяют Type-1).
        for i in range(num_neural):
            bee = NeuralBee(
                bee_id=i,
                city_graph=self.city_graph,
                cost_calculator=self.cost_calculator,
                od_matrix=self.od_matrix,
                neural_planner=self.neural_planner,
                alpha=self.alpha
            )
            bees.append(bee)

  # Type-2 Bees (остаются без изменений).
        for i in range(num_neural, num_neural + num_type2):
            bee = Type2Bee(
                bee_id=i,
                city_graph=self.city_graph,
                cost_calculator=self.cost_calculator,
                od_matrix=self.od_matrix
            )
            bees.append(bee)

        logger.info(f"Создано {num_neural} Neural Bees и {num_type2} Type-2 Bees")
        return bees

    def optimize(self,
                 initial_routes: Optional[List[List[int]]] = None,
                 save_raw_routes: Optional[str] = None,
                 post_process: bool = True):
        """
        Выполнить Neural BCO оптимизацию с улучшенными механизмами

        Добавлены:
        - Diversity tracking для предотвращения стагнации
        - Restart mechanism для восстановления разнообразия
        - Adaptive temperature для Neural Bees
        """
        logger.info(" Запуск Neural BCO оптимизации...")
        start_time = time.time()

  # Логируем начальные параметры.
        logger.info(" Параметры оптимизации:")
        logger.info(f"  Пчел: {self.num_bees} (Neural: {len([b for b in self.bees if isinstance(b, NeuralBee)])}, "
                    f"Type-2: {len([b for b in self.bees if isinstance(b, Type2Bee)])})")
        logger.info(f"  Итераций: {self.num_iterations}, Циклов: {self.num_cycles}, Проходов: {self.num_passes}")
        logger.info(f"  Граф: {len(self.city_graph.nodes())} узлов, {len(self.city_graph.edges())} ребер")
        logger.info(f"  Спрос: {self.od_matrix.sum():.0f} поездок/день")

  # Инициализация.
        if initial_routes is None:
            initial_routes = self.generate_initial_solution()

  # Инициализируем всех пчел одинаковым решением.
        for bee in self.bees:
            bee.current_solution = bee.evaluate_solution(initial_routes)
            bee.best_solution = bee.current_solution.copy()

  # Глобальное лучшее решение.
        self.best_solution = min([bee.best_solution for bee in self.bees], key=lambda x: x.cost)

  # RAW-DUMP.
  # Сохраняем маршруты, сгенерированные колонией, до любого сглаживания.
        if save_raw_routes:
            try:
                RouteUtils.save_routes(self.best_solution.routes, save_raw_routes)
                logger.info(f" Маршруты до 2-opt сохранены в {save_raw_routes}")
            except Exception as exc:
                logger.warning(f"️  Не удалось сохранить raw-routes: {exc}")

  # POST-PROCESS (2-opt).
        if post_process and getattr(config.mdp, "apply_post_opt", True):
            smoothed = RouteUtils.smooth_routes(self.best_solution.routes,
                                                self.city_graph)
            if smoothed != self.best_solution.routes:
                old_cost = self.best_solution.cost
                new_cost = self.cost_calculator.calculate_cost(
                    self.city_graph, smoothed, self.od_matrix
                )
                self.best_solution.routes = smoothed
                self.best_solution.cost = new_cost
                logger.info(f" 2-opt улучшил решение: {old_cost:.3f} → {new_cost:.3f}")
            else:
                logger.info("ℹ️  2-opt не изменил маршруты")
        else:
            logger.info(" Пост-обработка 2-opt пропущена")

        logger.info(f" Начальная стоимость: {self.best_solution.cost:.3f}")

  # НОВОЕ: Переменные для отслеживания стагнации и diversity.
        last_improvement_iteration = 0
        best_cost_history = [self.best_solution.cost]
        stagnation_counter = 0
        diversity_history = []

  # Основной цикл оптимизации.
        for iteration in range(self.num_iterations):
            iteration_start = time.time()

  # Детальное логирование начала итерации.
            if iteration % 10 == 0:
                logger.info(f" === ИТЕРАЦИЯ {iteration + 1}/{self.num_iterations} ===")

  # Текущее состояние роя.
                current_costs = [bee.current_solution.cost for bee in self.bees]
                logger.info(f" Состояние роя: лучшая={min(current_costs):.3f}, "
                            f"средняя={np.mean(current_costs):.3f}, "
                            f"худшая={max(current_costs):.3f}, "
                            f"разброс={np.std(current_costs):.3f}")

  # НОВОЕ: Устанавливаем адаптивную exploration температуру для Neural Bees.
            for bee in self.bees:
                if isinstance(bee, NeuralBee):
                    self._set_adaptive_temperature(bee, iteration, self.num_iterations)

  # Фаза исследования с детальным логированием.
            exploration_start = time.time()
            neural_improvements, type2_improvements = self._hybrid_exploration_phase()
            exploration_time = time.time() - exploration_start

  # Детальная статистика Neural Bees.
            neural_stats = self._get_neural_bee_statistics()

  # Логируем производительность каждые 5 итераций.
            if (iteration + 1) % 5 == 0:
                logger.info(f" Итерация {iteration + 1}: "
                            f"Neural улучшений={neural_improvements}, "
                            f"Type-2 улучшений={type2_improvements}, "
                            f"Neural успешность={neural_stats.get('avg_success_rate', 0):.1%}, "
                            f"время={exploration_time:.2f}s")

  # Фаза рекрутинга (каждые NP итераций).
            if (iteration + 1) % self.num_passes == 0:
                recruitment_start = time.time()
                recruitment_events = self._recruitment_phase()
                recruitment_time = time.time() - recruitment_start

                logger.info(f" Рекрутинг: {recruitment_events} событий за {recruitment_time:.2f}s")
            else:
                recruitment_events = 0
                recruitment_time = 0.0

  # НОВОЕ: Логируем данные итерации.
            iteration_data = self.log_iteration_data(iteration)

  # НОВОЕ: Улучшенное обновление лучшего решения с tracking стагнации.
            current_best = min([bee.best_solution for bee in self.bees], key=lambda x: x.cost)
            if current_best.cost < self.best_solution.cost:
                improvement = self.best_solution.cost - current_best.cost
                improvement_percent = (improvement / self.best_solution.cost) * 100

                self.best_solution = current_best.copy()
                last_improvement_iteration = iteration
                stagnation_counter = 0

                logger.info(f" НОВОЕ ЛУЧШЕЕ РЕШЕНИЕ на итерации {iteration + 1}!")
                logger.info(
                    f"   Стоимость: {current_best.cost:.3f} (улучшение: {improvement:.3f}, {improvement_percent:.1f}%)")
                logger.info(f"   Маршрутов: {len(current_best.routes)}")
                logger.info(f"   Допустимо: {current_best.is_feasible}")

  # Детальный анализ нового лучшего решения.
                if hasattr(self, 'cost_calculator'):
                    cost_breakdown = self.cost_calculator.get_cost_breakdown(
                        self.city_graph, current_best.routes, self.od_matrix, self.alpha
                    )
                    logger.info(f"    Пассажирская стоимость: {cost_breakdown['passenger_cost']:.2f} мин")
                    logger.info(f"    Операторская стоимость: {cost_breakdown['operator_cost']:.2f} мин")
                    logger.info(f"    Связность: {cost_breakdown['connectivity_ratio']:.1%}")
            else:
                stagnation_counter += 1

            best_cost_history.append(self.best_solution.cost)

  # НОВОЕ: Вычисляем и отслеживаем diversity.
            diversity = self._calculate_swarm_diversity()
            diversity_history.append(diversity)

  # НОВОЕ: Restart mechanism при длительной стагнации.
            if stagnation_counter >= 30 and iteration > 50:  # После 30 итераций без улучшения.
                if diversity < 0.1:  # Очень низкое разнообразие.
                    self._restart_worst_bees(fraction=0.25)
                    logger.info(f" RESTART: перезапущено 25% худших пчел на итерации {iteration + 1}")
                    logger.info(f"   Причина: стагнация {stagnation_counter} итераций, diversity={diversity:.3f}")
                    stagnation_counter = 0

  # Анализ сходимости.
            if iteration >= 20 and (iteration + 1) % 20 == 0:
                recent_improvement = best_cost_history[-20] - best_cost_history[-1]
                convergence_rate = recent_improvement / best_cost_history[-20] * 100 if best_cost_history[
                                                                                            -20] > 0 else 0

                logger.info(f" Анализ сходимости (последние 20 итераций):")
                logger.info(f"   Улучшение: {recent_improvement:.3f} ({convergence_rate:.2f}%)")
                logger.info(f"   Стагнация: {stagnation_counter} итераций")
                logger.info(f"   Последнее улучшение: итерация {last_improvement_iteration + 1}")
                logger.info(f"   Diversity: {diversity:.3f}")

  # Предупреждение о стагнации.
            if stagnation_counter >= 100:
                logger.warning(f"️ Стагнация: {stagnation_counter} итераций без улучшения")

  # Детальная статистика каждые 50 итераций.
            if (iteration + 1) % 50 == 0:
                self._log_detailed_iteration_stats(iteration, neural_stats, neural_improvements, type2_improvements)

  # Собираем статистику для внутреннего использования.
            iteration_time = time.time() - iteration_start
            stats = self._collect_neural_statistics(
                iteration, exploration_time, recruitment_time, recruitment_events,
                neural_improvements, type2_improvements
            )
            self.statistics.append(stats)

  # Финальная статистика.
        total_time = time.time() - start_time
        final_neural_stats = self._get_neural_bee_statistics()

        logger.info(" === ОПТИМИЗАЦИЯ ЗАВЕРШЕНА ===")
        logger.info(f"️ Общее время: {total_time:.1f}s ({total_time / 60:.1f} мин)")
        logger.info(f" Финальная стоимость: {self.best_solution.cost:.3f}")
        logger.info(
            f" Общее улучшение: {((best_cost_history[0] - self.best_solution.cost) / best_cost_history[0] * 100):.1f}%")
        logger.info(f" Neural Bees эффективность: {final_neural_stats.get('avg_success_rate', 0):.1%}")
        logger.info(f" Время работы политики: {final_neural_stats.get('total_policy_time', 0):.1f}s")
        logger.info(f" Финальная diversity: {diversity_history[-1] if diversity_history else 0:.3f}")
        logger.info(f" Последнее улучшение: итерация {last_improvement_iteration + 1}")

        return self.best_solution

    def _set_adaptive_temperature(self, neural_bee: 'NeuralBee', iteration: int, max_iterations: int):
        """Более агрессивная адаптивная температура"""
        progress = iteration / max_iterations

  # Нелинейная функция температуры.
        if progress < 0.3:
  # Высокая exploration в начале.
            temperature = 2.0 - progress * 2.0  # От 2.0 до 1.4.
        elif progress < 0.7:
  # Постепенное снижение.
            temperature = 1.4 - (progress - 0.3) * 1.0  # От 1.4 до 1.0.
        else:
  # Низкая exploration в конце.
            temperature = 1.0 - (progress - 0.7) * 0.5  # От 1.0 до 0.85.

  # Устанавливаем температуру.
        if hasattr(neural_bee.neural_planner, 'extend_head'):
            neural_bee.neural_planner.extend_head.set_exploration_temperature(temperature)

    def _restart_worst_bees(self, fraction: float = 0.25):
        """ НОВОЕ: Перезапустить худших пчел для восстановления diversity"""
        num_to_restart = int(len(self.bees) * fraction)

  # Сортируем пчел по стоимости (худшие первые).
        sorted_bees = sorted(self.bees, key=lambda b: b.current_solution.cost, reverse=True)

        restarted_bees = []
        for i in range(num_to_restart):
            bee = sorted_bees[i]
            old_cost = bee.current_solution.cost

  # Генерируем новое случайное решение.
            new_routes = self.generate_initial_solution()
            bee.current_solution = bee.evaluate_solution(new_routes)
            bee.best_solution = bee.current_solution.copy()

  # Сбрасываем статистику Neural Bee.
            if isinstance(bee, NeuralBee):
                bee.successful_replacements = 0
                bee.total_replacement_attempts = 0
                bee.policy_call_time = 0.0

            restarted_bees.append({
                'bee_id': bee.bee_id,
                'old_cost': old_cost,
                'new_cost': bee.current_solution.cost,
                'type': bee.type_name
            })

        logger.info(f" Перезапущено {num_to_restart} пчел для восстановления diversity:")
        for bee_info in restarted_bees:
            logger.info(f"   {bee_info['type']} Bee {bee_info['bee_id']}: "
                        f"{bee_info['old_cost']:.3f} → {bee_info['new_cost']:.3f}")

    def _log_detailed_iteration_stats(self, iteration, neural_stats, neural_improvements, type2_improvements):
        """Детальное логирование статистики итерации"""
        logger.info(f" === ДЕТАЛЬНАЯ СТАТИСТИКА ИТЕРАЦИИ {iteration + 1} ===")

  # Статистика роя.
        costs = [bee.current_solution.cost for bee in self.bees]
        feasible_count = sum(1 for bee in self.bees if bee.current_solution.is_feasible)

        logger.info(f" Рой пчел:")
        logger.info(f"   Лучшая стоимость: {min(costs):.3f}")
        logger.info(f"   Средняя стоимость: {np.mean(costs):.3f}")
        logger.info(f"   Худшая стоимость: {max(costs):.3f}")
        logger.info(f"   Стандартное отклонение: {np.std(costs):.3f}")
        logger.info(f"   Допустимых решений: {feasible_count}/{len(self.bees)}")

  # Статистика Neural Bees.
        neural_bees = [bee for bee in self.bees if isinstance(bee, NeuralBee)]
        if neural_bees:
            logger.info(f" Neural Bees ({len(neural_bees)} пчел):")
            logger.info(f"   Успешных улучшений: {neural_improvements}")
            logger.info(f"   Общая успешность: {neural_stats.get('avg_success_rate', 0):.1%}")
            logger.info(f"   Всего вызовов политики: {neural_stats.get('total_calls', 0)}")
            logger.info(f"   Время работы политики: {neural_stats.get('total_policy_time', 0):.2f}s")
            logger.info(f"   Среднее время на вызов: {neural_stats.get('avg_policy_time_per_call', 0):.4f}s")

  # Статистика Type-2 Bees.
        type2_bees = [bee for bee in self.bees if isinstance(bee, Type2Bee)]
        if type2_bees:
            logger.info(f" Type-2 Bees ({len(type2_bees)} пчел):")
            logger.info(f"   Успешных улучшений: {type2_improvements}")
            type2_costs = [bee.current_solution.cost for bee in type2_bees]
            logger.info(f"   Средняя стоимость: {np.mean(type2_costs):.3f}")

  # Анализ разнообразия.
        diversity = self._calculate_solution_diversity()
        logger.info(f" Разнообразие решений: {diversity:.3f}")

  # Анализ лучшего решения.
        if self.best_solution:
            routes = self.best_solution.routes
            if routes:
                route_lengths = [len(route) for route in routes]
                covered_nodes = set()
                for route in routes:
                    covered_nodes.update(route)

                logger.info(f" Лучшее решение:")
                logger.info(f"   Маршрутов: {len(routes)}")
                logger.info(f"   Средняя длина маршрута: {np.mean(route_lengths):.1f}")
                logger.info(
                    f"   Покрытие узлов: {len(covered_nodes)}/{len(self.city_graph.nodes())} ({len(covered_nodes) / len(self.city_graph.nodes()) * 100:.1f}%)")
                logger.info(f"   Нарушений ограничений: {self.best_solution.constraint_violations}")

        logger.info("=" * 60)

    def _hybrid_exploration_phase(self) -> Tuple[int, int]:
        """
        Гибридная фаза исследования: Neural Bees + Type-2 Bees

        Returns:
            (neural_improvements, type2_improvements)
        """
        neural_improvements = 0
        type2_improvements = 0

        for bee in self.bees:
            initial_cost = bee.current_solution.cost

  # Исследование с текущего решения.
            new_solution = bee.explore(bee.current_solution.routes, self.num_cycles)

  # Обновляем текущее решение если найдено улучшение.
            if new_solution.cost < bee.current_solution.cost:
                bee.current_solution = new_solution

  # Подсчитываем улучшения по типам пчел.
                if isinstance(bee, NeuralBee):
                    neural_improvements += 1
                elif isinstance(bee, Type2Bee):
                    type2_improvements += 1

  # Обновляем лучшее решение пчелы.
                if new_solution.cost < bee.best_solution.cost:
                    bee.best_solution = new_solution.copy()

        return neural_improvements, type2_improvements

    def _collect_neural_statistics(self,
                                   iteration: int,
                                   exploration_time: float,
                                   evaluation_time: float,
                                   recruitment_events: int,
                                   neural_improvements: int,
                                   type2_improvements: int) -> NeuralBCOStatistics:
        """Собрать расширенную статистику Neural BCO"""
        costs = [bee.current_solution.cost for bee in self.bees]
        feasible_solutions = sum(1 for bee in self.bees if bee.current_solution.is_feasible)

  # Статистика neural bees.
        neural_bees = [bee for bee in self.bees if isinstance(bee, NeuralBee)]
        neural_stats = self._get_neural_bee_statistics()

  # Вычисляем разнообразие решений.
        diversity_score = self._calculate_solution_diversity()

        return NeuralBCOStatistics(
            iteration=iteration + 1,
            best_cost=min(costs),
            avg_cost=np.mean(costs),
            worst_cost=max(costs),
            feasible_solutions=feasible_solutions,
            type1_improvements=neural_improvements,  # Neural bees заменяют Type-1.
            type2_improvements=type2_improvements,
            recruitment_events=recruitment_events,
            exploration_time=exploration_time,
            evaluation_time=evaluation_time,
            neural_bee_improvements=neural_improvements,
            neural_bee_calls=neural_stats.get('total_calls', 0),
            neural_policy_time=neural_stats.get('total_policy_time', 0.0),
            route_replacement_success_rate=neural_stats.get('avg_success_rate', 0.0),
            diversity_score=diversity_score
        )

    def _get_neural_bee_statistics(self) -> Dict[str, float]:
        """Получить агрегированную статистику neural bees"""
        neural_bees = [bee for bee in self.bees if isinstance(bee, NeuralBee)]

        if not neural_bees:
            return {}

        total_calls = sum(bee.total_replacement_attempts for bee in neural_bees)
        total_successes = sum(bee.successful_replacements for bee in neural_bees)
        total_policy_time = sum(bee.policy_call_time for bee in neural_bees)

        return {
            'total_calls': total_calls,
            'total_successes': total_successes,
            'total_policy_time': total_policy_time,
            'avg_success_rate': total_successes / total_calls if total_calls > 0 else 0.0,
            'avg_policy_time_per_call': total_policy_time / total_calls if total_calls > 0 else 0.0
        }

    def _calculate_solution_diversity(self) -> float:
        """Вычислить меру разнообразия решений в рое"""
        solutions = [bee.current_solution.routes for bee in self.bees]

        if len(solutions) < 2:
            return 0.0

  # Вычисляем попарные различия между решениями.
        diversity_scores = []

        for i in range(len(solutions)):
            for j in range(i + 1, len(solutions)):
                diversity = self._calculate_route_set_diversity(solutions[i], solutions[j])
                diversity_scores.append(diversity)

        return np.mean(diversity_scores) if diversity_scores else 0.0

    def _calculate_route_set_diversity(self, routes1: List[List[int]], routes2: List[List[int]]) -> float:
        """Вычислить различие между двумя наборами маршрутов"""
        if not routes1 or not routes2:
            return 1.0 if len(routes1) != len(routes2) else 0.0

  # Преобразуем маршруты в множества узлов.
        set1 = set()
        set2 = set()

        for route in routes1:
            set1.update(route)

        for route in routes2:
            set2.update(route)

  # Вычисляем коэффициент Жаккара.
        if len(set1) == 0 and len(set2) == 0:
            return 0.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        jaccard_similarity = intersection / union if union > 0 else 0.0
        return 1.0 - jaccard_similarity  # Возвращаем различие.

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Получить информацию об алгоритме"""
        base_info = super().get_algorithm_info()

        neural_info = {
            'algorithm': 'Neural Bee Colony Optimization',
            'neural_bee_ratio': self.neural_bee_ratio,
            'neural_bees': len([bee for bee in self.bees if isinstance(bee, NeuralBee)]),
            'type2_bees': len([bee for bee in self.bees if isinstance(bee, Type2Bee)]),
            'neural_planner_architecture': 'Graph Attention Network',
            'has_trained_policy': True
        }

        base_info.update(neural_info)
        return base_info

    def compare_with_classical_bco(self, classical_bco_result: BCOSolution) -> Dict[str, Any]:
        """
        Сравнить результаты с классическим BCO

        Args:
            classical_bco_result: результат классического BCO

        Returns:
            Словарь с метриками сравнения
        """
        if self.best_solution is None:
            return {'error': 'Neural BCO not yet optimized'}

        improvement = classical_bco_result.cost - self.best_solution.cost
        improvement_percent = (improvement / classical_bco_result.cost) * 100 if classical_bco_result.cost > 0 else 0

  # Дополнительные метрики качества.
        neural_metrics = self.cost_calculator.get_cost_breakdown(
            self.city_graph, self.best_solution.routes, self.od_matrix, self.alpha
        )

        classical_metrics = self.cost_calculator.get_cost_breakdown(
            self.city_graph, classical_bco_result.routes, self.od_matrix, self.alpha
        )

        return {
            'cost_improvement': improvement,
            'improvement_percent': improvement_percent,
            'neural_bco_cost': self.best_solution.cost,
            'classical_bco_cost': classical_bco_result.cost,
            'neural_metrics': neural_metrics,
            'classical_metrics': classical_metrics,
            'neural_feasible': self.best_solution.is_feasible,
            'classical_feasible': classical_bco_result.is_feasible,
            'neural_statistics': self._get_neural_bee_statistics()
        }

    def get_detailed_results_for_visualization(self) -> Dict[str, Any]:
        """НОВОЕ: Получить все собранные данные для визуализации"""
        if self.best_solution is None:
            return {'error': 'No optimization results available'}

  # Анализ маршрутов.
        routes = self.best_solution.routes
        covered_nodes = set()
        for route in routes:
            covered_nodes.update(route)

  # Пересечения маршрутов.
        route_overlaps = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                overlap = len(set(routes[i]).intersection(set(routes[j])))
                route_overlaps.append({
                    'route_1': i,
                    'route_2': j,
                    'overlap_count': overlap,
                    'overlap_ratio': overlap / min(len(routes[i]), len(routes[j])) if min(len(routes[i]),
                                                                                          len(routes[j])) > 0 else 0
                })

        return {
            'iteration_history': self.iteration_history,
            'neural_performance': self.neural_performance_history,
            'swarm_diversity_history': self.swarm_diversity_history,
            'route_analysis': {
                'route_lengths': [len(route) for route in routes],
                'covered_nodes': list(covered_nodes),
                'total_route_nodes': sum(len(route) for route in routes),
                'route_overlaps': route_overlaps,
                'avg_route_length': np.mean([len(route) for route in routes]),
                'route_length_std': np.std([len(route) for route in routes])
            },
            'final_statistics': self._get_neural_bee_statistics(),
            'algorithm_info': self.get_algorithm_info()
        }

def load_trained_neural_planner(model_path: str = None,
                                device: str = None) -> NeuralPlanner:
    """
    Загрузить обученную нейронную сеть

    Args:
        model_path: путь к файлу модели
        device: устройство для вычислений

    Returns:
        Загруженная нейронная сеть
    """
    if model_path is None:
        model_path = f"{config.files.models_dir}/{config.files.model_checkpoint_name}"

    if device is None:
        device = config.system.device

    try:
  # Создаем trainer для загрузки модели.
        trainer = NeuralPlannerTrainer(device=device)
        trainer.load_model(model_path)

        logger.info(f"Нейронная сеть загружена из {model_path}")
        return trainer.neural_planner

    except Exception as e:
        logger.error(f"Ошибка загрузки нейронной сети из {model_path}: {e}")
        raise

def run_neural_bco_optimization(
        G,
        od_matrix,
        alpha: float = 0.5,
        initial_routes: Optional[List[List[int]]] = None,
        **opt_kwargs,  # ←  здесь «поймаем» save_raw_routes и post_process.
):
    """
    Wrapper для запуска Neural Bee Colony Optimization.

    Параметры в **opt_kwargs передаются напрямую в
    NeuralBeeColonyOptimization.optimize(), поэтому можно указать:
        save_raw_routes="routes_raw.pkl",
        post_process=False,
        …
    Старые вызовы без дополнительных аргументов продолжают работать.
    """
  # 1) берём политику из opt_kwargs или загружаем с диска.
    neural_planner = opt_kwargs.pop("neural_planner", None)

    if neural_planner is None:
        neural_planner = load_trained_neural_planner()

  # 2) создаём оптимизатор, передавая параметры явно по именам.
    nbco = NeuralBeeColonyOptimization(
        city_graph = G,
        od_matrix = od_matrix,
        neural_planner = neural_planner,
        alpha = alpha,
  # Остальные «редкие» параметры передаём, если есть.
        ** {k: opt_kwargs[k] for k in (
            "cost_calculator", "num_bees", "num_cycles",
            "num_passes", "num_iterations", "neural_bee_ratio"
        ) if k in opt_kwargs}
    )

  # 3) упаковываем результат так же, как раньше.
    best_solution = nbco.optimize(initial_routes=initial_routes, **opt_kwargs)

    return {"neural_bco_result": best_solution,
            "iteration_history": getattr(nbco, "iteration_history", []),
            "neural_statistics": nbco._get_neural_bee_statistics()
                                if hasattr(nbco, "_get_neural_bee_statistics") else {},
            "neural_performance": getattr(nbco, "neural_performance_history", []),
            "route_analysis": getattr(nbco, "route_analysis", {}),
    }

def create_initial_solution_from_opt_fin2(opt_stops_file: str = None) -> Optional[List[List[int]]]:
    """
    Создать начальное решение из оптимизированных остановок opt_fin2.py
    Специально адаптировано для структуры данных из вашего файла

    Args:
        opt_stops_file: путь к файлу с остановками (по умолчанию из config)

    Returns:
        Список маршрутов или None при ошибке
    """
    if opt_stops_file is None:
        opt_stops_file = config.files.opt_stops_file

    if not os.path.exists(opt_stops_file):
        logger.error(f"Файл {opt_stops_file} не найден")
        return None

    try:
        with open(opt_stops_file, 'rb') as f:
            opt_stops = pickle.load(f)

        logger.info(f"Загружено {len(opt_stops)} остановок из {opt_stops_file}")

  # Группируем остановки по типам как в вашем opt_fin2.py.
        key_stops = opt_stops[opt_stops['type'] == 'key']['node_id'].tolist()
        connection_stops = opt_stops[opt_stops['type'] == 'connection']['node_id'].tolist()
        ordinary_stops = opt_stops[opt_stops['type'] == 'ordinary']['node_id'].tolist()

        logger.info(f"Key stops: {len(key_stops)}, Connection: {len(connection_stops)}, "
                    f"Ordinary: {len(ordinary_stops)}")

  # Создаем маршруты, используя стратегию из статьи.
        routes = []

  # Создаем маршруты соединяющие key stops через connection stops.
        all_important_stops = key_stops + connection_stops

        if len(all_important_stops) >= config.mdp.min_route_length:
  # Создаем магистральные маршруты между важными остановками.
            num_trunk_routes = min(config.mdp.num_routes // 2, len(all_important_stops) // config.mdp.min_route_length)

            for i in range(num_trunk_routes):
                start_idx = i * (len(all_important_stops) // num_trunk_routes)
                end_idx = min(start_idx + config.mdp.max_route_length, len(all_important_stops))

                if end_idx - start_idx >= config.mdp.min_route_length:
                    trunk_route = all_important_stops[start_idx:end_idx]
                    routes.append(trunk_route)

  # Добавляем фидерные маршруты с ordinary stops.
        remaining_routes = config.mdp.num_routes - len(routes)
        if remaining_routes > 0 and ordinary_stops:
            stops_per_route = max(config.mdp.min_route_length,
                                  len(ordinary_stops) // remaining_routes)

            for i in range(remaining_routes):
                start_idx = i * stops_per_route
                end_idx = min(start_idx + stops_per_route, len(ordinary_stops))

                if end_idx > start_idx:
                    feeder_route = ordinary_stops[start_idx:end_idx]

  # Дополняем до минимальной длины если нужно.
                    while len(feeder_route) < config.mdp.min_route_length:
                        if connection_stops:
                            feeder_route.append(random.choice(connection_stops))
                        else:
                            break

  # Ограничиваем максимальной длиной.
                    if len(feeder_route) > config.mdp.max_route_length:
                        feeder_route = feeder_route[:config.mdp.max_route_length]

                    if len(feeder_route) >= config.mdp.min_route_length:
                        routes.append(feeder_route)

  # Заполняем до нужного количества маршрутов если необходимо.
        while len(routes) < config.mdp.num_routes:
            all_stops = key_stops + connection_stops + ordinary_stops
            if len(all_stops) >= config.mdp.min_route_length:
                additional_route = random.sample(all_stops,
                                                 min(config.mdp.min_route_length, len(all_stops)))
                routes.append(additional_route)
            else:
                break

        logger.info(f"Создано {len(routes)} маршрутов из остановок opt_fin2")

  # Логируем статистику маршрутов.
        route_lengths = [len(route) for route in routes]
        logger.info(f"Длины маршрутов: мин={min(route_lengths)}, макс={max(route_lengths)}, "
                    f"среднее={np.mean(route_lengths):.1f}")

        return routes

    except Exception as e:
        logger.error(f"Ошибка создания начального решения из {opt_stops_file}: {e}")
        return None

if __name__ == "__main__":
  # Демонстрация Neural BCO алгоритма.
    logger.info("Демонстрация Neural Bee Colony Optimization...")

  # Создаем простой тестовый граф.
    G = nx.grid_2d_graph(5, 5)
    G = nx.convert_node_labels_to_integers(G)

  # Добавляем атрибуты времени поездки.
    for u, v in G.edges():
        G[u][v]['travel_time'] = 60.0  # 1 минута на сегмент.

  # Создаем простую OD матрицу.
    n_nodes = len(G.nodes())
    od_matrix = np.random.randint(10, 100, size=(n_nodes, n_nodes))
    np.fill_diagonal(od_matrix, 0)

    logger.info(f"Тестовый граф: {n_nodes} узлов, {len(G.edges())} ребер")

  # Настройки для быстрого тестирования.
    config.bco.num_iterations = 30
    config.bco.num_bees = 6
    config.mdp.num_routes = 3

    try:
  # Пытаемся загрузить обученную модель (если есть).
        neural_planner = load_trained_neural_planner()
        logger.info("Обученная модель загружена успешно")

  # Запускаем Neural BCO оптимизацию БЕЗ сравнения (по умолчанию).
        results = run_neural_bco_optimization(
            city_graph=G,
            od_matrix=od_matrix,
            alpha=0.5
  # Compare_with_classical=False по умолчанию.
        )

        logger.info(f"\nРезультаты Neural BCO:")
        logger.info(f"Лучшая стоимость: {results['neural_bco_result'].cost:.3f}")
        logger.info(f"Количество маршрутов: {len(results['neural_bco_result'].routes)}")
        logger.info(f"Решение допустимо: {results['neural_bco_result'].is_feasible}")

  # Показываем детальный анализ Neural BCO.
        if 'detailed_neural_analysis' in results:
            analysis = results['detailed_neural_analysis']
            logger.info(f"\nДетальный анализ Neural BCO:")

            coverage = analysis.get('network_coverage', {})
            logger.info(f"Покрытие сети: {coverage.get('coverage_ratio', 0):.1%}")

            neural_perf = analysis.get('neural_performance', {})
            logger.info(f"Эффективность Neural Bees: {neural_perf.get('neural_bee_efficiency', 0):.1%}")
            logger.info(f"Время работы политики: {neural_perf.get('policy_time', 0):.1f}s")

            convergence = analysis.get('convergence_analysis', {})
            logger.info(f"Улучшение за итерации: {convergence.get('improvement_percent', 0):.1f}%")

  # Показываем статистику Neural Bees.
        neural_stats = results.get('neural_statistics', {})
        logger.info(f"\nСтатистика Neural Bees:")
        logger.info(f"Успешность: {neural_stats.get('avg_success_rate', 0):.1%}")
        logger.info(f"Всего вызовов политики: {neural_stats.get('total_calls', 0)}")

        logger.info(f"\nРезультаты Neural BCO:")
        logger.info(f"Лучшая стоимость: {results['neural_bco_result'].cost:.3f}")
        logger.info(f"Количество маршрутов: {len(results['neural_bco_result'].routes)}")
        logger.info(f"Решение допустимо: {results['neural_bco_result'].is_feasible}")

        if 'comparison' in results:
            comp = results['comparison']
            logger.info(f"\nСравнение с классическим BCO:")
            logger.info(f"Классический BCO: {comp['classical_bco_cost']:.3f}")
            logger.info(f"Neural BCO: {comp['neural_bco_cost']:.3f}")
            logger.info(f"Улучшение: {comp['improvement_percent']:.1f}%")

            neural_stats = comp['neural_statistics']
            logger.info(f"Успешность Neural Bees: {neural_stats.get('avg_success_rate', 0):.1%}")

  # Выводим найденные маршруты.
        logger.info("\nНайденные маршруты Neural BCO:")
        for i, route in enumerate(results['neural_bco_result'].routes):
            logger.info(f"  Маршрут {i + 1}: {route} (длина {len(route)})")

  # Тестируем создание решения из opt_fin2.
        logger.info("\nТестирование интеграции с opt_fin2...")
        init_routes = create_initial_solution_from_opt_fin2()
        if init_routes:
            logger.info(f"Создано {len(init_routes)} маршрутов из opt_fin2 данных")
        else:
            logger.info("Не удалось загрузить данные opt_fin2 (это нормально для тестирования)")

    except Exception as e:
        logger.info(f"Ошибка загрузки обученной модели: {e}")
        logger.info("Для полного тестирования Neural BCO сначала обучите нейронную сеть с помощью neural_planner.py")

  # Создаем простую заглушку для демонстрации архитектуры.
        logger.info("\nСоздание Neural BCO с неинициализированной моделью...")

  # Создаем пустую нейронную сеть для демонстрации.
        neural_planner = NeuralPlanner()

        neural_bco = NeuralBeeColonyOptimization(
            city_graph=G,
            od_matrix=od_matrix,
            neural_planner=neural_planner,
            alpha=0.5
        )

        logger.info(f"Neural BCO создан с {len(neural_bco.bees)} пчелами")
        logger.info(f"Neural bees: {len([b for b in neural_bco.bees if isinstance(b, NeuralBee)])}")
        logger.info(f"Type-2 bees: {len([b for b in neural_bco.bees if isinstance(b, Type2Bee)])}")

  # Информация об алгоритме.
        info = neural_bco.get_algorithm_info()
        logger.info(f"\nИнформация об алгоритме:")
        for key, value in info.items():
            logger.info(f"  {key}: {value}")

    logger.info("Демонстрация Neural BCO завершена!")
