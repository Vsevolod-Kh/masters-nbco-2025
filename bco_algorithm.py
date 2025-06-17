  # Bco_algorithm.py.
"""
Классический алгоритм Bee Colony Optimization для Transit Network Design
Реализует точный алгоритм из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"

BCO Parameters из статьи [[12]]:
- B = 10 (число пчел)
- NC = 2 (число циклов)
- NP = 5 (число проходов)
- I = 400 (число итераций)
- Type-1 bees: замена терминалов маршрутов
- Type-2 bees: добавление/удаление узлов (p=0.2 delete, p=0.8 add)
"""

import numpy as np
import networkx as nx
import random
import copy
from typing import List, Dict, Tuple, Optional, Set, Any
import logging
from dataclasses import dataclass
import time
from collections import defaultdict

from config import config
from cost_functions import TransitCostCalculator
from utils import RouteUtils, GraphUtils
from transit_mdp import TransitMDP

logger = logging.getLogger(__name__)

@dataclass
class BCOSolution:
    """Структура для хранения решения BCO"""
    routes: List[List[int]]
    cost: float
    is_feasible: bool
    constraint_violations: Dict[str, Any]
    generation: int
    bee_id: int

    def copy(self) -> 'BCOSolution':
        """Создать копию решения"""
        return BCOSolution(
            routes=copy.deepcopy(self.routes),
            cost=self.cost,
            is_feasible=self.is_feasible,
            constraint_violations=copy.deepcopy(self.constraint_violations),
            generation=self.generation,
            bee_id=self.bee_id
        )

@dataclass
class BCOStatistics:
    """Статистика работы BCO алгоритма"""
    iteration: int
    best_cost: float
    avg_cost: float
    worst_cost: float
    feasible_solutions: int
    type1_improvements: int
    type2_improvements: int
    recruitment_events: int
    exploration_time: float
    evaluation_time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            'iteration': self.iteration,
            'best_cost': self.best_cost,
            'avg_cost': self.avg_cost,
            'worst_cost': self.worst_cost,
            'feasible_solutions': self.feasible_solutions,
            'type1_improvements': self.type1_improvements,
            'type2_improvements': self.type2_improvements,
            'recruitment_events': self.recruitment_events,
            'exploration_time': self.exploration_time,
            'evaluation_time': self.evaluation_time
        }

class BeeAgent:
    """Базовый класс для агента-пчелы"""

    def __init__(self,
                 bee_id: int,
                 city_graph: nx.Graph,
                 cost_calculator: TransitCostCalculator,
                 od_matrix: np.ndarray):
        self.bee_id = bee_id
        self.city_graph = city_graph
        self.cost_calculator = cost_calculator
        self.od_matrix = od_matrix
        self.current_solution: Optional[BCOSolution] = None
        self.best_solution: Optional[BCOSolution] = None

  # Предвычисляем кратчайшие пути.
        self.shortest_paths = self._compute_shortest_paths()

    def _compute_shortest_paths(self) -> Dict[Tuple[int, int], List[int]]:
        """Предвычислить кратчайшие пути между всеми парами узлов"""
        try:
  # Используем правильный API для расчета длин путей.
            all_paths = dict(nx.all_pairs_dijkstra_path_length(self.city_graph, weight='travel_time'))
            paths = {}
            for source in all_paths:
                for target in all_paths[source]:
                    if source != target:
                        paths[(source, target)] = all_paths[source][target]
            return paths
        except Exception as e:
            logger.error(f"Ошибка вычисления кратчайших путей: {e}")
  # Fallback без весов если есть проблема.
            try:
                all_paths = dict(nx.all_pairs_shortest_path_length(self.city_graph))
                paths = {}
                for source in all_paths:
                    for target in all_paths[source]:
                        if source != target:
                            paths[(source, target)] = all_paths[source][target]
                return paths
            except:
                logger.error("Критическая ошибка: не удалось вычислить кратчайшие пути")
                return {}

    def explore(self, current_routes: List[List[int]], num_cycles: int) -> BCOSolution:
        """
        Базовая реализация исследования (переопределяется в наследниках)
        """
        return self.evaluate_solution(current_routes)

    def evaluate_solution(self, routes: List[List[int]]) -> BCOSolution:
        """Оценить качество решения"""
  # Проверяем ограничения.
        violations = self._check_constraints(routes)
        is_feasible = all(v == 0 for v in violations.values())

  # Вычисляем стоимость.
        try:
            cost = self.cost_calculator.calculate_cost(self.city_graph, routes, self.od_matrix)
        except Exception as e:
            logger.warning(f"Ошибка вычисления стоимости: {e}")
            cost = float('inf')
            is_feasible = False

        return BCOSolution(
            routes=copy.deepcopy(routes),
            cost=cost,
            is_feasible=is_feasible,
            constraint_violations=violations,
            generation=0,
            bee_id=self.bee_id
        )

    def _check_constraints(self, routes: List[List[int]]) -> Dict[str, int]:
        """Проверить ограничения маршрутов"""
        violations = {
            'num_routes': abs(len(routes) - config.mdp.num_routes),
            'route_length': 0,
            'cycles': 0,
            'connectivity': 0
        }

  # Проверяем длину маршрутов.
        for route in routes:
            if len(route) < config.mdp.min_route_length:
                violations['route_length'] += config.mdp.min_route_length - len(route)
            elif len(route) > config.mdp.max_route_length:
                violations['route_length'] += len(route) - config.mdp.max_route_length

  # Проверяем циклы.
        for route in routes:
            if len(set(route)) != len(route):
                violations['cycles'] += 1

  # Проверяем связность (упрощенная версия).
        if routes:
            all_nodes = set()
            for route in routes:
                all_nodes.update(route)

  # Создаем подграф из узлов маршрутов.
            if all_nodes:
                subgraph = self.city_graph.subgraph(all_nodes)
                if not nx.is_connected(subgraph):
                    violations['connectivity'] = 1

        return violations

    def get_route_selection_probabilities(self, routes: List[List[int]]) -> List[float]:
        """
        Вычислить вероятности выбора маршрутов
        Маршруты с меньшим прямым обслуживанием спроса имеют больший шанс быть выбранными
        """
        if not routes:
            return []

  # Вычисляем прямое обслуживание спроса для каждого маршрута.
        route_demands = []
        for route in routes:
            direct_demand = 0.0

  # Суммируем спрос между всеми парами узлов в маршруте.
            for i in range(len(route)):
                for j in range(i + 1, len(route)):
                    node_i = route[i]
                    node_j = route[j]

  # Находим индексы узлов в OD матрице.
                    node_list = list(self.city_graph.nodes())
                    if node_i in node_list and node_j in node_list:
                        idx_i = node_list.index(node_i)
                        idx_j = node_list.index(node_j)

                        if (idx_i < self.od_matrix.shape[0] and
                                idx_j < self.od_matrix.shape[1]):
  # Суммируем спрос в обе стороны.
                            direct_demand += self.od_matrix[idx_i, idx_j]
                            direct_demand += self.od_matrix[idx_j, idx_i]

            route_demands.append(direct_demand)

  # Инвертируем для получения вероятностей (меньший спрос = больший шанс).
        if sum(route_demands) > 0:
            max_demand = max(route_demands)
            inverted_demands = [max_demand - demand + 1 for demand in route_demands]
            total = sum(inverted_demands)
            probabilities = [demand / total for demand in inverted_demands]
        else:
  # Равномерное распределение если нет спроса.
            probabilities = [1.0 / len(routes)] * len(routes)

        return probabilities

class Type1Bee(BeeAgent):
    """
    Type-1 пчела: заменяет терминалы маршрутов [[12]]
    Выбирает случайный терминал и заменяет его на другой случайный узел,
    создавая новый маршрут как кратчайший путь между новыми терминалами
    """

    def __init__(self, bee_id: int, city_graph: nx.Graph,
                 cost_calculator: TransitCostCalculator, od_matrix: np.ndarray):
        super().__init__(bee_id, city_graph, cost_calculator, od_matrix)
        self.type_name = "Type-1"

    def explore(self, current_routes: List[List[int]], num_cycles: int = 2) -> BCOSolution:
        """
        Выполнить исследование пространства решений

        Args:
            current_routes: текущие маршруты
            num_cycles: количество циклов модификации (NC в статье)

        Returns:
            Лучшее найденное решение
        """
        best_solution = self.evaluate_solution(current_routes)
        current_solution = best_solution.copy()

        for cycle in range(num_cycles):
  # Модифицируем текущее решение.
            modified_routes = self._modify_routes(current_solution.routes)
            new_solution = self.evaluate_solution(modified_routes)

  # Принимаем только улучшения.
            if new_solution.cost < current_solution.cost:
                current_solution = new_solution

  # Обновляем лучшее решение.
                if new_solution.cost < best_solution.cost:
                    best_solution = new_solution

        return best_solution

    def _modify_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Модифицировать маршруты согласно алгоритму Type-1"""
        if not routes:
            return routes

        modified_routes = copy.deepcopy(routes)

  # Выбираем маршрут для модификации.
        probabilities = self.get_route_selection_probabilities(routes)
        route_idx = np.random.choice(len(routes), p=probabilities)

        selected_route = modified_routes[route_idx]

        if len(selected_route) < 2:
            return modified_routes

  # Выбираем случайный терминал (первый или последний узел).
        terminal_position = random.choice([0, -1])  # 0 = первый, -1 = последний.

  # Получаем все доступные узлы кроме уже используемых в маршруте.
        all_nodes = set(self.city_graph.nodes())
        used_nodes = set(selected_route)
        available_nodes = list(all_nodes - used_nodes)

        if not available_nodes:
            return modified_routes

  # Выбираем новый терминал случайно.
        new_terminal = random.choice(available_nodes)

  # Создаем новый маршрут как кратчайший путь между терминалами.
        if terminal_position == 0:
  # Заменяем первый терминал.
            start_node = new_terminal
            end_node = selected_route[-1]
        else:
  # Заменяем последний терминал.
            start_node = selected_route[0]
            end_node = new_terminal

  # Находим кратчайший путь между новыми терминалами.
        new_route = self._find_shortest_path(start_node, end_node)

        if new_route and len(new_route) >= config.mdp.min_route_length:
            modified_routes[route_idx] = new_route

        return modified_routes

    def _find_shortest_path(self, start_node: int, end_node: int) -> List[int]:
        """Найти кратчайший путь между двумя узлами"""
        if (start_node, end_node) in self.shortest_paths:
            return self.shortest_paths[(start_node, end_node)]

  # Если предвычисленного пути нет, вычисляем на лету.
        try:
            path = nx.shortest_path(self.city_graph, start_node, end_node, weight='travel_time')
            return path
        except nx.NetworkXNoPath:
            return []

class Type2Bee(BeeAgent):
    """
    Type-2 пчела: добавляет/удаляет узлы с терминалов [[12]]
    С вероятностью 0.2 удаляет терминал, с вероятностью 0.8 добавляет соседний узел
    """

    def __init__(self, bee_id: int, city_graph: nx.Graph,
                 cost_calculator: TransitCostCalculator, od_matrix: np.ndarray):
        super().__init__(bee_id, city_graph, cost_calculator, od_matrix)
        self.type_name = "Type-2"
        self.delete_probability = config.bco.type2_delete_prob  # 0.2 в статье.
        self.add_probability = config.bco.type2_add_prob  # 0.8 в статье.

    def explore(self, current_routes: List[List[int]], num_cycles: int = 2) -> BCOSolution:
        """Выполнить исследование пространства решений Type-2"""
        best_solution = self.evaluate_solution(current_routes)
        current_solution = best_solution.copy()

        for cycle in range(num_cycles):
  # Модифицируем текущее решение.
            modified_routes = self._modify_routes(current_solution.routes)
            new_solution = self.evaluate_solution(modified_routes)

  # Принимаем только улучшения.
            if new_solution.cost < current_solution.cost:
                current_solution = new_solution

                if new_solution.cost < best_solution.cost:
                    best_solution = new_solution

        return best_solution

    def _modify_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Модифицировать маршруты согласно алгоритму Type-2"""
        if not routes:
            return routes

        modified_routes = copy.deepcopy(routes)

  # Выбираем маршрут для модификации.
        probabilities = self.get_route_selection_probabilities(routes)
        route_idx = np.random.choice(len(routes), p=probabilities)

        selected_route = modified_routes[route_idx]

        if len(selected_route) < 2:
            return modified_routes

  # Выбираем случайный терминал.
        terminal_position = random.choice([0, -1])  # 0 = первый, -1 = последний.

  # Решаем: удалять или добавлять.
        action_prob = random.random()

        if action_prob < self.delete_probability:
  # Удаляем терминал.
            modified_routes[route_idx] = self._delete_terminal(selected_route, terminal_position)
        else:
  # Добавляем узел к терминалу.
            modified_routes[route_idx] = self._add_to_terminal(selected_route, terminal_position)

        return modified_routes

    def _delete_terminal(self, route: List[int], terminal_position: int) -> List[int]:
        """Удалить терминал из маршрута"""
        if len(route) <= config.mdp.min_route_length:
            return route  # Не удаляем если маршрут уже минимальной длины.

        new_route = route.copy()

        if terminal_position == 0 and len(new_route) > 1:
  # Удаляем первый узел.
            new_route = new_route[1:]
        elif terminal_position == -1 and len(new_route) > 1:
  # Удаляем последний узел.
            new_route = new_route[:-1]

        return new_route

    def _add_to_terminal(self, route: List[int], terminal_position: int) -> List[int]:
        """Добавить узел к терминалу маршрута"""
        if len(route) >= config.mdp.max_route_length:
            return route  # Не добавляем если маршрут уже максимальной длины.

  # Получаем терминальный узел.
        if terminal_position == 0:
            terminal_node = route[0]
        else:
            terminal_node = route[-1]

  # Находим соседей терминального узла.
        neighbors = list(self.city_graph.neighbors(terminal_node))

  # Исключаем узлы, уже присутствующие в маршруте.
        available_neighbors = [n for n in neighbors if n not in route]

        if not available_neighbors:
            return route

  # Выбираем случайного соседа.
        new_node = random.choice(available_neighbors)

  # Добавляем узел к соответствующему концу маршрута.
        new_route = route.copy()
        if terminal_position == 0:
            new_route.insert(0, new_node)
        else:
            new_route.append(new_node)

        return new_route

class BeeColonyOptimization:
    """
    Основной класс для алгоритма Bee Colony Optimization
    Реализует точный алгоритм из статьи [[12]]
    """

    def __init__(self,
                 city_graph: nx.Graph,
                 od_matrix: np.ndarray,
                 cost_calculator: Optional[TransitCostCalculator] = None,
                 num_bees: int = None,
                 num_cycles: int = None,
                 num_passes: int = None,
                 num_iterations: int = None,
                 alpha: float = 0.5):
        """
        Args:
            city_graph: граф дорожной сети города
            od_matrix: матрица Origin-Destination спроса
            cost_calculator: калькулятор стоимости
            num_bees: количество пчел (B в статье)
            num_cycles: количество циклов (NC в статье)
            num_passes: количество проходов (NP в статье)
            num_iterations: количество итераций (I в статье)
            alpha: параметр стоимостной функции
        """
        self.city_graph = city_graph
        self.od_matrix = od_matrix
        self.alpha = alpha

  # Параметры алгоритма из статьи.
        self.num_bees = num_bees or config.bco.num_bees
        self.num_cycles = num_cycles or config.bco.num_cycles
        self.num_passes = num_passes or config.bco.num_passes
        self.num_iterations = num_iterations or config.bco.num_iterations

  # Калькулятор стоимости.
        if cost_calculator is None:
            self.cost_calculator = TransitCostCalculator(alpha=alpha)
        else:
            self.cost_calculator = cost_calculator

  # Создаем пчел.
        self.bees = self._create_bees()

  # Статистика.
        self.statistics: List[BCOStatistics] = []
        self.best_solution: Optional[BCOSolution] = None

  # Добавляем tracking разнообразия и стагнации.
        self.diversity_history = []
        self.stagnation_counter = 0
        self.last_improvement_iteration = 0

        logger.info(f"BCO инициализирован: {self.num_bees} пчел, {self.num_iterations} итераций")

    def _calculate_swarm_diversity(self) -> float:
        """Вычислить разнообразие роя по Жаккару"""
        if len(self.bees) < 2:
            return 1.0

        all_routes = [bee.current_solution.routes for bee in self.bees]
        total_similarity = 0
        comparisons = 0

        for i in range(len(all_routes)):
            for j in range(i + 1, len(all_routes)):
                similarity = self._route_sets_similarity(all_routes[i], all_routes[j])
                total_similarity += similarity
                comparisons += 1

        avg_similarity = total_similarity / comparisons if comparisons > 0 else 0
        return 1.0 - avg_similarity  # Diversity = 1 - similarity.

    def _route_sets_similarity(self, routes1: List[List[int]], routes2: List[List[int]]) -> float:
        """Коэффициент Жаккара для сходства маршрутов"""
        set1 = set()
        set2 = set()
        for route in routes1:
            set1.update(route)
        for route in routes2:
            set2.update(route)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0

    def _create_bees(self) -> List[BeeAgent]:
        """Создать пчел (50% Type-1, 50% Type-2)"""
        bees = []

        num_type1 = self.num_bees // 2
        num_type2 = self.num_bees - num_type1

  # Type-1 пчелы.
        for i in range(num_type1):
            bee = Type1Bee(i, self.city_graph, self.cost_calculator, self.od_matrix)
            bees.append(bee)

  # Type-2 пчелы.
        for i in range(num_type1, num_type1 + num_type2):
            bee = Type2Bee(i, self.city_graph, self.cost_calculator, self.od_matrix)
            bees.append(bee)

        logger.info(f"Создано {num_type1} Type-1 пчел и {num_type2} Type-2 пчел")
        return bees

    def generate_initial_solution(self) -> List[List[int]]:
        """
        Генерировать начальное решение
        Создает простые маршруты между случайными парами узлов
        """
        routes = []
  # ➊ Пробуем новые пары.
        od_pairs = RouteUtils.select_border_od_pairs(
            self.city_graph, max_pairs=config.mdp.num_routes)
        if od_pairs:  # Нашли пары в CSV.
            for origin, dest in od_pairs:
                try:
                    path = nx.shortest_path(self.city_graph, origin, dest,
                        weight = 'travel_time')
                except nx.NetworkXNoPath:
                    continue
                if len(path) < config.mdp.min_route_length:
                    path = self._extend_path_to_min_length(path)
                elif len(path) > config.mdp.max_route_length:
                    path = path[:config.mdp.max_route_length]
                routes.append(path)
  # Если удачно набрали требуемое число – готово.
            if len(routes) == config.mdp.num_routes:
                return routes

        all_nodes = list(self.city_graph.nodes())
        for _ in range(config.mdp.num_routes):
  # Выбираем два случайных узла.
            start_node = random.choice(all_nodes)
            end_node = random.choice(all_nodes)

  # Находим кратчайший путь между ними.
            try:
                path = nx.shortest_path(self.city_graph, start_node, end_node, weight='travel_time')

  # Проверяем ограничения длины.
                if config.mdp.min_route_length <= len(path) <= config.mdp.max_route_length:
                    routes.append(path)
                elif len(path) > config.mdp.max_route_length:
  # Обрезаем до максимальной длины.
                    routes.append(path[:config.mdp.max_route_length])
                else:
  # Расширяем до минимальной длины добавлением соседних узлов.
                    extended_path = self._extend_path_to_min_length(path)
                    routes.append(extended_path)

            except nx.NetworkXNoPath:
  # Если нет пути, создаем простой маршрут из одного узла.
                routes.append([start_node] * config.mdp.min_route_length)

        return routes

    def _extend_path_to_min_length(self, path: List[int]) -> List[int]:
        """Расширить путь до минимальной длины"""
        if len(path) >= config.mdp.min_route_length:
            return path

        extended = path.copy()

  # Пытаемся добавить соседей к концам.
        while len(extended) < config.mdp.min_route_length:
  # Пробуем добавить к концу.
            last_node = extended[-1]
            neighbors = [n for n in self.city_graph.neighbors(last_node) if n not in extended]

            if neighbors:
                extended.append(random.choice(neighbors))
            else:
  # Если нет доступных соседей, дублируем узлы.
                extended.append(extended[-1])

        return extended

    def optimize(self, initial_routes: Optional[List[List[int]]] = None,
                 save_raw_routes: Optional[str] = None,
                 post_process: bool = True) -> BCOSolution:
        """
        Выполнить оптимизацию BCO

        Args:
            initial_routes: начальные маршруты (если None, генерируются автоматически)

        Returns:
            Лучшее найденное решение
        """
        logger.info("🐝 Запуск классического BCO оптимизации...")
        start_time = time.time()

  # Логируем параметры.
        logger.info("📊 Параметры BCO:")
        logger.info(f"  Пчел: {self.num_bees} (Type-1: {sum(1 for b in self.bees if isinstance(b, Type1Bee))}, "
                    f"Type-2: {sum(1 for b in self.bees if isinstance(b, Type2Bee))})")
        logger.info(f"  Итераций: {self.num_iterations}, Циклы: {self.num_cycles}, Проходы: {self.num_passes}")
        logger.info(f"  Граф: {len(self.city_graph.nodes())} узлов, {len(self.city_graph.edges())} ребер")

  # Инициализация.
        if initial_routes is None:
            initial_routes = self.generate_initial_solution()

  # Инициализируем всех пчел одинаковым решением.
        for bee in self.bees:
            bee.current_solution = bee.evaluate_solution(initial_routes)
            bee.best_solution = bee.current_solution.copy()

  # Глобальное лучшее решение.
        self.best_solution = min([bee.best_solution for bee in self.bees], key=lambda x: x.cost)

        logger.info(f"Начальная стоимость: {self.best_solution.cost:.3f}")

  # Основной цикл оптимизации.
        for iteration in range(self.num_iterations):
            iteration_start = time.time()

  # Фаза исследования.
            exploration_start = time.time()
            self._exploration_phase()
            exploration_time = time.time() - exploration_start

  # Фаза рекрутинга (каждые NP итераций).
            if (iteration + 1) % self.num_passes == 0:
                recruitment_start = time.time()
                recruitment_events = self._recruitment_phase()
                recruitment_time = time.time() - recruitment_start
            else:
                recruitment_events = 0
                recruitment_time = 0.0

  # Обновляем глобальное лучшее решение.
            current_best = min([bee.best_solution for bee in self.bees], key=lambda x: x.cost)
            if current_best.cost < self.best_solution.cost:
                self.best_solution = current_best.copy()
                self.last_improvement_iteration = iteration
                self.stagnation_counter = 0
                logger.info(f"Новое лучшее решение на итерации {iteration + 1}: {current_best.cost:.3f}")
            else:
                self.stagnation_counter += 1

  # Restart mechanism при длительной стагнации.
            if self.stagnation_counter >= 30 and iteration > 50:  # После 30 итераций без улучшения.
                diversity = self._calculate_swarm_diversity()
                if diversity < 0.1:  # Очень низкое разнообразие.
                    self._restart_worst_bees(fraction=0.25)
                    logger.info(f"RESTART: перезапущено 25% худших пчел на итерации {iteration + 1}")
                    self.stagnation_counter = 0

  # Собираем статистику.
            iteration_time = time.time() - iteration_start
            stats = self._collect_statistics(iteration, exploration_time, recruitment_time, recruitment_events)
            self.statistics.append(stats)

  # Логирование прогресса.
            if (iteration + 1) % 10 == 0:  # Каждые 10 итераций вместо 50.
                improvement = "⭐" if current_best.cost < self.best_solution.cost else "➡️"
                logger.info(f"{improvement} Итерация {iteration + 1}/{self.num_iterations}: "
                            f"лучшая={self.best_solution.cost:.3f}, "
                            f"средняя={stats.avg_cost:.3f}, "
                            f"время={iteration_time:.2f}s")

        total_time = time.time() - start_time
        logger.info(f"BCO завершен за {total_time:.1f}s. Лучшая стоимость: {self.best_solution.cost:.3f}")

        smoothed = RouteUtils.smooth_routes(self.best_solution.routes,
                    self.city_graph)

        if smoothed != self.best_solution.routes:  # Есть изменения.
            self.best_solution.routes = smoothed
            self.best_solution.cost = self.cost_calculator.calculate_cost(
                                        self.city_graph, smoothed, self.od_matrix)

  # Сохраняем «сырые» маршруты до 2-opt.

        if save_raw_routes:
            RouteUtils.save_routes(self.best_solution.routes, save_raw_routes)
            logger.info(f"🔖 Маршруты до 2-opt сохранены в {save_raw_routes}")

  # Пост-обработка (2-opt) при необходимости.

        if post_process and config.mdp.apply_post_opt:
            smoothed = RouteUtils.smooth_routes(self.best_solution.routes, self.city_graph)

            if smoothed != self.best_solution.routes:
                self.best_solution.routes = smoothed
                self.best_solution.cost = self.cost_calculator.calculate_cost(
                    self.city_graph, smoothed, self.od_matrix)
                logger.info("✅ 2-opt улучшил решение: новая стоимость "
                    f"{self.best_solution.cost:.3f}")

        return self.best_solution

    def _restart_worst_bees(self, fraction: float = 0.25):
        """Перезапустить худших пчел для восстановления diversity"""
        num_to_restart = int(len(self.bees) * fraction)

  # Сортируем пчел по стоимости (худшие первые).
        sorted_bees = sorted(self.bees, key=lambda b: b.current_solution.cost, reverse=True)

        for i in range(num_to_restart):
            bee = sorted_bees[i]
  # Генерируем новое случайное решение.
            new_routes = self.generate_initial_solution()
            bee.current_solution = bee.evaluate_solution(new_routes)
            bee.best_solution = bee.current_solution.copy()

        logger.info(f"Перезапущено {num_to_restart} пчел для восстановления diversity")

    def _exploration_phase(self):
        """Фаза исследования: каждая пчела исследует пространство решений"""
        for bee in self.bees:
  # Исследование с текущего решения.
            new_solution = bee.explore(bee.current_solution.routes, self.num_cycles)

  # Обновляем текущее решение если найдено улучшение.
            if new_solution.cost < bee.current_solution.cost:
                bee.current_solution = new_solution

  # Обновляем лучшее решение пчелы.
                if new_solution.cost < bee.best_solution.cost:
                    bee.best_solution = new_solution.copy()

    def _recruitment_phase(self) -> int:
        """
        Фаза рекрутинга: пчелы-следователи копируют решения пчел-рекрутеров

        Returns:
            Количество событий рекрутинга
        """
        diversity = self._calculate_swarm_diversity()
        self.diversity_history.append(diversity)

  # Если diversity низкое, уменьшаем recruitment агрессивность.
        diversity_factor = max(0.3, diversity)  # Минимум 30% recruitment.

        recruitment_events = 0
        solutions_costs = [bee.current_solution.cost for bee in self.bees]

  # Назначаем роли пчелам на основе качества их решений.
        solutions_costs = [bee.current_solution.cost for bee in self.bees]

  # Инвертируем стоимости для получения весов (меньшая стоимость = больший вес).
        max_cost = max(solutions_costs) if solutions_costs else 1.0
        weights = [max_cost - cost + 1e-6 for cost in solutions_costs]
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]

        recruitment_events = 0

  # Каждая пчела с некоторой вероятностью становится следователем.
        for follower_bee in self.bees:
  # Базовая вероятность стать follower.
            base_follower_prob = 1.0 - (weights[follower_bee.bee_id] / max(weights))

  # Модифицируем с учетом diversity.
            follower_prob = base_follower_prob * diversity_factor

            if random.random() < follower_prob:
                recruiter_idx = np.random.choice(len(self.bees), p=probabilities)
                recruiter_bee = self.bees[recruiter_idx]

                if recruiter_bee.bee_id != follower_bee.bee_id:
                    follower_bee.current_solution = recruiter_bee.current_solution.copy()
                    recruitment_events += 1

        return recruitment_events

    def _collect_statistics(self, iteration: int, exploration_time: float,
                            evaluation_time: float, recruitment_events: int) -> BCOStatistics:
        """Собрать статистику текущей итерации"""
        costs = [bee.current_solution.cost for bee in self.bees]
        feasible_solutions = sum(1 for bee in self.bees if bee.current_solution.is_feasible)

        return BCOStatistics(
            iteration=iteration + 1,
            best_cost=min(costs),
            avg_cost=np.mean(costs),
            worst_cost=max(costs),
            feasible_solutions=feasible_solutions,
            type1_improvements=0,  # TODO: подсчитывать улучшения по типам.
            type2_improvements=0,
            recruitment_events=recruitment_events,
            exploration_time=exploration_time,
            evaluation_time=evaluation_time
        )

    def get_convergence_data(self) -> Dict[str, List[float]]:
        """Получить данные сходимости для анализа"""
        return {
            'iterations': [s.iteration for s in self.statistics],
            'best_costs': [s.best_cost for s in self.statistics],
            'avg_costs': [s.avg_cost for s in self.statistics],
            'worst_costs': [s.worst_cost for s in self.statistics],
            'feasible_ratios': [s.feasible_solutions / self.num_bees for s in self.statistics]
        }

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Получить информацию об алгоритме"""
        return {
            'algorithm': 'Bee Colony Optimization',
            'num_bees': self.num_bees,
            'num_cycles': self.num_cycles,
            'num_passes': self.num_passes,
            'num_iterations': self.num_iterations,
            'alpha': self.alpha,
            'type1_bees': sum(1 for bee in self.bees if isinstance(bee, Type1Bee)),
            'type2_bees': sum(1 for bee in self.bees if isinstance(bee, Type2Bee)),
            'city_nodes': len(self.city_graph.nodes()),
            'city_edges': len(self.city_graph.edges())
        }

def create_initial_solution_from_stops(opt_stops_file: str = 'opt_stops.pkl') -> Optional[List[List[int]]]:
    """
    Создать начальное решение из оптимизированных остановок opt_fin2.py

    Args:
        opt_stops_file: путь к файлу с остановками

    Returns:
        Список маршрутов или None при ошибке
    """
    try:
        import pickle

        with open(opt_stops_file, 'rb') as f:
            opt_stops = pickle.load(f)

        logger.info(f"Загружено {len(opt_stops)} остановок из {opt_stops_file}")

  # Группируем остановки по типам.
        key_stops = opt_stops[opt_stops['type'] == 'key']['node_id'].tolist()
        connection_stops = opt_stops[opt_stops['type'] == 'connection']['node_id'].tolist()
        ordinary_stops = opt_stops[opt_stops['type'] == 'ordinary']['node_id'].tolist()

  # Создаем простые маршруты.
        routes = []
        all_stops = key_stops + connection_stops + ordinary_stops

  # Распределяем остановки по маршрутам.
        stops_per_route = len(all_stops) // config.mdp.num_routes

        for i in range(config.mdp.num_routes):
            start_idx = i * stops_per_route
            end_idx = start_idx + stops_per_route

            if i == config.mdp.num_routes - 1:  # Последний маршрут получает оставшиеся остановки.
                route_stops = all_stops[start_idx:]
            else:
                route_stops = all_stops[start_idx:end_idx]

  # Обеспечиваем минимальную длину маршрута.
            while len(route_stops) < config.mdp.min_route_length:
                route_stops.extend(route_stops[:config.mdp.min_route_length - len(route_stops)])

  # Ограничиваем максимальную длину.
            if len(route_stops) > config.mdp.max_route_length:
                route_stops = route_stops[:config.mdp.max_route_length]

            routes.append(route_stops)

        logger.info(f"Создано {len(routes)} маршрутов из остановок opt_fin2")
        return routes

    except Exception as e:
        logger.error(f"Ошибка создания начального решения из {opt_stops_file}: {e}")
        return None

def run_bco_optimization(
        G,
        od_matrix,
        alpha: float = 0.5,
        initial_routes: Optional[List[List[int]]] = None,
        save_raw_routes: Optional[str] = None,
        post_process: bool = True,
):
    """
    Classic (не-нейронная) Bee Colony optimisation.

    Parameters
    ----------
    G : networkx.Graph
        Дорожный граф.
    od_matrix : np.ndarray
        Матрица спроса O-D.
    alpha : float, optional
        Вес пассажирской части стоимости.
    initial_routes : list[list[int]] | None, optional
        Стартовое решение (можно None).
    save_raw_routes : str | None, optional
        Путь, куда сохранить маршруты до применения 2-opt.
    post_process : bool, optional
        Применять ли 2-/3-opt в конце оптимизации.

    Returns
    -------
    BCOSolution
        Лучшее найденное решение.
    """
    bco = BeeColonyOptimization(G, od_matrix, alpha)
    return bco.optimize(
        initial_routes=initial_routes,
        save_raw_routes=save_raw_routes,
        post_process=post_process,
    )

if __name__ == "__main__":
  # Демонстрация BCO алгоритма.
    print("Демонстрация Bee Colony Optimization для транзитных сетей...")

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

    print(f"Тестовый граф: {n_nodes} узлов, {len(G.edges())} ребер")

  # Настройки для быстрого тестирования.
    config.bco.num_iterations = 50
    config.bco.num_bees = 6
    config.mdp.num_routes = 3

  # Создаем BCO.
    bco = BeeColonyOptimization(G, od_matrix, alpha=0.5)

    print(f"BCO создан: {bco.num_bees} пчел, {bco.num_iterations} итераций")
    print(f"Type-1 пчел: {sum(1 for bee in bco.bees if isinstance(bee, Type1Bee))}")
    print(f"Type-2 пчел: {sum(1 for bee in bco.bees if isinstance(bee, Type2Bee))}")

  # Запускаем оптимизацию.
    best_solution = bco.optimize()

    print(f"\nРезультаты BCO:")
    print(f"Лучшая стоимость: {best_solution.cost:.3f}")
    print(f"Количество маршрутов: {len(best_solution.routes)}")
    print(f"Решение допустимо: {best_solution.is_feasible}")

  # Выводим маршруты.
    print("\nНайденные маршруты:")
    for i, route in enumerate(best_solution.routes):
        print(f"  Маршрут {i + 1}: {route} (длина {len(route)})")

  # Выводим нарушения ограничений.
    if not best_solution.is_feasible:
        print("\nНарушения ограничений:")
        for constraint, violation in best_solution.constraint_violations.items():
            if violation > 0:
                print(f"  {constraint}: {violation}")

  # Анализ сходимости.
    convergence_data = bco.get_convergence_data()

    print(f"\nСходимость алгоритма:")
    print(f"Начальная стоимость: {convergence_data['best_costs'][0]:.3f}")
    print(f"Финальная стоимость: {convergence_data['best_costs'][-1]:.3f}")
    print(f"Улучшение: {(convergence_data['best_costs'][0] - convergence_data['best_costs'][-1]):.3f}")

  # Тестируем создание решения из opt_fin2.
    print("\nТестирование интеграции с opt_fin2...")
    init_routes = create_initial_solution_from_stops()
    if init_routes:
        print(f"Создано {len(init_routes)} маршрутов из opt_fin2 данных")
    else:
        print("Не удалось загрузить данные opt_fin2 (это нормально для тестирования)")

    print("Демонстрация BCO завершена успешно!")

