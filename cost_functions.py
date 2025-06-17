  # Cost_functions.py.
"""
Функции стоимости для Neural Bee Colony Optimization в транзитных сетях
Реализует функцию стоимости из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"

C(C, R) = α * wp * Cp(C, R) + (1-α) * wo * Co(C, R) + β * Cc(C, R)

где:
- Cp: пассажирская стоимость (среднее время поездки)
- Co: операторская стоимость (общее время работы маршрутов)
- Cc: штрафы за нарушение ограничений
- α: баланс между пассажирской и операторской стоимостью [0,1]
- β: вес штрафов за ограничения (β=5 в статье)
- wp, wo: веса нормализации
"""

import numpy as np
import networkx as nx
import geopandas as gpd
from typing import List, Dict, Tuple, Optional, Set, Union
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
import logging
from dataclasses import dataclass
import time

from config import config
from utils import GraphUtils, TransitMetrics

logger = logging.getLogger(__name__)


@dataclass
class CostComponents:
    """Структура для хранения компонентов функции стоимости"""
    passenger_cost: float
    operator_cost: float
    constraint_cost: float
    total_cost: float

  # Дополнительные метрики для анализа.
    avg_travel_time: float
    total_route_time: float
    connectivity_ratio: float
    length_violations: int
    transfer_count: int

    def to_dict(self) -> Dict[str, float]:
        """Преобразовать в словарь для логирования"""
        return {
            'passenger_cost': self.passenger_cost,
            'operator_cost': self.operator_cost,
            'constraint_cost': self.constraint_cost,
            'total_cost': self.total_cost,
            'avg_travel_time': self.avg_travel_time,
            'total_route_time': self.total_route_time,
            'connectivity_ratio': self.connectivity_ratio,
            'length_violations': self.length_violations,
            'transfer_count': self.transfer_count
        }


class TransitCostCalculator:
    """
    Калькулятор стоимости транзитных сетей
    Реализует функцию стоимости из статьи [[12]]
    """

    def __init__(self,
                 alpha: float = 0.5,
                 beta: float = 5.0,
                 transfer_penalty_minutes: float = 5.0,
                 enable_caching: bool = True):
        """
        Args:
            alpha: вес пассажирской стоимости [0,1]
            beta: вес штрафов за ограничения (β=5 в статье)
            transfer_penalty_minutes: штраф за пересадку в минутах (5 в статье)
            enable_caching: кэшировать вычисления кратчайших путей
        """
        self.alpha = alpha
        self.beta = beta
        self.transfer_penalty = transfer_penalty_minutes * 60  # В секундах.
        self.enable_caching = enable_caching

  # Кэш для кратчайших путей в базовом графе.
        self._shortest_paths_cache = {}
        self._shortest_times_cache = {}
        self._transit_sp_cache = {}

  # Нормализационные веса (будут вычислены при первом использовании).
        self._wp = None
        self._wo = None
        self._max_travel_time = None

    def calculate_cost(self,
                       city_graph: nx.Graph,
                       routes: List[List[int]],
                       od_matrix: np.ndarray,
                       alpha: Optional[float] = None,
                       return_components: bool = False) -> Union[float, CostComponents]:
        """
        Вычислить общую стоимость транзитной сети

        Args:
            city_graph: граф дорожной сети города
            routes: список маршрутов [[node_ids]]
            od_matrix: матрица Origin-Destination спроса (N x N)
            alpha: переопределить вес пассажирской стоимости
            return_components: вернуть детализированные компоненты стоимости

        Returns:
            Общая стоимость или CostComponents с деталями
        """
        if alpha is None:
            alpha = self.alpha

        start_time = time.time()

  # Валидация входных данных.
        if not self._validate_inputs(city_graph, routes, od_matrix):
            if return_components:
                return self._create_invalid_cost_components()
            return float('inf')

  # Вычисляем нормализационные веса.
        if self._wp is None or self._wo is None:
            self._compute_normalization_weights(city_graph)

  # Вычисляем компоненты стоимости.
        passenger_cost = self._calculate_passenger_cost(city_graph, routes, od_matrix)
        operator_cost = self._calculate_operator_cost(city_graph, routes)
        constraint_cost = self._calculate_constraint_cost(city_graph, routes)

  # Нормализуем и комбинируем компоненты [[12]].
        normalized_passenger = self._wp * passenger_cost
        normalized_operator = self._wo * operator_cost

        total_cost = (alpha * normalized_passenger +
                      (1.0 - alpha) * normalized_operator +
                      self.beta * constraint_cost)

        logger.debug(f"Cost calculation took {time.time() - start_time:.3f}s")

        if return_components:
            return self._create_cost_components(
                passenger_cost, operator_cost, constraint_cost, total_cost,
                city_graph, routes, od_matrix
            )

        return total_cost

    def _validate_inputs(self,
                         city_graph: nx.Graph,
                         routes: List[List[int]],
                         od_matrix: np.ndarray) -> bool:
        """Валидация входных данных"""
        if len(city_graph.nodes()) == 0:
            logger.error("Пустой граф города")
            return False

        if not isinstance(routes, list):
            logger.error("Маршруты должны быть списком")
            return False

        if len(routes) == 0:
            logger.warning("Пустой список маршрутов")
            return True  # Технически валидно, но стоимость будет бесконечной.

  # Проверяем, что все узлы маршрутов существуют в графе.
        all_route_nodes = set()
        for route in routes:
            if not isinstance(route, list) or len(route) == 0:
                continue
            all_route_nodes.update(route)

        if not all_route_nodes.issubset(set(city_graph.nodes())):
            logger.error("Некоторые узлы маршрутов не существуют в графе города")
            return False

  # Проверяем размер OD матрицы.
        n_nodes = len(city_graph.nodes())
        if od_matrix.shape != (n_nodes, n_nodes):
            logger.error(f"Неправильный размер OD матрицы: {od_matrix.shape}, ожидается ({n_nodes}, {n_nodes})")
            return False

        return True

    def _compute_normalization_weights(self, city_graph: nx.Graph):
        """
        Вычислить нормализационные веса wp и wo [[12]]
        wp = 1 / max_travel_time
        wo = 1 / (3 * S * max_travel_time)
        """
  # Вычисляем максимальное время поездки в базовом графе.
        if self.enable_caching and 'all_pairs' in self._shortest_times_cache:
            shortest_times = self._shortest_times_cache['all_pairs']
        else:
            shortest_times = dict(nx.all_pairs_dijkstra_path_length(
                city_graph, weight='travel_time'
            ))
            if self.enable_caching:
                self._shortest_times_cache['all_pairs'] = shortest_times

        max_time = 0.0
        for source in shortest_times:
            for target, time_val in shortest_times[source].items():
                if source != target and time_val < float('inf'):
                    max_time = max(max_time, time_val)

        if max_time == 0:
            logger.warning("Максимальное время поездки равно 0, используем значение по умолчанию")
            max_time = 3600.0  # 1 час по умолчанию.

        max_time_min = max_time / 60.0
        self._max_travel_time = max_time_min
        self._wp = 1.0 / max_time_min
        self._wo = 1.0 / (3 * config.mdp.num_routes * max_time_min)

        logger.debug(f"Нормализационные веса: wp={self._wp:.6f}, wo={self._wo:.6f}, max_time={max_time:.1f}s")

    def _calculate_passenger_cost(self,
                                  city_graph: nx.Graph,
                                  routes: List[List[int]],
                                  od_matrix: np.ndarray) -> float:
        """
        Вычислить пассажирскую стоимость Cp [[12]]
        Среднее взвешенное время поездки с учетом пересадок
        """
        if not routes or len(routes) == 0:
            return float('inf')

  # Создаем граф транзитной сети.
        transit_graph = self._create_transit_graph(city_graph, routes)

        if len(transit_graph.nodes()) == 0:
            return float('inf')

        total_weighted_time = 0.0
        total_demand = 0.0

        node_list = list(city_graph.nodes())
        n_nodes = len(node_list)

  # Быстрое вычисление кратчайших путей в транзитной сети.
  # --- all-pairs (seconds) с умным кэшем -----------------------------.
        routes_key = tuple(sorted(tuple(r) for r in routes))

        if self.enable_caching and routes_key in self._transit_sp_cache:
            transit_shortest_paths = self._transit_sp_cache[routes_key]
        else:
            try:
                transit_shortest_paths = dict(nx.all_pairs_dijkstra_path_length(
                    transit_graph, weight='travel_time'))
                if self.enable_caching:
                    self._transit_sp_cache[routes_key] = transit_shortest_paths
            except (nx.NetworkXError, nx.NetworkXNoPath):
                logger.warning("Ошибка вычисления кратчайших путей в транзитной сети")
                return float('inf')

  # Вычисляем взвешенное среднее время поездки.
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and od_matrix[i, j] > 0:
                    origin = node_list[i]
                    destination = node_list[j]

  # Получаем время поездки в транзитной сети.
                    if (origin in transit_shortest_paths and
                            destination in transit_shortest_paths[origin]):
                        travel_time = transit_shortest_paths[origin][destination]
                    else:
  # Если нет соединения в транзитной сети, применяем большой штраф.
                        travel_time = self._max_travel_time * 10  # Большой штраф.

                    total_weighted_time += od_matrix[i, j] * travel_time
                    total_demand += od_matrix[i, j]

        if total_demand == 0:
            return 0.0

        avg_travel_time = total_weighted_time / total_demand
        return avg_travel_time / 60.0  # Возвращаем в минутах для согласованности с статьей.

    def _calculate_operator_cost(self,
                                 city_graph: nx.Graph,
                                 routes: List[List[int]]) -> float:
        """
        Вычислить операторскую стоимость Co [[12]]
        Общее время работы всех маршрутов (в оба направления)
        """
        total_route_time = 0.0

        for route in routes:
            if len(route) < 2:
                continue

            route_time = self._calculate_route_travel_time(city_graph, route)
  # Умножаем на 2 для движения в оба направления [[12]].
            total_route_time += 2 * route_time

        return total_route_time / 60.0  # Возвращаем в минутах.

    def _calculate_route_travel_time(self,
                                     city_graph: nx.Graph,
                                     route: List[int]) -> float:
        """Вычислить время проезда маршрута"""
        if len(route) < 2:
            return 0.0

        total_time = 0.0

        for i in range(len(route) - 1):
            current_node = route[i]
            next_node = route[i + 1]

  # Если есть прямое ребро, используем его.
            if city_graph.has_edge(current_node, next_node):
                edge_data = city_graph[current_node][next_node]
                segment_time = edge_data.get('travel_time', 0)
            else:
  # Используем кратчайший путь.
                try:
                    segment_time = nx.shortest_path_length(
                        city_graph, current_node, next_node, weight='travel_time'
                    )
                except nx.NetworkXNoPath:
  # Если нет пути, применяем большой штраф.
                    segment_time = self._max_travel_time * 5
                    logger.warning(f"Нет пути между узлами {current_node} и {next_node} в маршруте")

            total_time += segment_time

        return total_time

    def _calculate_constraint_cost(self,
                                   city_graph: nx.Graph,
                                   routes: List[List[int]]) -> float:
        """
        Вычислить штрафы за нарушение ограничений Cc [[12]]
        Включает нарушения связности и длины маршрутов
        """
        if not routes:
            return 1.0  # Максимальный штраф за отсутствие маршрутов.

        constraint_violations = 0.0

  # 1. Штрафы за нарушение длины маршрутов.
        length_violations = 0
        for route in routes:
            route_length = len(route)
            if route_length < config.mdp.min_route_length:
                length_violations += config.mdp.min_route_length - route_length
            elif route_length > config.mdp.max_route_length:
                length_violations += route_length - config.mdp.max_route_length

  # Нормализуем по количеству маршрутов.
        if len(routes) > 0:
            constraint_violations += length_violations / len(routes)

  # 2. Штрафы за несвязность сети.
        connectivity_penalty = self._calculate_connectivity_penalty(city_graph, routes)
        constraint_violations += connectivity_penalty

  # 3. Штрафы за циклы в маршрутах.
        cycle_penalty = self._calculate_cycle_penalty(routes)
        constraint_violations += cycle_penalty

        return min(constraint_violations, 1.0)  # Ограничиваем максимальным значением 1.0.

    def _calculate_connectivity_penalty(self,
                                        city_graph: nx.Graph,
                                        routes: List[List[int]]) -> float:
        """Вычислить штраф за несвязность транзитной сети"""
        if not routes:
            return 1.0

  # Создаем граф транзитной сети без штрафов за пересадки.
        transit_graph = self._create_transit_graph(city_graph, routes, transfer_penalty=0)

        if len(transit_graph.nodes()) == 0:
            return 1.0

  # Считаем долю узлов, достижимых в транзитной сети.
        total_city_nodes = len(city_graph.nodes())
        total_possible_pairs = total_city_nodes * (total_city_nodes - 1)

        if total_possible_pairs == 0:
            return 0.0

        connected_pairs = 0

  # Для каждого узла в оригинальном графе проверяем достижимость в транзитной сети.
        for source in city_graph.nodes():
            if source in transit_graph.nodes():
                try:
                    reachable = nx.single_source_shortest_path_length(transit_graph, source)
  # Считаем только узлы из оригинального графа.
                    reachable_city_nodes = [node for node in reachable.keys()
                                            if node in city_graph.nodes() and node != source]
                    connected_pairs += len(reachable_city_nodes)
                except nx.NetworkXError:
                    continue

        connectivity_ratio = connected_pairs / total_possible_pairs
        return 1.0 - connectivity_ratio

    def _calculate_cycle_penalty(self, routes: List[List[int]]) -> float:
        """Вычислить штраф за циклы в маршрутах"""
        cycle_violations = 0

        for route in routes:
            if len(route) != len(set(route)):
  # Есть повторяющиеся узлы = цикл.
                cycle_violations += 1

  # Нормализуем по количеству маршрутов.
        if len(routes) > 0:
            return cycle_violations / len(routes)
        return 0.0

    def _create_transit_graph(self,
                              city_graph: nx.Graph,
                              routes: List[List[int]],
                              transfer_penalty: Optional[float] = None) -> nx.Graph:
        """
        Создать граф транзитной сети из маршрутов

        Args:
            city_graph: исходный граф города
            routes: список маршрутов
            transfer_penalty: штраф за пересадку (если None, используется self.transfer_penalty)

        Returns:
            Граф транзитной сети с возможностями пересадок
        """
        if transfer_penalty is None:
            transfer_penalty = self.transfer_penalty

        return TransitMetrics._create_transit_graph(city_graph, routes, transfer_penalty)

    def _create_cost_components(self,
                                passenger_cost: float,
                                operator_cost: float,
                                constraint_cost: float,
                                total_cost: float,
                                city_graph: nx.Graph,
                                routes: List[List[int]],
                                od_matrix: np.ndarray) -> CostComponents:
        """Создать объект с детализированными компонентами стоимости"""

  # Дополнительные метрики для анализа.
        avg_travel_time = passenger_cost  # Уже в минутах.
        total_route_time = operator_cost  # Уже в минутах.

  # Подсчет пересадок (примерная оценка).
        transfer_count = self._estimate_transfers(city_graph, routes, od_matrix)

  # Связность сети.
        connectivity_ratio = 1.0 - self._calculate_connectivity_penalty(city_graph, routes)

  # Нарушения длины маршрутов.
        length_violations = 0
        for route in routes:
            route_length = len(route)
            if (route_length < config.mdp.min_route_length or
                    route_length > config.mdp.max_route_length):
                length_violations += 1

        return CostComponents(
            passenger_cost=passenger_cost,
            operator_cost=operator_cost,
            constraint_cost=constraint_cost,
            total_cost=total_cost,
            avg_travel_time=avg_travel_time,
            total_route_time=total_route_time,
            connectivity_ratio=connectivity_ratio,
            length_violations=length_violations,
            transfer_count=transfer_count
        )

    def _create_invalid_cost_components(self) -> CostComponents:
        """Создать объект стоимости для невалидных входных данных"""
        return CostComponents(
            passenger_cost=float('inf'),
            operator_cost=float('inf'),
            constraint_cost=1.0,
            total_cost=float('inf'),
            avg_travel_time=float('inf'),
            total_route_time=float('inf'),
            connectivity_ratio=0.0,
            length_violations=999,
            transfer_count=0
        )

    def _estimate_transfers(self,
                            city_graph: nx.Graph,
                            routes: List[List[int]],
                            od_matrix: np.ndarray) -> int:
        """Примерная оценка общего количества пересадок"""
        if not routes:
            return 0

  # Создаем граф транзитной сети.
        transit_graph = self._create_transit_graph(city_graph, routes)

        total_transfers = 0
        total_demand = od_matrix.sum()

        if total_demand == 0:
            return 0

  # Примерная оценка: считаем среднее количество пересадок на поездку.
  # Это упрощенная версия, точный подсчет требует анализа всех путей.

        node_list = list(city_graph.nodes())
        sample_pairs = min(100, len(node_list) * len(node_list))  # Сэмплируем для производительности.

        sampled_transfers = 0
        sampled_demand = 0

        import random
        random.seed(42)  # Для воспроизводимости.

        for _ in range(sample_pairs):
            i = random.randint(0, len(node_list) - 1)
            j = random.randint(0, len(node_list) - 1)

            if i != j and od_matrix[i, j] > 0:
                origin = node_list[i]
                destination = node_list[j]

                try:
  # Получаем путь в транзитной сети.
                    path = nx.shortest_path(transit_graph, origin, destination, weight='travel_time')

  # Считаем пересадки как переходы между разными маршрутами.
                    transfers_in_path = self._count_transfers_in_path(path, routes)

                    sampled_transfers += transfers_in_path * od_matrix[i, j]
                    sampled_demand += od_matrix[i, j]

                except nx.NetworkXNoPath:
                    continue

        if sampled_demand > 0:
            avg_transfers_per_trip = sampled_transfers / sampled_demand
            total_transfers = int(avg_transfers_per_trip * total_demand)

        return total_transfers

    def _count_transfers_in_path(self, path: List[int], routes: List[List[int]]) -> int:
        """Подсчитать количество пересадок в пути"""
        if len(path) < 2:
            return 0

  # Определяем, какие маршруты покрывают каждый сегмент пути.
        current_routes = set()
        transfers = 0

        for i in range(len(path) - 1):
            segment_start = path[i]
            segment_end = path[i + 1]

  # Находим маршруты, которые содержат этот сегмент.
            segment_routes = set()
            for route_idx, route in enumerate(routes):
                if segment_start in route and segment_end in route:
  # Проверяем, что узлы следуют друг за другом в маршруте.
                    start_pos = route.index(segment_start)
                    end_pos = route.index(segment_end)
                    if abs(start_pos - end_pos) == 1:
                        segment_routes.add(route_idx)

            if i == 0:
                current_routes = segment_routes
            else:
  # Если нет пересечения с предыдущими маршрутами, это пересадка.
                if not current_routes.intersection(segment_routes):
                    transfers += 1
                current_routes = segment_routes

        return transfers

    def get_cost_breakdown(self,
                           city_graph: nx.Graph,
                           routes: List[List[int]],
                           od_matrix: np.ndarray,
                           alpha: Optional[float] = None) -> Dict[str, float]:
        """
        Получить детализированную разбивку стоимости для анализа

        Returns:
            Словарь с компонентами стоимости и метриками
        """
        components = self.calculate_cost(
            city_graph, routes, od_matrix, alpha, return_components=True
        )

        if alpha is None:
            alpha = self.alpha

        result = components.to_dict()
        result.update({
            'alpha': alpha,
            'beta': self.beta,
            'wp': self._wp or 0.0,
            'wo': self._wo or 0.0,
            'max_travel_time': self._max_travel_time or 0.0,
            'num_routes': len(routes),
            'total_route_nodes': sum(len(route) for route in routes),
            'avg_route_length': np.mean([len(route) for route in routes]) if routes else 0.0
        })

        return result

    def clear_cache(self):
        """Очистить кэш кратчайших путей"""
        self._shortest_paths_cache.clear()
        self._shortest_times_cache.clear()
        self._transit_sp_cache.clear()
        logger.debug("Кэш кратчайших путей очищен")


class CostFunctionFactory:
    """Фабрика для создания функций стоимости с разными параметрами"""

    @staticmethod
    def create_passenger_focused(transfer_penalty: float = 5.0) -> TransitCostCalculator:
        """Создать функцию стоимости, ориентированную на пассажиров (α=1.0)"""
        return TransitCostCalculator(
            alpha=1.0,
            beta=5.0,
            transfer_penalty_minutes=transfer_penalty
        )

    @staticmethod
    def create_operator_focused(beta: float = 5.0) -> TransitCostCalculator:
        """Создать функцию стоимости, ориентированную на операторов (α=0.0)"""
        return TransitCostCalculator(
            alpha=0.0,
            beta=beta,
            transfer_penalty_minutes=5.0
        )

    @staticmethod
    def create_balanced(alpha: float = 0.5, beta: float = 5.0) -> TransitCostCalculator:
        """Создать сбалансированную функцию стоимости"""
        return TransitCostCalculator(
            alpha=alpha,
            beta=beta,
            transfer_penalty_minutes=5.0
        )

    @staticmethod
    def create_from_config(alpha: float,
                           beta: Optional[float] = None,
                           transfer_penalty: Optional[float] = None) -> TransitCostCalculator:
        """Создать функцию стоимости из конфигурации"""
        return TransitCostCalculator(
            alpha=alpha,
            beta=beta or config.mdp.constraint_penalty_weight,
            transfer_penalty_minutes=transfer_penalty or config.mdp.transfer_penalty
        )


def evaluate_cost_sensitivity(city_graph: nx.Graph,
                              routes: List[List[int]],
                              od_matrix: np.ndarray,
                              alpha_range: np.ndarray = None) -> Dict[float, CostComponents]:
    """
    Анализ чувствительности стоимости к параметру α

    Args:
        city_graph: граф города
        routes: маршруты для оценки
        od_matrix: матрица спроса
        alpha_range: диапазон значений α для тестирования

    Returns:
        Словарь {alpha: CostComponents} с результатами для каждого α
    """
    if alpha_range is None:
        alpha_range = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, ..., 1.0].

    results = {}

    for alpha in alpha_range:
        calculator = TransitCostCalculator(alpha=alpha)
        components = calculator.calculate_cost(
            city_graph, routes, od_matrix, return_components=True
        )
        results[alpha] = components

    return results


def compare_route_sets(city_graph: nx.Graph,
                       route_sets: Dict[str, List[List[int]]],
                       od_matrix: np.ndarray,
                       alpha: float = 0.5) -> Dict[str, CostComponents]:
    """
    Сравнить несколько наборов маршрутов

    Args:
        city_graph: граф города
        route_sets: словарь {название: список_маршрутов}
        od_matrix: матрица спроса
        alpha: вес пассажирской стоимости

    Returns:
        Словарь {название: CostComponents} с результатами сравнения
    """
    calculator = TransitCostCalculator(alpha=alpha)
    results = {}

    for name, routes in route_sets.items():
        try:
            components = calculator.calculate_cost(
                city_graph, routes, od_matrix, return_components=True
            )
            results[name] = components
            logger.info(f"Оценен набор маршрутов '{name}': общая стоимость = {components.total_cost:.3f}")
        except Exception as e:
            logger.error(f"Ошибка оценки набора '{name}': {e}")
            results[name] = calculator._create_invalid_cost_components()

    return results


if __name__ == "__main__":
  # Демонстрация и тестирование функций стоимости.
    print("Тестирование функций стоимости для Neural BCO...")

  # Создаем простой тестовый граф.
    import networkx as nx
    from utils import DataUtils

  # Тестовый граф 4x4 сетка.
    G = nx.grid_2d_graph(4, 4)
    G = nx.convert_node_labels_to_integers(G)

  # Добавляем атрибуты времени поездки.
    for u, v in G.edges():
        G[u][v]['travel_time'] = 60.0  # 1 минута на сегмент.

  # Создаем тестовые маршруты.
    test_routes = [
        [0, 1, 2, 3],  # Горизонтальный маршрут.
        [0, 4, 8, 12],  # Вертикальный маршрут.
        [5, 6, 7, 11, 15]  # Диагональный маршрут.
    ]

  # Создаем простую OD матрицу.
    n_nodes = len(G.nodes())
    od_matrix = np.random.randint(10, 100, size=(n_nodes, n_nodes))
    np.fill_diagonal(od_matrix, 0)

    print(f"Тестовый граф: {n_nodes} узлов, {len(G.edges())} ребер")
    print(f"Тестовые маршруты: {len(test_routes)} маршрутов")

  # Тестируем разные функции стоимости.
    calculators = {
        'Пассажиро-ориентированная (α=1.0)': CostFunctionFactory.create_passenger_focused(),
        'Оператор-ориентированная (α=0.0)': CostFunctionFactory.create_operator_focused(),
        'Сбалансированная (α=0.5)': CostFunctionFactory.create_balanced(),
    }

    print("\nСравнение функций стоимости:")
    print("-" * 60)

    for name, calculator in calculators.items():
        components = calculator.calculate_cost(G, test_routes, od_matrix, return_components=True)
        print(f"\n{name}:")
        print(f"  Общая стоимость: {components.total_cost:.3f}")
        print(f"  Пассажирская: {components.passenger_cost:.3f} мин")
        print(f"  Операторская: {components.operator_cost:.3f} мин")
        print(f"  Штрафы: {components.constraint_cost:.3f}")
        print(f"  Связность: {components.connectivity_ratio:.1%}")
        print(f"  Нарушения длины: {components.length_violations}")

  # Тест анализа чувствительности.
    print("\nАнализ чувствительности к параметру α:")
    print("-" * 60)

    sensitivity_results = evaluate_cost_sensitivity(G, test_routes, od_matrix)

    for alpha, components in sensitivity_results.items():
        print(f"α={alpha:.1f}: общая={components.total_cost:.3f}, "
              f"пассажир={components.passenger_cost:.1f}, "
              f"оператор={components.operator_cost:.1f}")

  # Тест сравнения наборов маршрутов.
    print("\nСравнение разных наборов маршрутов:")
    print("-" * 60)

    route_sets = {
        'Оригинал': test_routes,
        'Только горизонтальные': [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        'Только вертикальные': [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14]]
    }

    comparison_results = compare_route_sets(G, route_sets, od_matrix)

    for name, components in comparison_results.items():
        print(f"{name}: стоимость={components.total_cost:.3f}, "
              f"связность={components.connectivity_ratio:.1%}")

    print("\nТестирование завершено успешно!")
