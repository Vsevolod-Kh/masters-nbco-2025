
  # Utils.py.
"""
Утилиты для Neural Bee Colony Optimization для транзитных сетей
Содержит базовые функции для работы с графами, геометрией и метриками качества
"""

import numpy as np
import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import nearest_points
from scipy.spatial import KDTree, distance_matrix
from scipy.spatial.distance import cdist
from typing import Dict, List, Tuple, Optional, Union, Set, Any
import pickle
import json
import logging
from config import config
import os
import shutil
from datetime import datetime

  # Настройка логирования.
logging.basicConfig(level=getattr(logging, config.system.log_level))
logger = logging.getLogger(__name__)

class GraphUtils:
    """Утилиты для работы с графами дорожной сети"""

    @staticmethod
    def create_graph_from_edges(edges_gdf: gpd.GeoDataFrame,
                                nodes_gdf: gpd.GeoDataFrame) -> nx.Graph:
        """
        Создать NetworkX граф из GeoDataFrame ребер и узлов

        Args:
            edges_gdf: GeoDataFrame с ребрами (должен содержать 'u', 'v')
            nodes_gdf: GeoDataFrame с узлами (должен содержать 'node_id')

        Returns:
            NetworkX граф
        """
        G = nx.Graph()

  # Добавляем узлы с атрибутами.
        for idx, node in nodes_gdf.iterrows():
            node_attrs = {
                'x': node.geometry.x,
                'y': node.geometry.y,
                'geometry': node.geometry
            }
  # Добавляем дополнительные атрибуты узла.
            for col in nodes_gdf.columns:
                if col not in ['geometry', 'node_id']:
                    node_attrs[col] = node[col]

            G.add_node(node['node_id'], **node_attrs)

  # Добавляем ребра с весами.
        for idx, edge in edges_gdf.iterrows():
            if edge['u'] in G.nodes and edge['v'] in G.nodes:
  # Вычисляем длину ребра если не указана.
                edge_length = edge.geometry.length if hasattr(edge.geometry, 'length') else 0
                travel_time = edge_length / config.data.vehicle_speed_ms  # Время в секундах.

                edge_attrs = {
                    'length': edge_length,
                    'travel_time': travel_time,
                    'geometry': edge.geometry
                }

  # Добавляем дополнительные атрибуты ребра.
                for col in edges_gdf.columns:
                    if col not in ['geometry', 'u', 'v']:
                        edge_attrs[col] = edge[col]

                G.add_edge(edge['u'], edge['v'], **edge_attrs)

        return G

    @staticmethod
    def get_largest_connected_component(G: nx.Graph) -> nx.Graph:
        """Получить крупнейший связный компонент графа"""
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest_cc = max(components, key=len)
            logger.warning(f"Граф несвязный. Используем крупнейший компонент: {len(largest_cc)}/{len(G)} узлов")
            return G.subgraph(largest_cc).copy()
        return G

    @staticmethod
    def compute_shortest_paths(G: nx.Graph,
                               weight: str = 'travel_time') -> Dict[int, Dict[int, float]]:
        """
        Вычислить кратчайшие пути между всеми парами узлов

        Args:
            G: NetworkX граф
            weight: атрибут ребра для весов

        Returns:
            Словарь расстояний {source: {target: distance}}
        """
        try:
            return dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
        except nx.NetworkXError as e:
            logger.error(f"Ошибка вычисления кратчайших путей: {e}")
            return {}

    @staticmethod
    def get_shortest_path_routes(G: nx.Graph,
                                 weight: str = 'travel_time') -> Dict[Tuple[int, int], List[int]]:
        """
        Получить кратчайшие маршруты между всеми парами узлов

        Returns:
            Словарь {(source, target): [path_nodes]}
        """
        try:
            path_dict = dict(nx.all_pairs_dijkstra_path(G, weight=weight))
            routes = {}
            for source in path_dict:
                for target in path_dict[source]:
                    routes[(source, target)] = path_dict[source][target]
            return routes
        except nx.NetworkXError as e:
            logger.error(f"Ошибка получения маршрутов: {e}")
            return {}

class GeometryUtils:
    """Утилиты для геометрических вычислений"""

    @staticmethod
    def euclidean_distance(p1: Tuple[float, float],
                           p2: Tuple[float, float]) -> float:
        """Евклидово расстояние между двумя точками"""
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def haversine_distance(lon1: float, lat1: float,
                           lon2: float, lat2: float) -> float:
        """
        Расстояние по формула Гаверсина (для WGS84 координат)

        Returns:
            Расстояние в метрах
        """
        R = 6371000  # Радиус Земли в метрах.

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    @staticmethod
    def points_within_radius(points: np.ndarray,
                             center: Tuple[float, float],
                             radius: float) -> np.ndarray:
        """
        Найти точки в радиусе от центра

        Args:
            points: массив координат точек (N x 2)
            center: координаты центра (x, y)
            radius: радиус в тех же единицах что и координаты

        Returns:
            Булев массив True для точек в радиусе
        """
        distances = np.sqrt(np.sum((points - np.array(center)) ** 2, axis=1))
        return distances <= radius

    @staticmethod
    def create_buffer_polygons(points_gdf: gpd.GeoDataFrame,
                               radius: float) -> gpd.GeoDataFrame:
        """Создать буферные полигоны вокруг точек"""
        buffered = points_gdf.copy()
        buffered['geometry'] = points_gdf.geometry.buffer(radius)
        return buffered

    @staticmethod
    def assign_points_to_nearest(source_points: gpd.GeoDataFrame,
                                 target_points: gpd.GeoDataFrame) -> pd.Series:
        """
        Назначить каждую исходную точку к ближайшей целевой точке

        Returns:
            Series с индексами ближайших целевых точек
        """
        if len(target_points) == 0:
            return pd.Series(index=source_points.index, dtype='int64')

  # Создаем массивы координат.
        source_coords = np.array([[p.x, p.y] for p in source_points.geometry])
        target_coords = np.array([[p.x, p.y] for p in target_points.geometry])

  # Используем KDTree для эффективного поиска.
        tree = KDTree(target_coords)
        distances, indices = tree.query(source_coords)

        return pd.Series(target_points.index[indices], index=source_points.index)

class TransitMetrics:
    """Метрики качества транзитных сетей"""

    @staticmethod
    def calculate_passenger_cost(G: nx.Graph,
                                 routes: List[List[int]],
                                 od_matrix: np.ndarray,
                                 transfer_penalty: float = 5.0) -> float:
        """
        Вычислить пассажирскую стоимость (среднее время поездки)

        Args:
            G: граф дорожной сети
            routes: список маршрутов [[node_ids]]
            od_matrix: матрица Origin-Destination спроса
            transfer_penalty: штраф за пересадку в минутах

        Returns:
            Средняя стоимость поездки для пассажиров
        """
        if not routes or len(routes) == 0:
            return float('inf')

  # Создаем граф транзитной сети.
        transit_graph = TransitMetrics._create_transit_graph(G, routes, transfer_penalty)

        total_weighted_time = 0.0
        total_demand = 0.0

        n_nodes = len(G.nodes())
        node_list = list(G.nodes())

  # Вычисляем время поездки для каждой пары OD.
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and od_matrix[i, j] > 0:
                    origin = node_list[i]
                    destination = node_list[j]

                    try:
  # Ищем кратчайший путь в транзитной сети.
                        travel_time = nx.shortest_path_length(
                            transit_graph, origin, destination, weight='travel_time'
                        )
                        total_weighted_time += od_matrix[i, j] * travel_time
                        total_demand += od_matrix[i, j]

                    except nx.NetworkXNoPath:
  # Если нет пути в транзитной сети, используем очень большой штраф.
                        total_weighted_time += od_matrix[i, j] * 1e6
                        total_demand += od_matrix[i, j]

        return total_weighted_time / total_demand if total_demand > 0 else float('inf')

    @staticmethod
    def calculate_operator_cost(G: nx.Graph, routes: List[List[int]]) -> float:
        """
        Вычислить операторскую стоимость (общее время работы маршрутов)

        Args:
            G: граф дорожной сети
            routes: список маршрутов [[node_ids]]

        Returns:
            Общее время работы всех маршрутов
        """
        total_time = 0.0

        for route in routes:
            if len(route) < 2:
                continue

            route_time = 0.0

  # Вычисляем время для каждого сегмента маршрута.
            for i in range(len(route) - 1):
                if G.has_edge(route[i], route[i + 1]):
                    edge_data = G[route[i]][route[i + 1]]
                    route_time += edge_data.get('travel_time', 0)
                else:
  # Если прямой связи нет, используем кратчайший путь.
                    try:
                        segment_time = nx.shortest_path_length(
                            G, route[i], route[i + 1], weight='travel_time'
                        )
                        route_time += segment_time
                    except nx.NetworkXNoPath:
                        route_time += 1e6  # Большой штраф.

  # Учитываем движение в обе стороны.
            total_time += 2 * route_time

        return total_time

    @staticmethod
    def calculate_constraint_violations(routes: List[List[int]],
                                        G: nx.Graph,
                                        min_length: int = 2,
                                        max_length: int = 15) -> float:
        """
        Вычислить штрафы за нарушение ограничений

        Returns:
            Штраф за нарушения (доля несвязных пар + нарушения длины)
        """
        if not routes:
            return 1.0  # Максимальный штраф если нет маршрутов.

        violations = 0.0

  # 1. Проверяем длины маршрутов.
        length_violations = 0
        for route in routes:
            if len(route) < min_length:
                length_violations += min_length - len(route)
            elif len(route) > max_length:
                length_violations += len(route) - max_length

        violations += length_violations / len(routes)

  # 2. Проверяем связность сети.
        transit_graph = TransitMetrics._create_transit_graph(G, routes)

        if len(transit_graph.nodes()) == 0:
            violations += 1.0
        else:
  # Доля несвязных пар узлов.
            total_pairs = len(G.nodes()) * (len(G.nodes()) - 1)
            connected_pairs = 0

            for source in G.nodes():
                if source in transit_graph:
                    reachable = nx.single_source_shortest_path_length(transit_graph, source)
                    connected_pairs += len(reachable) - 1  # -1 чтобы не считать сам узел.

            connectivity_ratio = connected_pairs / total_pairs if total_pairs > 0 else 0
            violations += (1.0 - connectivity_ratio)

        return violations

    @staticmethod
    def _create_transit_graph(G: nx.Graph,
                              routes: List[List[int]],
                              transfer_penalty: float = 0.0) -> nx.Graph:
        """
        Создать граф транзитной сети из маршрутов

        Args:
            G: исходный граф дорожной сети
            routes: список маршрутов
            transfer_penalty: штраф за пересадку

        Returns:
            Граф транзитной сети с возможностями пересадок
        """
        transit_graph = nx.Graph()

  # Добавляем узлы из исходного графа.
        for node in G.nodes():
            transit_graph.add_node(node, **G.nodes[node])

  # Добавляем ребра вдоль маршрутов.
        for route in routes:
            if len(route) < 2:
                continue

            for i in range(len(route) - 1):
                curr_node = route[i]
                next_node = route[i + 1]

                if G.has_edge(curr_node, next_node):
                    edge_data = G[curr_node][next_node].copy()

  # Если ребро уже есть в транзитной сети, оставляем минимальное время.
                    if transit_graph.has_edge(curr_node, next_node):
                        existing_time = transit_graph[curr_node][next_node]['travel_time']
                        new_time = edge_data.get('travel_time', float('inf'))
                        if new_time < existing_time:
                            transit_graph[curr_node][next_node].update(edge_data)
                    else:
                        transit_graph.add_edge(curr_node, next_node, **edge_data)

  # Добавляем возможности пересадок (если требуется штраф).
        if transfer_penalty > 0:
  # Находим узлы где пересекаются маршруты.
            route_nodes = {}  # Node_id -> список маршрутов.
            for route_idx, route in enumerate(routes):
                for node in route:
                    if node not in route_nodes:
                        route_nodes[node] = []
                    route_nodes[node].append(route_idx)

  # В узлах пересечения добавляем штраф за пересадку.
            for node, route_list in route_nodes.items():
                if len(route_list) > 1:  # Узел принадлежит нескольким маршрутам.
  # Добавляем ребра с штрафом между всеми соседними узлами в разных маршрутах.
                    neighbors_by_route = {}
                    for route_idx in route_list:
                        route = routes[route_idx]
                        node_pos = route.index(node)
                        neighbors = []
                        if node_pos > 0:
                            neighbors.append(route[node_pos - 1])
                        if node_pos < len(route) - 1:
                            neighbors.append(route[node_pos + 1])
                        neighbors_by_route[route_idx] = neighbors

  # Добавляем "пересадочные" ребра между разными маршрутами.
                    for r1 in route_list:
                        for r2 in route_list:
                            if r1 != r2:
                                for n1 in neighbors_by_route[r1]:
                                    for n2 in neighbors_by_route[r2]:
                                        if not transit_graph.has_edge(n1, n2):
  # Время = базовое время + штраф за пересадку.
                                            base_time = 0
                                            if G.has_edge(n1, n2):
                                                base_time = G[n1][n2].get('travel_time', 0)

                                            transfer_time = base_time + transfer_penalty * 60  # Штраф в секундах.
                                            transit_graph.add_edge(n1, n2,
                                                                   travel_time=transfer_time,
                                                                   is_transfer=True)

        return transit_graph

class DataUtils:
    """Утилиты для обработки данных"""

    @staticmethod
    def normalize_features(data: np.ndarray,
                           mean: Optional[np.ndarray] = None,
                           std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Нормализация признаков (z-score)

        Args:
            data: исходные данные
            mean: среднее (если None, вычисляется из data)
            std: стандартное отклонение (если None, вычисляется из data)

        Returns:
            (normalized_data, mean, std)
        """
        if mean is None:
            mean = np.mean(data, axis=0)
        if std is None:
            std = np.std(data, axis=0)
            std[std == 0] = 1.0  # Избегаем деления на ноль.

        normalized = (data - mean) / std
        return normalized, mean, std

    @staticmethod
    def create_od_matrix(nodes_gdf: gpd.GeoDataFrame,
                         demand_range: Tuple[int, int] = (60, 800)) -> np.ndarray:
        """
        Создать синтетическую OD матрицу

        Args:
            nodes_gdf: GeoDataFrame с узлами
            demand_range: диапазон значений спроса

        Returns:
            Матрица Origin-Destination спроса
        """
        n_nodes = len(nodes_gdf)
        od_matrix = np.zeros((n_nodes, n_nodes))

  # Генерируем случайный спрос для недиагональных элементов.
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    od_matrix[i, j] = np.random.randint(demand_range[0], demand_range[1] + 1)

        return od_matrix

    @staticmethod
    def augment_city_data(nodes_coords: np.ndarray,
                          od_matrix: np.ndarray,
                          scale_range: Tuple[float, float] = (0.4, 1.6),
                          rotation_range: Tuple[float, float] = (0, 360),
                          demand_scale_range: Tuple[float, float] = (0.8, 1.2)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Аугментация данных города для обучения [[11]]

        Args:
            nodes_coords: координаты узлов (N x 2)
            od_matrix: матрица спроса (N x N)
            scale_range: диапазон масштабирования координат
            rotation_range: диапазон поворота в градусах
            demand_scale_range: диапазон масштабирования спроса

        Returns:
            (augmented_coords, augmented_od_matrix)
        """
        coords = nodes_coords.copy()
        od_aug = od_matrix.copy()

  # Масштабирование координат.
        scale_factor = np.random.uniform(scale_range[0], scale_range[1])
        coords *= scale_factor

  # Поворот координат.
        angle = np.random.uniform(rotation_range[0], rotation_range[1])
        angle_rad = np.radians(angle)

  # Центрируем координаты относительно центроида.
        centroid = np.mean(coords, axis=0)
        coords_centered = coords - centroid

  # Матрица поворота.
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])

  # Применяем поворот.
        coords_rotated = coords_centered @ rotation_matrix.T
        coords = coords_rotated + centroid

  # Масштабирование спроса.
        demand_scale = np.random.uniform(demand_scale_range[0], demand_scale_range[1])
        od_aug = (od_aug * demand_scale).astype(int)

        return coords, od_aug

    @staticmethod
    def save_data(data: any, filepath: str) -> None:
        """Сохранить данные в файл"""
        if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        elif filepath.endswith('.json'):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {filepath}")

    @staticmethod
    def load_data(filepath: str) -> any:
        """Загрузить данные из файла"""
        if filepath.endswith('.pkl') or filepath.endswith('.pickle'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filepath.endswith('.json'):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Неподдерживаемый формат файла: {filepath}")

class RouteUtils:
    """Утилиты для работы с маршрутами"""

    @staticmethod
    def validate_route(route: List[int],
                       G: nx.Graph,
                       min_length: int = 2,
                       max_length: int = 15) -> Tuple[bool, str]:
        """
        Проверить валидность маршрута

        Returns:
            (is_valid, error_message)
        """
        if len(route) < min_length:
            return False, f"Маршрут слишком короткий: {len(route)} < {min_length}"

        if len(route) > max_length:
            return False, f"Маршрут слишком длинный: {len(route)} > {max_length}"

  # Проверяем на циклы.
        if len(set(route)) != len(route):
            return False, "Маршрут содержит циклы"

  # Проверяем, что все узлы существуют в графе.
        for node in route:
            if node not in G.nodes():
                return False, f"Узел {node} не существует в графе"

  # Проверяем связность маршрута.
        for i in range(len(route) - 1):
            if not nx.has_path(G, route[i], route[i + 1]):
                return False, f"Нет пути между узлами {route[i]} и {route[i + 1]}"

        return True, "OK"

    @staticmethod
    def routes_to_node_coverage(routes: List[List[int]]) -> Set[int]:
        """Получить множество всех узлов, покрытых маршрутами"""
        covered_nodes = set()
        for route in routes:
            covered_nodes.update(route)
        return covered_nodes

    @staticmethod
    def _haversine(lon1, lat1, lon2, lat2):
        """Быстрая Haversine-дистанция (метры)"""
        R = 6_371_000  # М.
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = phi2 - phi1
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
        return 2 * R * np.arcsin(np.sqrt(a))

    @staticmethod
    def select_border_od_pairs(G: nx.Graph,
                               max_pairs: int,
                               distance_threshold: float = 1000.0
                               ) -> Optional[List[Tuple[int, int]]]:
        """
        Вернёт до `max_pairs` (entry, exit) из border_stops_direction_1.csv.

        • Поддерживает реальные файлы с колонками:
            - stop_id  | node_id
            - coordinates ("lat,lon") **ИЛИ** lat/lon **ИЛИ** x/y
        • Отбрасывает дубликаты и пары ближе `distance_threshold` м.
        • Если CSV не найден или невалиден – вернётся None → BCO упадёт
          на старую случайную логику (как при обучении на synthetic).
        """
        csv_path = config.files.border_stops_csv
        if not os.path.exists(csv_path):
            logger.warning("CSV с border-stops не найден – использую рандомные пары")
            return None

        df = pd.read_csv(csv_path)

  # --- node_id / stop_id -------------------------------------------------.
        if 'node_id' not in df.columns and 'stop_id' in df.columns:
            df = df.rename(columns={'stop_id': 'node_id'})
        if 'node_id' not in df.columns:
            raise ValueError("CSV: не хватает 'node_id' (или 'stop_id')")

  # --- координаты --------------------------------------------------------.
        if {'x', 'y'}.issubset(df.columns):
            df['lon'] = df['x']
            df['lat'] = df['y']
        elif 'coordinates' in df.columns:
            latlon = df['coordinates'].str.split(',', expand=True)
            if latlon.shape[1] != 2:
                raise ValueError("CSV: колонка 'coordinates' должна быть вида 'lat,lon'")
            df['lat'] = latlon[0].astype(float)
            df['lon'] = latlon[1].astype(float)
        elif {'lat', 'lon'}.issubset(df.columns):
            df['lat'] = df['lat'].astype(float)
            df['lon'] = df['lon'].astype(float)
        else:
            raise ValueError("CSV: нужны координаты (x/y, lat/lon или coordinates)")

  # --- фильтрация --------------------------------------------------------.
        df = df[['node_id', 'type', 'lat', 'lon']].drop_duplicates()
        entries = df[df.type == 'entry'].reset_index(drop=True)
        exits = df[df.type == 'exit'].reset_index(drop=True)

        if entries.empty or exits.empty:
            raise ValueError("CSV: нет достаточного числа entry/exit")

        exit_tree  = KDTree(exits[['lat', 'lon']].values)
        used_e_idx, used_x_idx, pairs = set(), set(), []
        rng = np.random.default_rng(42)

  # Случайный порядок для разнообразия.
        entry_order = rng.permutation(entries.index)

        for e_idx in entry_order:
            if len(pairs) >= max_pairs:
                break
            if e_idx in used_e_idx:
                continue

            e_row = entries.loc[e_idx]
  # Ближайшие выходы в порядке возрастания евклидовой (градусной) дистанции.
            dists, idxs = exit_tree.query([e_row[['lat', 'lon']].values],
                                           k=len(exits))
            for x_idx in idxs[0]:
                if x_idx in used_x_idx:
                    continue
                x_row = exits.loc[x_idx]

                dist_m = RouteUtils._haversine(e_row.lon, e_row.lat,
                                               x_row.lon, x_row.lat)
                if dist_m < distance_threshold:
                    continue
  # Узлы должны существовать в графе.
                if e_row.node_id not in G.nodes or x_row.node_id not in G.nodes:
                    continue

                pairs.append((int(e_row.node_id), int(x_row.node_id)))
                used_e_idx.add(e_idx)
                used_x_idx.add(x_idx)
                break  # Переходим к следующему entry.

        return pairs if pairs else None

    @staticmethod
    def calculate_route_overlap(route1: List[int], route2: List[int]) -> float:
        """
        Вычислить долю пересечения между двумя маршрутами

        Returns:
            Доля общих узлов (0.0 - 1.0)
        """
        set1 = set(route1)
        set2 = set(route2)

        if len(set1) == 0 and len(set2) == 0:
            return 1.0

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    @staticmethod
    def two_opt(route: List[int], G: nx.Graph) -> List[int]:
        """
        Простая версия 2-opt: пытаемся уменьшить суммарное travel_time,
        разворачивая ребро (i, i+1) ↔ (k, k+1).
        Кончаем, когда улучшений нет.
        """

        if len(route) < 4:
            return route

        def time(u, v):
            try:
                return nx.shortest_path_length(G, u, v, weight='travel_time')
            except nx.NetworkXNoPath:
                return float('inf')
        improved = True
        while improved:
            improved = False
            for i in range(1, len(route) - 2):
                for k in range(i + 1, len(route) - 1):
                    a, b = route[i - 1], route[i]
                    c, d = route[k], route[k + 1]

                    if (time(a, b) + time(c, d)) > (time(a, c) + time(b, d)):
                        route[i:k + 1] = reversed(route[i:k + 1])
                        improved = True
                        break
            if improved:
                break
        return route

    @staticmethod
    def smooth_routes(routes: List[List[int]], G: nx.Graph) -> List[List[int]]:
        """Применяем 2-opt к каждому маршруту"""
        return [RouteUtils.two_opt(r, G) for r in routes]

    @staticmethod
    def save_routes(routes: List[List[int]], path: str) -> None:
        """Сохраняет list[list[int]] в pickle."""
        with open(path, "wb") as f:
            pickle.dump(routes, f)

    @staticmethod
    def load_routes(path: str) -> List[List[int]]:
        with open(path, "rb") as f:
            return pickle.load(f)

def save_results(results: Dict[str, Any],
                 filename: str = None,
                 results_dir: str = None,
                 backup: bool = True) -> str:
    """
    Сохранить результаты в JSON файл с улучшенной обработкой ошибок

    Args:
        results: словарь с результатами
        filename: имя файла (если None, генерируется автоматически)
        results_dir: директория для сохранения
        backup: создавать ли резервную копию если файл существует

    Returns:
        Путь к сохраненному файлу
    """
    if results_dir is None:
        results_dir = config.files.results_dir

  # Создаем директорию если не существует.
    os.makedirs(results_dir, exist_ok=True)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.json"

  # Убеждаемся что файл имеет расширение .json.
    if not filename.endswith('.json'):
        filename += '.json'

    filepath = os.path.join(results_dir, filename)

    try:
  # Создаем резервную копию если файл существует.
        if backup and os.path.exists(filepath):
            backup_path = filepath.replace('.json', '_backup.json')
            shutil.copy2(filepath, backup_path)
            logger.info(f"Создана резервная копия: {backup_path}")

  # Проверяем что results можно сериализовать.
        try:
            json.dumps(results, default=str)
        except (TypeError, ValueError) as e:
            logger.error(f"Данные не могут быть сериализованы в JSON: {e}")
            return ""

  # Сохраняем с временным файлом для атомарности.
        temp_filepath = filepath + '.tmp'

        with open(temp_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

  # Атомарно перемещаем временный файл.
        shutil.move(temp_filepath, filepath)

  # Проверяем размер файла.
        file_size = os.path.getsize(filepath)
        logger.info(f" Результаты сохранены в {filepath} (размер: {file_size} байт)")

        return filepath

    except PermissionError as e:
        logger.error(f"Нет прав для записи в {filepath}: {e}")
        return ""

    except OSError as e:
        logger.error(f"Ошибка файловой системы при сохранении {filepath}: {e}")
        return ""

    except Exception as e:
        logger.error(f"Неожиданная ошибка сохранения результатов: {e}")

  # Пытаемся удалить временный файл если он остался.
        temp_filepath = filepath + '.tmp'
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass

        return ""

def check_system_requirements() -> Dict[str, Any]:
    """
    Проверить системные требования для Neural BCO

    Returns:
        Словарь с результатами проверки
    """
    import sys
    import platform
    import psutil

    requirements = {
        'system_info': {
            'platform': platform.platform(),
            'python_version': sys.version,
            'architecture': platform.architecture()[0]
        },
        'hardware': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': round(psutil.virtual_memory().total / (1024 ** 3), 1),
            'available_memory_gb': round(psutil.virtual_memory().available / (1024 ** 3), 1)
        },
        'dependencies': {},
        'recommendations': [],
        'status': 'ok'
    }

  # Проверяем Python версию.
    if sys.version_info < (3, 8):
        requirements['recommendations'].append("Рекомендуется Python 3.8+")
        requirements['status'] = 'warning'

  # Проверяем память.
    if requirements['hardware']['memory_gb'] < 4:
        requirements['recommendations'].append("Рекомендуется минимум 4GB RAM")
        requirements['status'] = 'warning'

  # Проверяем зависимости.
    required_packages = ['torch', 'networkx', 'geopandas', 'numpy', 'pandas']

    for package in required_packages:
        try:
            __import__(package)
            requirements['dependencies'][package] = 'installed'
        except ImportError:
            requirements['dependencies'][package] = 'missing'
            requirements['recommendations'].append(f"Установите {package}")
            requirements['status'] = 'error'

  # Проверяем PyTorch и устройство.
    try:
        import torch
        requirements['dependencies']['torch_version'] = torch.__version__

        if torch.cuda.is_available():
            requirements['hardware']['cuda_available'] = True
            requirements['hardware']['cuda_device'] = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            requirements['hardware']['mps_available'] = True
        else:
            requirements['hardware']['gpu_acceleration'] = False
            requirements['recommendations'].append("GPU ускорение недоступно, будет использоваться CPU")

    except ImportError:
        pass

    return requirements

def create_data_summary(opt_stops: gpd.GeoDataFrame) -> str:
    """
    Создать текстовую сводку по данным остановок

    Args:
        opt_stops: GeoDataFrame с остановками

    Returns:
        Текстовая сводка
    """
    if len(opt_stops) == 0:
        return "Данные остановок отсутствуют"

    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("СВОДКА ПО ДАННЫМ ОСТАНОВОК")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Общее количество остановок: {len(opt_stops)}")

  # Информация о геометрии.
    if 'geometry' in opt_stops.columns:
        bounds = opt_stops.total_bounds
        width_km = (bounds[2] - bounds[0]) / 1000
        height_km = (bounds[3] - bounds[1]) / 1000
        area_km2 = width_km * height_km

        summary_lines.append(f"Область покрытия: {width_km:.1f} x {height_km:.1f} км")
        summary_lines.append(f"Площадь: {area_km2:.1f} км²")
        summary_lines.append(f"Плотность: {len(opt_stops) / area_km2:.1f} остановок/км²")

  # Информация о типах остановок.
    if 'type' in opt_stops.columns:
        type_counts = opt_stops['type'].value_counts()
        summary_lines.append("\nТипы остановок:")
        for stop_type, count in type_counts.items():
            percentage = count / len(opt_stops) * 100
            summary_lines.append(f"  {stop_type}: {count} ({percentage:.1f}%)")

  # Информация о дополнительных атрибутах.
    optional_attrs = ['population', 'jobs', 'importance']
    for attr in optional_attrs:
        if attr in opt_stops.columns:
            values = opt_stops[attr]
            summary_lines.append(f"\n{attr.capitalize()}:")
            summary_lines.append(f"  Среднее: {values.mean():.1f}")
            summary_lines.append(f"  Диапазон: {values.min():.1f} - {values.max():.1f}")

  # CRS информация.
    if opt_stops.crs:
        summary_lines.append(f"\nСистема координат: {opt_stops.crs}")

    summary_lines.append("=" * 60)

    return "\n".join(summary_lines)

def load_opt_fin2_data() -> Tuple[gpd.GeoDataFrame, Dict[str, Any]]:
    """
    Загрузить данные остановок из opt_fin2.py с улучшенной обработкой ошибок

    Returns:
        Tuple[GeoDataFrame, metadata]: остановки и метаданные
    """
  # Используем путь из конфигурации.
    opt_stops_file = config.files.opt_stops_file

  # Проверяем существование файла.
    if not os.path.exists(opt_stops_file):
        logger.error(f"Файл {opt_stops_file} не найден в текущей директории")
        logger.info("Убедитесь, что файл opt_stops.pkl находится в корневой папке проекта")
        logger.info("Или запустите opt_fin2.py для создания файла")
        return gpd.GeoDataFrame(), {'error': 'file_not_found', 'file': opt_stops_file}

  # Проверяем размер файла.
    file_size = os.path.getsize(opt_stops_file)
    if file_size == 0:
        logger.error(f"Файл {opt_stops_file} пуст")
        return gpd.GeoDataFrame(), {'error': 'empty_file', 'file': opt_stops_file}

    try:
        logger.info(f"Загрузка данных из {opt_stops_file} (размер: {file_size} байт)")

        with open(opt_stops_file, 'rb') as f:
            opt_stops = pickle.load(f)

  # Валидация загруженных данных.
        if not isinstance(opt_stops, gpd.GeoDataFrame):
            logger.error(f"Неожиданный тип данных: {type(opt_stops)}, ожидался GeoDataFrame")
            return gpd.GeoDataFrame(), {'error': 'invalid_data_type', 'type': str(type(opt_stops))}

        if len(opt_stops) == 0:
            logger.warning("Загруженный GeoDataFrame пуст")
            return opt_stops, {'error': 'empty_dataframe', 'num_stops': 0}

  # Проверяем обязательные колонки.
        required_columns = ['geometry']
        missing_columns = [col for col in required_columns if col not in opt_stops.columns]
        if missing_columns:
            logger.error(f"Отсутствуют обязательные колонки: {missing_columns}")
            return gpd.GeoDataFrame(), {'error': 'missing_columns', 'missing': missing_columns}

  # Проверяем геометрию.
        if opt_stops.geometry.isnull().any():
            null_count = opt_stops.geometry.isnull().sum()
            logger.warning(f"Найдено {null_count} остановок с пустой геометрией")
            opt_stops = opt_stops.dropna(subset=['geometry'])
            logger.info(f"Удалено {null_count} остановок с пустой геометрией")

  # Проверяем CRS.
        if opt_stops.crs is None:
            logger.warning("CRS не установлена, устанавливаем EPSG:3857")
            opt_stops = opt_stops.set_crs('EPSG:3857')

  # Добавляем node_id если отсутствует.
        if 'node_id' not in opt_stops.columns:
            opt_stops['node_id'] = range(len(opt_stops))
            logger.info("Добавлена колонка node_id")

  # Создаем детальные метаданные.
        metadata = {
            'source': 'opt_fin2.py',
            'file_path': opt_stops_file,
            'file_size_bytes': file_size,
            'num_stops': len(opt_stops),
            'columns': list(opt_stops.columns),
            'crs': str(opt_stops.crs),
            'bounds': opt_stops.total_bounds.tolist() if len(opt_stops) > 0 else None,
            'loaded_at': datetime.now().isoformat(),
            'data_types': {col: str(dtype) for col, dtype in opt_stops.dtypes.items()},
            'has_type_column': 'type' in opt_stops.columns,
            'has_population_column': 'population' in opt_stops.columns,
            'geometry_types': opt_stops.geometry.geom_type.value_counts().to_dict()
        }

  # Дополнительная статистика если есть колонка type.
        if 'type' in opt_stops.columns:
            metadata['stop_types'] = opt_stops['type'].value_counts().to_dict()

        logger.info(f" Успешно загружено {len(opt_stops)} остановок")
        logger.info(f"  Колонки: {list(opt_stops.columns)}")
        logger.info(f"  CRS: {opt_stops.crs}")

        return opt_stops, metadata

    except pickle.UnpicklingError as e:
        logger.error(f"Ошибка десериализации pickle файла {opt_stops_file}: {e}")
        logger.info("Возможно файл поврежден или создан несовместимой версией Python")
        return gpd.GeoDataFrame(), {'error': 'pickle_error', 'details': str(e)}

    except MemoryError as e:
        logger.error(f"Недостаточно памяти для загрузки {opt_stops_file}: {e}")
        logger.info(f"Размер файла: {file_size / 1024 / 1024:.1f} MB")
        return gpd.GeoDataFrame(), {'error': 'memory_error', 'file_size_mb': file_size / 1024 / 1024}

    except PermissionError as e:
        logger.error(f"Нет прав доступа к файлу {opt_stops_file}: {e}")
        return gpd.GeoDataFrame(), {'error': 'permission_error', 'details': str(e)}

    except Exception as e:
        logger.error(f"Неожиданная ошибка при загрузке {opt_stops_file}: {e}")
        logger.error(f"Тип ошибки: {type(e).__name__}")
        return gpd.GeoDataFrame(), {'error': 'unexpected_error', 'type': type(e).__name__, 'details': str(e)}

def validate_opt_fin2_data(opt_stops: gpd.GeoDataFrame) -> Dict[str, Any]:
    """
    Валидировать данные остановок из opt_fin2.py

    Args:
        opt_stops: GeoDataFrame с остановками

    Returns:
        Словарь с результатами валидации
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }

    try:
  # Проверка базовой структуры.
        if len(opt_stops) == 0:
            validation_results['errors'].append("GeoDataFrame пуст")
            validation_results['is_valid'] = False
            return validation_results

  # Проверка геометрии.
        if 'geometry' not in opt_stops.columns:
            validation_results['errors'].append("Отсутствует колонка 'geometry'")
            validation_results['is_valid'] = False
        else:
  # Проверяем типы геометрии.
            geom_types = opt_stops.geometry.geom_type.value_counts()
            validation_results['statistics']['geometry_types'] = geom_types.to_dict()

            if 'Point' not in geom_types:
                validation_results['warnings'].append("Не найдены Point геометрии")

  # Проверяем пустые геометрии.
            null_geom_count = opt_stops.geometry.isnull().sum()
            if null_geom_count > 0:
                validation_results['warnings'].append(f"Найдено {null_geom_count} пустых геометрий")

  # Проверка CRS.
        if opt_stops.crs is None:
            validation_results['warnings'].append("CRS не установлена")
        else:
            validation_results['statistics']['crs'] = str(opt_stops.crs)

  # Проверка node_id.
        if 'node_id' not in opt_stops.columns:
            validation_results['warnings'].append("Отсутствует колонка 'node_id'")
        else:
  # Проверяем уникальность node_id.
            duplicate_ids = opt_stops['node_id'].duplicated().sum()
            if duplicate_ids > 0:
                validation_results['warnings'].append(f"Найдено {duplicate_ids} дублированных node_id")

  # Проверка типов остановок.
        if 'type' in opt_stops.columns:
            type_counts = opt_stops['type'].value_counts()
            validation_results['statistics']['stop_types'] = type_counts.to_dict()

  # Проверяем ожидаемые типы.
            expected_types = ['key', 'connection', 'ordinary']
            found_types = set(opt_stops['type'].unique())
            unexpected_types = found_types - set(expected_types)

            if unexpected_types:
                validation_results['warnings'].append(f"Неожиданные типы остановок: {unexpected_types}")
        else:
            validation_results['warnings'].append("Отсутствует колонка 'type'")

  # Проверка пространственного распределения.
        if len(opt_stops) > 1 and 'geometry' in opt_stops.columns:
            bounds = opt_stops.total_bounds
            width = bounds[2] - bounds[0]
            height = bounds[3] - bounds[1]

            validation_results['statistics']['spatial_bounds'] = {
                'width_m': width,
                'height_m': height,
                'area_km2': (width * height) / 1000000
            }

  # Проверяем разумность размеров.
            if width < 1000 or height < 1000:
                validation_results['warnings'].append("Очень маленькая область покрытия")
            elif width > 100000 or height > 100000:
                validation_results['warnings'].append("Очень большая область покрытия")

  # Проверка плотности остановок.
        if len(opt_stops) > 0:
            validation_results['statistics']['num_stops'] = len(opt_stops)

            if len(opt_stops) < 10:
                validation_results['warnings'].append("Очень мало остановок для оптимизации")
            elif len(opt_stops) > 1000:
                validation_results['warnings'].append("Очень много остановок, может быть медленно")

  # Проверка дополнительных колонок.
        optional_columns = ['population', 'jobs', 'importance', 'demand']
        for col in optional_columns:
            if col in opt_stops.columns:
                validation_results['statistics'][f'has_{col}'] = True

  # Проверяем на отрицательные значения.
                if opt_stops[col].dtype in ['int64', 'float64']:
                    negative_count = (opt_stops[col] < 0).sum()
                    if negative_count > 0:
                        validation_results['warnings'].append(
                            f"Найдено {negative_count} отрицательных значений в {col}")
            else:
                validation_results['statistics'][f'has_{col}'] = False

  # Общая оценка качества данных.
        error_count = len(validation_results['errors'])
        warning_count = len(validation_results['warnings'])

        if error_count > 0:
            validation_results['is_valid'] = False
            validation_results['quality_score'] = 0.0
        elif warning_count == 0:
            validation_results['quality_score'] = 1.0
        else:
            validation_results['quality_score'] = max(0.1, 1.0 - warning_count * 0.1)

        logger.info(f"Валидация данных завершена: {error_count} ошибок, {warning_count} предупреждений")
        logger.info(f"Оценка качества данных: {validation_results['quality_score']:.1f}")

    except Exception as e:
        validation_results['errors'].append(f"Ошибка валидации: {e}")
        validation_results['is_valid'] = False
        logger.error(f"Ошибка при валидации данных: {e}")

    return validation_results

def repair_opt_fin2_data(opt_stops: gpd.GeoDataFrame) -> Tuple[gpd.GeoDataFrame, List[str]]:
    """
    Попытаться исправить распространенные проблемы в данных opt_fin2

    Args:
        opt_stops: GeoDataFrame с остановками

    Returns:
        Tuple[исправленный GeoDataFrame, список выполненных исправлений]
    """
    repairs = []

    try:
  # Создаем копию для безопасности.
        fixed_stops = opt_stops.copy()

  # Исправление 1: Удаляем пустые геометрии.
        if fixed_stops.geometry.isnull().any():
            null_count = fixed_stops.geometry.isnull().sum()
            fixed_stops = fixed_stops.dropna(subset=['geometry'])
            repairs.append(f"Удалено {null_count} остановок с пустой геометрией")

  # Исправление 2: Устанавливаем CRS если отсутствует.
        if fixed_stops.crs is None:
            fixed_stops = fixed_stops.set_crs('EPSG:3857')
            repairs.append("Установлена CRS: EPSG:3857")

  # Исправление 3: Добавляем node_id если отсутствует.
        if 'node_id' not in fixed_stops.columns:
            fixed_stops['node_id'] = range(len(fixed_stops))
            repairs.append("Добавлена колонка node_id")

  # Исправление 4: Исправляем дублированные node_id.
        if 'node_id' in fixed_stops.columns:
            duplicate_mask = fixed_stops['node_id'].duplicated()
            if duplicate_mask.any():
                duplicate_count = duplicate_mask.sum()
                fixed_stops.loc[duplicate_mask, 'node_id'] = range(
                    fixed_stops['node_id'].max() + 1,
                    fixed_stops['node_id'].max() + 1 + duplicate_count
                )
                repairs.append(f"Исправлено {duplicate_count} дублированных node_id")

  # Исправление 5: Добавляем тип остановок если отсутствует.
        if 'type' not in fixed_stops.columns:
  # Простая эвристика для определения типов.
            if len(fixed_stops) > 0:
  # Ключевые остановки - случайные 20%.
                num_key = max(1, len(fixed_stops) // 5)
  # Соединительные остановки - случайные 15%.
                num_connection = max(1, len(fixed_stops) // 7)

                fixed_stops['type'] = 'ordinary'

  # Случайно выбираем ключевые остановки.
                key_indices = np.random.choice(len(fixed_stops), num_key, replace=False)
                fixed_stops.iloc[key_indices, fixed_stops.columns.get_loc('type')] = 'key'

  # Случайно выбираем соединительные из оставшихся.
                remaining_indices = [i for i in range(len(fixed_stops)) if i not in key_indices]
                connection_indices = np.random.choice(remaining_indices,
                                                      min(num_connection, len(remaining_indices)),
                                                      replace=False)
                fixed_stops.iloc[connection_indices, fixed_stops.columns.get_loc('type')] = 'connection'

                repairs.append(
                    f"Добавлена колонка type: {num_key} key, {num_connection} connection, {len(fixed_stops) - num_key - num_connection} ordinary")

  # Исправление 6: Исправляем отрицательные значения в числовых колонках.
        numeric_columns = ['population', 'jobs', 'importance', 'demand']
        for col in numeric_columns:
            if col in fixed_stops.columns and fixed_stops[col].dtype in ['int64', 'float64']:
                negative_mask = fixed_stops[col] < 0
                if negative_mask.any():
                    negative_count = negative_mask.sum()
                    fixed_stops.loc[negative_mask, col] = 0
                    repairs.append(f"Исправлено {negative_count} отрицательных значений в {col}")

  # Исправление 7: Добавляем базовые атрибуты если отсутствуют.
        if 'importance' not in fixed_stops.columns:
  # Важность зависит от типа остановки.
            importance_map = {'key': 1.0, 'connection': 0.7, 'ordinary': 0.3}
            if 'type' in fixed_stops.columns:
                fixed_stops['importance'] = fixed_stops['type'].map(importance_map).fillna(0.3)
            else:
                fixed_stops['importance'] = 0.5
            repairs.append("Добавлена колонка importance")

  # Исправление 8: Проверяем и исправляем геометрию.
        if len(fixed_stops) > 0:
  # Проверяем валидность геометрии.
            invalid_geom = ~fixed_stops.geometry.is_valid
            if invalid_geom.any():
                invalid_count = invalid_geom.sum()
  # Пытаемся исправить через buffer(0).
                fixed_stops.loc[invalid_geom, 'geometry'] = fixed_stops.loc[invalid_geom, 'geometry'].buffer(0)
                repairs.append(f"Исправлено {invalid_count} невалидных геометрий")

  # Исправление 9: Убираем дубликаты по координатам.
        if len(fixed_stops) > 1:
  # Создаем строки координат для сравнения.
            coords_str = fixed_stops.geometry.apply(lambda geom: f"{geom.x:.6f},{geom.y:.6f}")
            duplicate_coords = coords_str.duplicated()

            if duplicate_coords.any():
                duplicate_count = duplicate_coords.sum()
                fixed_stops = fixed_stops[~duplicate_coords]
  # Обновляем node_id после удаления дубликатов.
                fixed_stops['node_id'] = range(len(fixed_stops))
                repairs.append(f"Удалено {duplicate_count} дубликатов по координатам")

        logger.info(f"Выполнено {len(repairs)} исправлений данных")
        for repair in repairs:
            logger.info(f"  - {repair}")

        return fixed_stops, repairs

    except Exception as e:
        logger.error(f"Ошибка при исправлении данных: {e}")
        return opt_stops, [f"Ошибка исправления: {e}"]

def create_test_opt_stops(num_stops: int = 50, area_size: float = 10000) -> gpd.GeoDataFrame:
    """
    Создать тестовые данные остановок для демонстрации

    Args:
        num_stops: количество остановок
        area_size: размер области в метрах

    Returns:
        GeoDataFrame с тестовыми остановками
    """
    logger.info(f"Создание {num_stops} тестовых остановок в области {area_size}x{area_size} м")

    try:
  # Генерируем случайные координаты.
        np.random.seed(42)  # Для воспроизводимости.

        x_coords = np.random.uniform(0, area_size, num_stops)
        y_coords = np.random.uniform(0, area_size, num_stops)

  # Создаем геометрию.
        geometry = [Point(x, y) for x, y in zip(x_coords, y_coords)]

  # Определяем типы остановок.
        num_key = max(1, num_stops // 5)  # 20% ключевых.
        num_connection = max(1, num_stops // 7)  # ~15% соединительных.

        stop_types = ['ordinary'] * num_stops

  # Случайно назначаем ключевые остановки.
        key_indices = np.random.choice(num_stops, num_key, replace=False)
        for idx in key_indices:
            stop_types[idx] = 'key'

  # Случайно назначаем соединительные из оставшихся.
        remaining_indices = [i for i in range(num_stops) if i not in key_indices]
        connection_indices = np.random.choice(remaining_indices,
                                              min(num_connection, len(remaining_indices)),
                                              replace=False)
        for idx in connection_indices:
            stop_types[idx] = 'connection'

  # Создаем дополнительные атрибуты.
        importance_map = {'key': 1.0, 'connection': 0.7, 'ordinary': 0.3}
        importance = [importance_map[t] + np.random.normal(0, 0.1) for t in stop_types]
        importance = np.clip(importance, 0.1, 1.0)

        population = np.random.randint(100, 2000, num_stops)
        jobs = np.random.randint(50, 1500, num_stops)

  # Создаем GeoDataFrame.
        test_stops = gpd.GeoDataFrame({
            'node_id': range(num_stops),
            'type': stop_types,
            'importance': importance,
            'population': population,
            'jobs': jobs,
            'geometry': geometry
        }, crs='EPSG:3857')

        logger.info(f" Создано {num_stops} тестовых остановок:")
        logger.info(f"  - {num_key} ключевых остановок")
        logger.info(f"  - {num_connection} соединительных остановок")
        logger.info(f"  - {num_stops - num_key - num_connection} обычных остановок")

        return test_stops

    except Exception as e:
        logger.error(f"Ошибка создания тестовых данных: {e}")
        return gpd.GeoDataFrame()

if __name__ == "__main__":
  # Тестирование утилит.
    print("Тестирование утилит Neural BCO...")

  # Проверяем системные требования.
    print("\n1. Проверка системных требований:")
    requirements = check_system_requirements()
    print(f"Статус: {requirements['status']}")
    print(f"Python: {requirements['system_info']['python_version']}")
    print(f"Память: {requirements['hardware']['memory_gb']} GB")

    if requirements['recommendations']:
        print("Рекомендации:")
        for rec in requirements['recommendations']:
            print(f"  - {rec}")

  # Тест загрузки данных из opt_fin2.
    print("\n2. Тестирование загрузки данных:")
    opt_stops, metadata = load_opt_fin2_data()

    if metadata.get('error'):
        print(f"Ошибка загрузки: {metadata['error']}")
        print("Создаем тестовые данные...")
        opt_stops = create_test_opt_stops(num_stops=20)

    if len(opt_stops) > 0:
        print(f" Загружено {len(opt_stops)} остановок")

  # Валидация данных.
        print("\n3. Валидация данных:")
        validation = validate_opt_fin2_data(opt_stops)
        print(f"Валидность: {validation['is_valid']}")
        print(f"Качество: {validation['quality_score']:.1f}")

        if validation['warnings']:
            print("Предупреждения:")
            for warning in validation['warnings'][:3]:  # Показываем первые 3.
                print(f"  - {warning}")

  # Исправление данных если нужно.
        if not validation['is_valid'] or validation['warnings']:
            print("\n4. Исправление данных:")
            fixed_stops, repairs = repair_opt_fin2_data(opt_stops)
            print(f"Выполнено исправлений: {len(repairs)}")
            for repair in repairs[:3]:  # Показываем первые 3.
                print(f"  - {repair}")
            opt_stops = fixed_stops

  # Создаем сводку.
        print("\n5. Сводка по данным:")
        summary = create_data_summary(opt_stops)
        print(summary)

  # Тест создания графа и метрик.
    print("\n6. Тестирование графовых утилит:")

  # Создаем простой тестовый граф.
    nodes_data = [
        {'node_id': 0, 'geometry': Point(0, 0)},
        {'node_id': 1, 'geometry': Point(1000, 0)},
        {'node_id': 2, 'geometry': Point(0, 1000)},
        {'node_id': 3, 'geometry': Point(1000, 1000)}
    ]
    nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:3857')

    edges_data = [
        {'u': 0, 'v': 1, 'geometry': LineString([(0, 0), (1000, 0)])},
        {'u': 1, 'v': 3, 'geometry': LineString([(1000, 0), (1000, 1000)])},
        {'u': 3, 'v': 2, 'geometry': LineString([(1000, 1000), (0, 1000)])},
        {'u': 2, 'v': 0, 'geometry': LineString([(0, 1000), (0, 0)])}
    ]
    edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:3857')

  # Создаем граф.
    G = GraphUtils.create_graph_from_edges(edges_gdf, nodes_gdf)
    print(f" Создан граф: {len(G.nodes())} узлов, {len(G.edges())} ребер")

  # Тест метрик.
    test_routes = [[0, 1, 3], [2, 0]]
    od_matrix = DataUtils.create_od_matrix(nodes_gdf)

    passenger_cost = TransitMetrics.calculate_passenger_cost(G, test_routes, od_matrix)
    operator_cost = TransitMetrics.calculate_operator_cost(G, test_routes)
    violations = TransitMetrics.calculate_constraint_violations(test_routes, G)

    print(f" Метрики рассчитаны:")
    print(f"  Пассажирская стоимость: {passenger_cost:.2f}")
    print(f"  Операторская стоимость: {operator_cost:.2f}")
    print(f"  Нарушения ограничений: {violations:.2f}")

  # Тест сохранения результатов.
    print("\n7. Тестирование сохранения:")
    test_results = {
        'test_data': True,
        'num_stops': len(opt_stops),
        'graph_nodes': len(G.nodes()),
        'timestamp': datetime.now().isoformat()
    }

    saved_path = save_results(test_results, 'test_results.json')
    if saved_path:
        print(f" Результаты сохранены: {saved_path}")
    else:
        print(" Ошибка сохранения результатов")

    print("\n Тестирование завершено успешно!")
