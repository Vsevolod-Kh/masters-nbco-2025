  # Data_generator.py.
"""
Генератор синтетических городов для обучения Neural BCO
Реализует 5 типов графов из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"
"""

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree, distance_matrix
from typing import Dict, List, Tuple, Optional, Set
import random
import logging
from dataclasses import dataclass
import pickle
import matplotlib.pyplot as plt
import warnings

from config import config
from utils import DataUtils, GraphUtils, load_opt_fin2_data

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class CityInstance:
    """Структура для хранения экземпляра синтетического города"""
    nodes_gdf: gpd.GeoDataFrame
    edges_gdf: gpd.GeoDataFrame
    city_graph: nx.Graph
    od_matrix: np.ndarray
    graph_type: str
    city_bounds: Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax).
    metadata: Dict

    def to_dict(self) -> Dict:
        """Преобразовать в словарь для сохранения"""
        return {
            'nodes_coords': np.array([[p.x, p.y] for p in self.nodes_gdf.geometry]),
            'edges_list': [(row['u'], row['v']) for _, row in self.edges_gdf.iterrows()],
            'od_matrix': self.od_matrix,
            'graph_type': self.graph_type,
            'city_bounds': self.city_bounds,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """Восстановить из словаря"""
  # Создаем nodes_gdf.
        coords = data['nodes_coords']
        nodes_data = []
        for i, (x, y) in enumerate(coords):
            nodes_data.append({
                'node_id': i,
                'geometry': Point(x, y),
                'x': x,
                'y': y
            })
        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:3857')

  # Создаем edges_gdf.
        edges_data = []
        for u, v in data['edges_list']:
            edge_geom = LineString([coords[u], coords[v]])
            edges_data.append({
                'u': u,
                'v': v,
                'geometry': edge_geom,
                'length': edge_geom.length,
                'travel_time': edge_geom.length / config.data.vehicle_speed_ms
            })
        edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:3857')

  # Создаем граф.
        city_graph = GraphUtils.create_graph_from_edges(edges_gdf, nodes_gdf)

        return cls(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            city_graph=city_graph,
            od_matrix=data['od_matrix'],
            graph_type=data['graph_type'],
            city_bounds=data['city_bounds'],
            metadata=data['metadata']
        )

class SyntheticCityGenerator:
    """Генератор синтетических городов для обучения нейросети [[11]]"""

    def __init__(self,
                 num_nodes: int = 20,
                 city_area_km: float = 30.0,
                 vehicle_speed_ms: float = 15.0,
                 demand_range: Tuple[int, int] = (60, 800),
                 edge_deletion_prob: float = 0.1):
        """
        Args:
            num_nodes: количество узлов в синтетическом городе
            city_area_km: размер города в км (квадрат city_area_km x city_area_km)
            vehicle_speed_ms: скорость транспорта в м/с
            demand_range: диапазон спроса в OD матрице
            edge_deletion_prob: вероятность удаления ребра (ρ в статье)
        """
        self.num_nodes = num_nodes
        self.city_area_m = city_area_km * 1000  # В метрах.
        self.vehicle_speed_ms = vehicle_speed_ms
        self.demand_range = demand_range
        self.edge_deletion_prob = edge_deletion_prob

        logger.info(f"Инициализирован генератор городов: {num_nodes} узлов, {city_area_km}x{city_area_km} км")

    def generate_dataset(self, num_cities: int) -> List[CityInstance]:
        """
        Генерировать набор синтетических городов

        Args:
            num_cities: количество городов для генерации

        Returns:
            Список экземпляров городов
        """
        cities = []
        graph_types = config.data.graph_types

        logger.info(f"Генерация {num_cities} синтетических городов...")

        for i in range(num_cities):
  # Случайно выбираем тип графа.
            graph_type = random.choice(graph_types)

            try:
                city = self.generate_single_city(graph_type)
                cities.append(city)

                if (i + 1) % 1000 == 0:
                    logger.info(f"Сгенерировано {i + 1}/{num_cities} городов")

            except Exception as e:
                logger.warning(f"Ошибка генерации города {i} (тип {graph_type}): {e}")
                continue

        logger.info(f"Успешно сгенерировано {len(cities)} городов")
        return cities

    def generate_single_city(self, graph_type: str) -> CityInstance:
        """
        Генерировать один синтетический город

        Args:
            graph_type: тип графа ('incoming_4nn', 'outgoing_4nn', 'voronoi', '4_grid', '8_grid')

        Returns:
            Экземпляр города
        """
  # Генерируем узлы и ребра в зависимости от типа графа.
        if graph_type == 'incoming_4nn':
            nodes_coords, edges_list = self._generate_incoming_4nn()
        elif graph_type == 'outgoing_4nn':
            nodes_coords, edges_list = self._generate_outgoing_4nn()
        elif graph_type == 'voronoi':
            nodes_coords, edges_list = self._generate_voronoi()
        elif graph_type == '4_grid':
            nodes_coords, edges_list = self._generate_4_grid()
        elif graph_type == '8_grid':
            nodes_coords, edges_list = self._generate_8_grid()
        else:
            raise ValueError(f"Неизвестный тип графа: {graph_type}")

  # Удаляем случайные ребра (кроме voronoi).
        if graph_type != 'voronoi' and self.edge_deletion_prob > 0:
            edges_list = self._delete_random_edges(edges_list, nodes_coords)

  # Проверяем связность.
        if not self._is_connected(nodes_coords, edges_list):
  # Если граф несвязный, пробуем еще раз.
            return self.generate_single_city(graph_type)

  # Создаем GeoDataFrames.
        nodes_gdf, edges_gdf = self._create_geodataframes(nodes_coords, edges_list)

  # Создаем NetworkX граф.
        city_graph = GraphUtils.create_graph_from_edges(edges_gdf, nodes_gdf)

  # Генерируем OD матрицу.
        od_matrix = self._generate_od_matrix(len(nodes_coords))

  # Границы города.
        coords_array = np.array(nodes_coords)
        city_bounds = (
            coords_array[:, 0].min(),
            coords_array[:, 1].min(),
            coords_array[:, 0].max(),
            coords_array[:, 1].max()
        )

  # Метаданные.
        metadata = {
            'num_nodes': len(nodes_coords),
            'num_edges': len(edges_list),
            'graph_type': graph_type,
            'connected_components': nx.number_connected_components(city_graph),
            'avg_degree': np.mean([d for n, d in city_graph.degree()]),
            'total_demand': od_matrix.sum()
        }

        return CityInstance(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            city_graph=city_graph,
            od_matrix=od_matrix,
            graph_type=graph_type,
            city_bounds=city_bounds,
            metadata=metadata
        )

    def _generate_incoming_4nn(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """Генерация графа Incoming 4-nearest neighbors [[11]]"""
  # Генерируем случайные точки в квадрате.
        nodes_coords = []
        for _ in range(self.num_nodes):
            x = random.uniform(0, self.city_area_m)
            y = random.uniform(0, self.city_area_m)
            nodes_coords.append((x, y))

  # Для каждого узла добавляем ребра ОТ его 4 ближайших соседей.
        coords_array = np.array(nodes_coords)
        tree = KDTree(coords_array)

        edges_set = set()
        for i in range(len(nodes_coords)):
  # Находим 5 ближайших (включая сам узел).
            distances, indices = tree.query(coords_array[i], k=min(5, len(nodes_coords)))

  # Берем 4 ближайших (исключая сам узел).
            neighbors = [idx for idx in indices[1:5] if idx != i]

  # Добавляем ребра ОТ соседей К текущему узлу.
            for neighbor in neighbors:
                if neighbor < len(nodes_coords):
                    edges_set.add((neighbor, i))

        return nodes_coords, list(edges_set)

    def _generate_outgoing_4nn(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """Генерация графа Outgoing 4-nearest neighbors [[11]]"""
  # Генерируем случайные точки в квадрате.
        nodes_coords = []
        for _ in range(self.num_nodes):
            x = random.uniform(0, self.city_area_m)
            y = random.uniform(0, self.city_area_m)
            nodes_coords.append((x, y))

  # Для каждого узла добавляем ребра К его 4 ближайшим соседям.
        coords_array = np.array(nodes_coords)
        tree = KDTree(coords_array)

        edges_set = set()
        for i in range(len(nodes_coords)):
  # Находим 5 ближайших (включая сам узел).
            distances, indices = tree.query(coords_array[i], k=min(5, len(nodes_coords)))

  # Берем 4 ближайших (исключая сам узел).
            neighbors = [idx for idx in indices[1:5] if idx != i]

  # Добавляем ребра ОТ текущего узла К соседям.
            for neighbor in neighbors:
                if neighbor < len(nodes_coords):
                    edges_set.add((i, neighbor))

        return nodes_coords, list(edges_set)

    def _generate_voronoi(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """Генерация графа на основе диаграммы Вороного [[11]]"""
  # Генерируем больше точек, чем нужно, чтобы получить нужное количество узлов Вороного.
        m = int(self.num_nodes * config.data.voronoi_seed_multiplier)

        seed_points = []
        for _ in range(m):
            x = random.uniform(0, self.city_area_m)
            y = random.uniform(0, self.city_area_m)
            seed_points.append((x, y))

  # Строим диаграмму Вороного.
        vor = Voronoi(seed_points)

  # Извлекаем вершины как узлы графа.
        vertices = vor.vertices

  # Фильтруем вершины, которые находятся в пределах области.
        valid_vertices = []
        vertex_mapping = {}  # Старый индекс -> новый индекс.

        for i, vertex in enumerate(vertices):
            x, y = vertex
            if (0 <= x <= self.city_area_m and 0 <= y <= self.city_area_m):
                vertex_mapping[i] = len(valid_vertices)
                valid_vertices.append((x, y))

  # Если получилось слишком много или мало узлов, корректируем.
        if len(valid_vertices) < self.num_nodes:
  # Добавляем случайные узлы.
            while len(valid_vertices) < self.num_nodes:
                x = random.uniform(0, self.city_area_m)
                y = random.uniform(0, self.city_area_m)
                valid_vertices.append((x, y))
        elif len(valid_vertices) > self.num_nodes:
  # Случайно выбираем подмножество.
            indices = random.sample(range(len(valid_vertices)), self.num_nodes)
            valid_vertices = [valid_vertices[i] for i in indices]
  # Обновляем маппинг.
            new_mapping = {}
            for old_idx, new_idx in vertex_mapping.items():
                if new_idx in indices:
                    new_mapping[old_idx] = indices.index(new_idx)
            vertex_mapping = new_mapping

  # Извлекаем ребра из структуры Вороного.
        edges_set = set()
        for ridge in vor.ridge_vertices:
            if len(ridge) == 2 and ridge[0] != -1 and ridge[1] != -1:
                v1, v2 = ridge
                if v1 in vertex_mapping and v2 in vertex_mapping:
                    new_v1 = vertex_mapping[v1]
                    new_v2 = vertex_mapping[v2]
                    if new_v1 < len(valid_vertices) and new_v2 < len(valid_vertices):
                        edges_set.add((min(new_v1, new_v2), max(new_v1, new_v2)))

        return valid_vertices, list(edges_set)

    def _generate_4_grid(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """Генерация 4-связной сетки [[11]]"""
  # Определяем размеры сетки как можно ближе к квадрату.
        grid_width = int(np.sqrt(self.num_nodes))
        grid_height = int(np.ceil(self.num_nodes / grid_width))

  # Если получается слишком много узлов, корректируем.
        while grid_width * grid_height > self.num_nodes + grid_width:
            if grid_width > grid_height:
                grid_width -= 1
            else:
                grid_height -= 1

  # Размещаем узлы в сетке.
        nodes_coords = []
        node_grid = {}  # (i, j) -> node_index.

        spacing_x = self.city_area_m / (grid_width + 1)
        spacing_y = self.city_area_m / (grid_height + 1)

        node_idx = 0
        for i in range(grid_height):
            for j in range(grid_width):
                if node_idx < self.num_nodes:
                    x = spacing_x * (j + 1)
                    y = spacing_y * (i + 1)
                    nodes_coords.append((x, y))
                    node_grid[(i, j)] = node_idx
                    node_idx += 1

  # Добавляем горизонтальные и вертикальные связи.
        edges_list = []
        for i in range(grid_height):
            for j in range(grid_width):
                if (i, j) in node_grid:
                    current = node_grid[(i, j)]

  # Горизонтальная связь вправо.
                    if j + 1 < grid_width and (i, j + 1) in node_grid:
                        right = node_grid[(i, j + 1)]
                        edges_list.append((current, right))

  # Вертикальная связь вниз.
                    if i + 1 < grid_height and (i + 1, j) in node_grid:
                        down = node_grid[(i + 1, j)]
                        edges_list.append((current, down))

        return nodes_coords, edges_list

    def _generate_8_grid(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """Генерация 8-связной сетки [[11]]"""
  # Сначала создаем 4-связную сетку.
        nodes_coords, edges_list = self._generate_4_grid()

  # Определяем размеры сетки.
        grid_width = int(np.sqrt(self.num_nodes))
        grid_height = int(np.ceil(self.num_nodes / grid_width))

        while grid_width * grid_height > self.num_nodes + grid_width:
            if grid_width > grid_height:
                grid_width -= 1
            else:
                grid_height -= 1

  # Воссоздаем маппинг узлов.
        node_grid = {}
        node_idx = 0
        for i in range(grid_height):
            for j in range(grid_width):
                if node_idx < len(nodes_coords):
                    node_grid[(i, j)] = node_idx
                    node_idx += 1

  # Добавляем диагональные связи.
        edges_set = set(edges_list)
        for i in range(grid_height):
            for j in range(grid_width):
                if (i, j) in node_grid:
                    current = node_grid[(i, j)]

  # Диагональ вправо-вниз.
                    if (i + 1 < grid_height and j + 1 < grid_width and
                            (i + 1, j + 1) in node_grid):
                        diag_rd = node_grid[(i + 1, j + 1)]
                        edges_set.add((current, diag_rd))

  # Диагональ влево-вниз.
                    if (i + 1 < grid_height and j - 1 >= 0 and
                            (i + 1, j - 1) in node_grid):
                        diag_ld = node_grid[(i + 1, j - 1)]
                        edges_set.add((current, diag_ld))

        return nodes_coords, list(edges_set)

    def _delete_random_edges(self,
                             edges_list: List[Tuple[int, int]],
                             nodes_coords: List[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """Случайное удаление ребер с вероятностью ρ [[11]]"""
        filtered_edges = []

        for edge in edges_list:
            if random.random() > self.edge_deletion_prob:
                filtered_edges.append(edge)

        return filtered_edges

    def _is_connected(self,
                      nodes_coords: List[Tuple[float, float]],
                      edges_list: List[Tuple[int, int]]) -> bool:
        """Проверка связности графа"""
        if len(nodes_coords) <= 1:
            return True

  # Создаем временный граф для проверки.
        G = nx.Graph()
        G.add_nodes_from(range(len(nodes_coords)))
        G.add_edges_from(edges_list)

        return nx.is_connected(G)

    def _create_geodataframes(self,
                              nodes_coords: List[Tuple[float, float]],
                              edges_list: List[Tuple[int, int]]) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Создание GeoDataFrame для узлов и ребер"""
  # Создаем узлы.
        nodes_data = []
        for i, (x, y) in enumerate(nodes_coords):
            nodes_data.append({
                'node_id': i,
                'geometry': Point(x, y),
                'x': x,
                'y': y
            })
        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:3857')

  # Создаем ребра.
        edges_data = []
        for u, v in edges_list:
            if u < len(nodes_coords) and v < len(nodes_coords):
                start_point = nodes_coords[u]
                end_point = nodes_coords[v]
                edge_geom = LineString([start_point, end_point])

                edge_length = edge_geom.length
                travel_time = edge_length / self.vehicle_speed_ms

                edges_data.append({
                    'u': u,
                    'v': v,
                    'geometry': edge_geom,
                    'length': edge_length,
                    'travel_time': travel_time
                })

        edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:3857')
        return nodes_gdf, edges_gdf

    def _generate_od_matrix(self, num_nodes: int) -> np.ndarray:
        """Генерация матрицы Origin-Destination спроса [[11]]"""
        od_matrix = np.zeros((num_nodes, num_nodes))

  # Генерируем случайный спрос для недиагональных элементов.
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    demand = random.randint(self.demand_range[0], self.demand_range[1])
                    od_matrix[i, j] = demand

        return od_matrix

class DatasetManager:
    """Менеджер для создания, сохранения и загрузки наборов данных"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or config.files.data_dir

    def create_training_dataset(self,
                                num_cities: int = None,
                                save_path: str = None) -> List[CityInstance]:
        """
        Создать и сохранить набор данных для обучения

        Args:
            num_cities: количество городов (по умолчанию из конфигурации)
            save_path: путь для сохранения

        Returns:
            Список экземпляров городов
        """
        if num_cities is None:
            num_cities = config.data.num_training_cities

        generator = SyntheticCityGenerator(
            num_nodes=config.data.num_nodes,
            city_area_km=config.data.city_area_km,
            vehicle_speed_ms=config.data.vehicle_speed_ms,
            demand_range=config.data.demand_range,
            edge_deletion_prob=config.data.edge_deletion_prob
        )

        logger.info(f"Создание обучающего набора данных из {num_cities} городов...")
        cities = generator.generate_dataset(num_cities)

  # Разделяем на обучающую и валидационную выборки.
        split_idx = int(len(cities) * config.data.train_val_split)
        train_cities = cities[:split_idx]
        val_cities = cities[split_idx:]

        if save_path is None:
            save_path = f"{self.data_dir}/training_cities_{num_cities}.pkl"

        self.save_dataset({
            'train': train_cities,
            'val': val_cities,
            'metadata': {
                'total_cities': len(cities),
                'train_cities': len(train_cities),
                'val_cities': len(val_cities),
                'num_nodes': config.data.num_nodes,
                'graph_types': config.data.graph_types
            }
        }, save_path)

        logger.info(f"Набор данных сохранен в {save_path}")
        logger.info(f"Обучающих городов: {len(train_cities)}, валидационных: {len(val_cities)}")

        return cities

    def save_dataset(self, dataset: Dict, filepath: str):
        """Сохранить набор данных в файл"""
  # Преобразуем CityInstance в dict для сериализации.
        serializable_dataset = {}

        for split_name, cities in dataset.items():
            if split_name == 'metadata':
                serializable_dataset[split_name] = cities
            else:
                serializable_dataset[split_name] = [city.to_dict() for city in cities]

        with open(filepath, 'wb') as f:
            pickle.dump(serializable_dataset, f)

    def load_dataset(self, filepath: str) -> Dict[str, List[CityInstance]]:
        """Загрузить набор данных из файла"""
        with open(filepath, 'rb') as f:
            dataset = pickle.load(f)

  # Восстанавливаем CityInstance из dict.
        loaded_dataset = {}

        for split_name, data in dataset.items():
            if split_name == 'metadata':
                loaded_dataset[split_name] = data
            else:
                loaded_dataset[split_name] = [CityInstance.from_dict(city_data) for city_data in data]

        return loaded_dataset

    def create_augmented_batch(self, cities: List[CityInstance], batch_size: int) -> List[CityInstance]:
        """
        Создать batch с аугментацией данных [[11]]

        Args:
            cities: исходные города
            batch_size: размер batch

        Returns:
            Список аугментированных городов
        """
        batch = []

        for _ in range(batch_size):
  # Случайно выбираем город.
            base_city = random.choice(cities)

  # Применяем аугментацию.
            augmented_city = self._augment_city(base_city)
            batch.append(augmented_city)

        return batch

    def _augment_city(self, city: CityInstance) -> CityInstance:
        """
        Применить аугментацию данных к городу [[11]]

        Включает:
        - Масштабирование координат
        - Поворот
        - Масштабирование спроса
        """
  # Извлекаем координаты узлов.
        coords = np.array([[p.x, p.y] for p in city.nodes_gdf.geometry])
        od_matrix = city.od_matrix.copy()

  # Применяем аугментацию.
        augmented_coords, augmented_od = DataUtils.augment_city_data(
            coords, od_matrix,
            scale_range=config.data.scale_range,
            rotation_range=config.data.rotation_range,
            demand_scale_range=config.data.demand_scale_range
        )

  # Создаем новый экземпляр города с аугментированными данными.
        augmented_nodes_data = []
        for i, (x, y) in enumerate(augmented_coords):
            augmented_nodes_data.append({
                'node_id': i,
                'geometry': Point(x, y),
                'x': x,
                'y': y
            })
        augmented_nodes_gdf = gpd.GeoDataFrame(augmented_nodes_data, crs='EPSG:3857')

  # Создаем новые ребра с аугментированными координатами.
        augmented_edges_data = []
        for _, edge in city.edges_gdf.iterrows():
            u, v = edge['u'], edge['v']
            start_point = augmented_coords[u]
            end_point = augmented_coords[v]
            edge_geom = LineString([start_point, end_point])

            augmented_edges_data.append({
                'u': u,
                'v': v,
                'geometry': edge_geom,
                'length': edge_geom.length,
                'travel_time': edge_geom.length / config.data.vehicle_speed_ms
            })

        augmented_edges_gdf = gpd.GeoDataFrame(augmented_edges_data, crs='EPSG:3857')

  # Создаем новый граф.
        augmented_graph = GraphUtils.create_graph_from_edges(augmented_edges_gdf, augmented_nodes_gdf)

  # Обновляем границы.
        augmented_bounds = (
            augmented_coords[:, 0].min(),
            augmented_coords[:, 1].min(),
            augmented_coords[:, 0].max(),
            augmented_coords[:, 1].max()
        )

        return CityInstance(
            nodes_gdf=augmented_nodes_gdf,
            edges_gdf=augmented_edges_gdf,
            city_graph=augmented_graph,
            od_matrix=augmented_od,
            graph_type=city.graph_type + '_augmented',
            city_bounds=augmented_bounds,
            metadata=city.metadata.copy()
        )

def load_opt_fin2_city() -> Optional[CityInstance]:
    """
    Загружает данные из opt_fin2 и создает CityInstance с простым подходом 3 ближайших соседей
    """
    try:
        opt_stops, metadata = load_opt_fin2_data()

        if len(opt_stops) == 0:
            logger.warning("Нет данных opt_fin2 для создания города")
            return None

        logger.info(f"Загружено {len(opt_stops)} остановок из opt_stops.pkl")

  # ОТКАТ: Убираем сложный расчет среднего расстояния, оставляем простую диагностику.
        logger.info("📏 Базовая диагностика данных...")

  # Убеждаемся, что координаты в метрах (UTM).
        opt_stops_utm = opt_stops.to_crs('EPSG:32636')  # UTM для СПб.

  # Извлекаем координаты.
        coords = np.array([(point.x, point.y) for point in opt_stops_utm.geometry])

        logger.info(f"📊 Система координат: EPSG:32636 (UTM)")
        logger.info(f"📍 Координаты загружены: {len(coords)} точек")

  # Анализ пространственного распределения.
        if len(coords) > 1:
            bounds = opt_stops_utm.total_bounds
            area_km2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1_000_000
            density = len(opt_stops) / area_km2 if area_km2 > 0 else 0

            logger.info("🗺️ === АНАЛИЗ ПРОСТРАНСТВЕННОГО РАСПРЕДЕЛЕНИЯ ===")
            logger.info(f"   📍 Общая площадь: {area_km2:.2f} км²")
            logger.info(f"   🚏 Плотность остановок: {density:.1f} остановок/км²")

  # Преобразуем координаты в единую систему для работы.
        opt_stops = opt_stops_utm  # Используем UTM координаты.

  # Создаем узлы графа.
        nodes = []
        node_mapping = {}  # Словарь для сопоставления исходных ID с новыми.

        for idx, stop in opt_stops.iterrows():
            node_id = len(nodes)  # Последовательная нумерация узлов.
            node_mapping[stop['node_id']] = node_id

            nodes.append({
                'node_id': node_id,
                'original_id': stop['node_id'],
                'geometry': stop.geometry,
                'x': stop.geometry.x,
                'y': stop.geometry.y,
                'population': stop.get('population', 0),
                'jobs': stop.get('jobs', 0),
                'type': stop.get('type', 'ordinary')
            })

        logger.info(f"Создано {len(nodes)} узлов")

  # ОТКАТ К ПРОСТОМУ ПОДХОДУ: 3 БЛИЖАЙШИХ СОСЕДА.
        from scipy.spatial import KDTree

  # Создаем массив координат для KDTree.
        coords_array = np.array([(node['x'], node['y']) for node in nodes])
        tree = KDTree(coords_array)

        logger.info("🔗 Создание ребер с простым подходом (3 ближайших соседа)...")

        edges_set = set()

  # ПРОСТАЯ СВЯЗНОСТЬ: каждый узел соединен с 3 ближайшими соседями.
        for i in range(len(coords_array)):
  # Соединяем с 3 ближайшими соседями.
            distances, indices = tree.query(coords_array[i], k=min(4, len(coords_array)))
            for neighbor in indices[1:]:  # Исключаем сам узел.
                if neighbor < len(coords_array):
                    edges_set.add((min(i, neighbor), max(i, neighbor)))

        logger.info(f"   Базовых соединений создано: {len(edges_set)}")

  # Логируем типы остановок для информации.
        if hasattr(opt_stops, 'columns') and 'type' in opt_stops.columns:
            key_stops_count = len([n for n in nodes if n.get('type') == 'key'])
            connection_stops_count = len([n for n in nodes if n.get('type') == 'connection'])
            ordinary_stops_count = len([n for n in nodes if n.get('type') == 'ordinary'])

            logger.info(f"   Ключевых остановок: {key_stops_count}")
            logger.info(f"   Пересадочных остановок: {connection_stops_count}")
            logger.info(f"   Обычных остановок: {ordinary_stops_count}")

        logger.info(f"   Итого ребер создано: {len(edges_set)}")
        logger.info(f"   Средняя степень узла: {(len(edges_set) * 2) / len(coords_array):.1f}")

  # Проверяем связность.
        G_temp = nx.Graph()
        G_temp.add_nodes_from(range(len(coords_array)))
        G_temp.add_edges_from(edges_set)

        final_connected = nx.is_connected(G_temp)
        logger.info(f"🔗 Финальная связность графа: {'Да' if final_connected else 'Нет'}")

  # Создаем финальный граф NetworkX.
        city_graph = nx.Graph()

  # Добавляем узлы с атрибутами.
        for node in nodes:
            city_graph.add_node(
                node['node_id'],
                x=node['x'],
                y=node['y'],
                population=node['population'],
                jobs=node['jobs'],
                type=node['type']
            )

  # Добавляем ребра с весами (время поездки).
        for u, v in edges_set:
            distance = np.linalg.norm(coords_array[u] - coords_array[v])
            travel_time = distance / 15.0  # 15 м/с = скорость автобуса.
            city_graph.add_edge(u, v, distance=distance, travel_time=travel_time)

        logger.info(f"Финальный граф: {len(city_graph.nodes())} узлов, {len(city_graph.edges())} ребер")

  # Создаем GeoDataFrames.
        nodes_data = []
        for node in nodes:
            nodes_data.append({
                'node_id': node['node_id'],
                'geometry': node['geometry'],
                'x': node['x'],
                'y': node['y'],
                'population': node['population'],
                'jobs': node['jobs'],
                'type': node['type']
            })
        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:32636')

        edges_data = []
        for u, v in edges_set:
            start_point = coords_array[u]
            end_point = coords_array[v]
            edge_geom = LineString([start_point, end_point])
            distance = np.linalg.norm(coords_array[u] - coords_array[v])

            edges_data.append({
                'u': u,
                'v': v,
                'geometry': edge_geom,
                'length': distance,
                'travel_time': distance / 15.0
            })
        edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:32636')

  # Создаем OD матрицу на основе населения и рабочих мест.
        logger.info("Создание OD матрицы...")

        num_nodes = len(nodes)
        od_matrix = np.zeros((num_nodes, num_nodes))

  # Простая модель: спрос пропорционален произведению населения источника и рабочих мест назначения.
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    origin_pop = nodes[i]['population']
                    dest_jobs = nodes[j]['jobs']

  # Базовый спрос + компонента на основе населения и работ.
                    base_demand = 10
                    pop_demand = origin_pop * 0.1  # 10% населения путешествует.
                    job_demand = dest_jobs * 0.15  # 15% привлекательности рабочих мест.

                    total_demand = base_demand + pop_demand + job_demand

  # Добавляем случайность.
                    noise = np.random.normal(0, total_demand * 0.1)
                    od_matrix[i, j] = max(0, total_demand + noise)

  # Нормализуем матрицу спроса.
        total_demand = od_matrix.sum()
        if total_demand > 0:
            od_matrix = (od_matrix / total_demand) * (len(nodes) * 100)  # Целевой общий спрос.

        logger.info(f"OD матрица создана: общий спрос {od_matrix.sum():.0f} поездок/день")

  # Определяем тип графа.
        avg_degree = (len(edges_set) * 2) / len(nodes)
        if avg_degree < 3:
            graph_type = "sparse"
        elif avg_degree < 6:
            graph_type = "medium_density"
        else:
            graph_type = "dense"

  # Создаем массив координат для city_bounds.
        coords_array = np.array([(node['x'], node['y']) for node in nodes])

  # Создаем CityInstance с правильными параметрами.
        city_instance = CityInstance(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            city_graph=city_graph,
            od_matrix=od_matrix,
            graph_type=graph_type,
            city_bounds=(
                coords_array[:, 0].min(),
                coords_array[:, 1].min(),
                coords_array[:, 0].max(),
                coords_array[:, 1].max()
            ),
            metadata={
                'source': 'opt_fin2',
                'creation_method': 'simple_3_neighbors',
                'total_population': sum(node['population'] for node in nodes),
                'total_jobs': sum(node['jobs'] for node in nodes),
                'stop_types': {
                    'key': len([n for n in nodes if n.get('type') == 'key']),
                    'connection': len([n for n in nodes if n.get('type') == 'connection']),
                    'ordinary': len([n for n in nodes if n.get('type') == 'ordinary'])
                }
            }
        )

        logger.info("✅ CityInstance успешно создан с простым подходом 3 соседей")
        return city_instance

    except Exception as e:
        logger.error(f"Ошибка при создании города из opt_fin2: {e}")
        return None

def visualize_city_examples():
    """Создать примеры каждого типа города для визуализации"""
    generator = SyntheticCityGenerator(num_nodes=20)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    graph_types = ['incoming_4nn', 'outgoing_4nn', 'voronoi', '4_grid', '8_grid']

    for i, graph_type in enumerate(graph_types):
        city = generator.generate_single_city(graph_type)

        ax = axes[i]

  # Рисуем ребра.
        for _, edge in city.edges_gdf.iterrows():
            line = edge.geometry
            x, y = line.xy
            ax.plot(x, y, 'gray', alpha=0.6, linewidth=1)

  # Рисуем узлы.
        coords = np.array([[p.x, p.y] for p in city.nodes_gdf.geometry])
        ax.scatter(coords[:, 0], coords[:, 1], c='red', s=30, zorder=5)

        ax.set_title(f'{graph_type}\n({len(city.nodes_gdf)} узлов, {len(city.edges_gdf)} ребер)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

  # Скрываем последний subplot если нечетное количество.
    if len(graph_types) < len(axes):
        axes[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig('synthetic_city_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
  # Демонстрация генератора синтетических городов.
    logger.info("Демонстрация генератора синтетических городов...")

  # Создаем примеры городов.
    logger.info("Создание примеров разных типов городов...")
    visualize_city_examples()

  # Тестируем создание небольшого набора данных.
    logger.info("Создание тестового набора данных...")
    dataset_manager = DatasetManager()

  # Создаем маленький набор данных для тестирования.
    test_cities = dataset_manager.create_training_dataset(num_cities=10,
                                                          save_path="test_cities.pkl")

    logger.info(f"Создано {len(test_cities)} тестовых городов")

  # Статистика по типам графов.
    graph_type_counts = {}
    for city in test_cities:
        graph_type = city.graph_type
        graph_type_counts[graph_type] = graph_type_counts.get(graph_type, 0) + 1

    logger.info("Распределение по типам графов:")
    for graph_type, count in graph_type_counts.items():
        logger.info(f"  {graph_type}: {count}")

  # Тестируем аугментацию.
    logger.info("Тестирование аугментации данных...")
    original_city = test_cities[0]
    augmented_city = dataset_manager._augment_city(original_city)

    logger.info(f"Исходный город: {len(original_city.nodes_gdf)} узлов, общий спрос: {original_city.od_matrix.sum()}")
    logger.info(
        f"Аугментированный город: {len(augmented_city.nodes_gdf)} узлов, общий спрос: {augmented_city.od_matrix.sum()}")

  # Тестируем загрузку данных из opt_fin2.
    logger.info("Тестирование загрузки данных из opt_fin2...")
    real_city = load_opt_fin2_city()
    if real_city:
        logger.info(f"Загружен реальный город: {len(real_city.nodes_gdf)} узлов, {len(real_city.edges_gdf)} ребер")
        logger.info(f"Метаданные: {real_city.metadata}")
    else:
        logger.info("Не удалось загрузить данные из opt_fin2.py")

    logger.info("Демонстрация завершена успешно!")
