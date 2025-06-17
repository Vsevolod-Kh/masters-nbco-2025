  # Evaluation.py.
"""
Система оценки алгоритмов на стандартных бенчмарках для Neural BCO
Реализует оценку на датасетах Mandl и Mumford из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"
"""

import os
import time
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, LineString

from config import config
from neural_planner import NeuralPlanner, NeuralPlannerTrainer
from bco_algorithm import BeeColonyOptimization, run_bco_optimization
from neural_bco import NeuralBeeColonyOptimization, run_neural_bco_optimization
from cost_functions import TransitCostCalculator, compare_route_sets, evaluate_cost_sensitivity
from data_generator import CityInstance

  # Настройка логирования.
logger = logging.getLogger(__name__)

  # Подавляем предупреждения.
warnings.filterwarnings('ignore')

class BenchmarkCity:
    """Класс для представления города из бенчмарка"""

    def __init__(self,
                 name: str,
                 nodes_coords: np.ndarray,
                 edges_list: List[Tuple[int, int]],
                 od_matrix: np.ndarray,
                 constraints: Dict[str, int],
                 area_km2: float = None):
        """
        Args:
            name: название города
            nodes_coords: координаты узлов (N x 2)
            edges_list: список ребер [(u, v)]
            od_matrix: матрица спроса (N x N)
            constraints: ограничения {'S': routes, 'MIN': min_length, 'MAX': max_length}
            area_km2: площадь города в км²
        """
        self.name = name
        self.nodes_coords = nodes_coords
        self.edges_list = edges_list
        self.od_matrix = od_matrix
        self.constraints = constraints
        self.area_km2 = area_km2

  # Создаем граф NetworkX.
        self.city_graph = self._create_networkx_graph()

  # Создаем CityInstance для совместимости с другими компонентами.
        self.city_instance = self._create_city_instance()

        logger.info(f"Создан бенчмарк {name}: {len(self.nodes_coords)} узлов, "
                    f"{len(self.edges_list)} ребер, S={constraints.get('S', 'N/A')}")

    def _create_networkx_graph(self) -> nx.Graph:
        """Создать граф NetworkX из данных бенчмарка"""
        G = nx.Graph()

  # Добавляем узлы.
        for i, (x, y) in enumerate(self.nodes_coords):
            G.add_node(i, x=x, y=y)

  # Добавляем ребра с временами поездки.
        for u, v in self.edges_list:
            if u < len(self.nodes_coords) and v < len(self.nodes_coords):
  # Вычисляем евклидово расстояние.
                x1, y1 = self.nodes_coords[u]
                x2, y2 = self.nodes_coords[v]
                distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

  # Конвертируем в время поездки (предполагаем расстояние в метрах).
                travel_time = distance / config.data.vehicle_speed_ms

                G.add_edge(u, v,
                           length=distance,
                           travel_time=travel_time)

        return G

    def _create_city_instance(self) -> CityInstance:
        """Создать CityInstance для совместимости"""
  # Создаем nodes_gdf.
        nodes_data = []
        for i, (x, y) in enumerate(self.nodes_coords):
            nodes_data.append({
                'node_id': i,
                'geometry': Point(x, y),
                'x': x,
                'y': y
            })
        nodes_gdf = gpd.GeoDataFrame(nodes_data, crs='EPSG:3857')

  # Создаем edges_gdf.
        edges_data = []
        for u, v in self.edges_list:
            if u < len(self.nodes_coords) and v < len(self.nodes_coords):
                start_point = self.nodes_coords[u]
                end_point = self.nodes_coords[v]
                edge_geom = LineString([start_point, end_point])

                edges_data.append({
                    'u': u,
                    'v': v,
                    'geometry': edge_geom,
                    'length': edge_geom.length,
                    'travel_time': edge_geom.length / config.data.vehicle_speed_ms
                })

        edges_gdf = gpd.GeoDataFrame(edges_data, crs='EPSG:3857')

  # Вычисляем границы.
        coords_array = np.array(self.nodes_coords)
        city_bounds = (
            coords_array[:, 0].min(),
            coords_array[:, 1].min(),
            coords_array[:, 0].max(),
            coords_array[:, 1].max()
        )

        metadata = {
            'source': f'benchmark_{self.name}',
            'num_nodes': len(self.nodes_coords),
            'num_edges': len(self.edges_list),
            'constraints': self.constraints,
            'area_km2': self.area_km2
        }

        return CityInstance(
            nodes_gdf=nodes_gdf,
            edges_gdf=edges_gdf,
            city_graph=self.city_graph,
            od_matrix=self.od_matrix,
            graph_type=f'benchmark_{self.name}',
            city_bounds=city_bounds,
            metadata=metadata
        )

class BenchmarkLoader:
    """Загрузчик стандартных бенчмарков Mandl и Mumford"""

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or f"{config.files.data_dir}/benchmarks"
        os.makedirs(self.data_dir, exist_ok=True)

  # Статические данные бенчмарков из статьи [[12]].
        self.benchmark_specs = {
            'mandl': {
                'nodes': 15,
                'edges': 20,
                'routes': 6,
                'min_length': 2,
                'max_length': 8,
                'area_km2': 352.7
            },
            'mumford0': {
                'nodes': 30,
                'edges': 90,
                'routes': 12,
                'min_length': 2,
                'max_length': 15,
                'area_km2': 354.2
            },
            'mumford1': {
                'nodes': 70,
                'edges': 210,
                'routes': 15,
                'min_length': 10,
                'max_length': 30,
                'area_km2': 858.5
            },
            'mumford2': {
                'nodes': 110,
                'edges': 385,
                'routes': 56,
                'min_length': 10,
                'max_length': 22,
                'area_km2': 1394.3
            },
            'mumford3': {
                'nodes': 127,
                'edges': 425,
                'routes': 60,
                'min_length': 12,
                'max_length': 25,
                'area_km2': 1703.2
            }
        }

    def load_benchmark(self, benchmark_name: str) -> BenchmarkCity:
        """
        Загрузить конкретный бенчмарк

        Args:
            benchmark_name: имя бенчмарка ('mandl', 'mumford0', etc.)

        Returns:
            BenchmarkCity объект
        """
        if benchmark_name not in self.benchmark_specs:
            raise ValueError(f"Неизвестный бенчмарк: {benchmark_name}")

  # Проверяем, есть ли сохраненный файл.
        benchmark_file = f"{self.data_dir}/{benchmark_name}.pkl"

        if os.path.exists(benchmark_file):
            logger.info(f"Загрузка бенчмарка {benchmark_name} из файла")
            return self._load_from_file(benchmark_file)
        else:
            logger.info(f"Создание синтетического бенчмарка {benchmark_name}")
            return self._create_synthetic_benchmark(benchmark_name)

    def _load_from_file(self, filepath: str) -> BenchmarkCity:
        """Загрузить бенчмарк из файла"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return BenchmarkCity(
            name=data['name'],
            nodes_coords=data['nodes_coords'],
            edges_list=data['edges_list'],
            od_matrix=data['od_matrix'],
            constraints=data['constraints'],
            area_km2=data.get('area_km2')
        )

    def _create_synthetic_benchmark(self, benchmark_name: str) -> BenchmarkCity:
        """Создать синтетический бенчмарк с параметрами из статьи"""
        spec = self.benchmark_specs[benchmark_name]

  # Генерируем узлы в квадрате (имитируем реальные города).
        area_side = np.sqrt(spec['area_km2'] * 1000000)  # В метрах.

  # Случайное размещение узлов.
        np.random.seed(42 + hash(benchmark_name) % 1000)  # Воспроизводимость.
        nodes_coords = np.random.uniform(0, area_side, (spec['nodes'], 2))

  # Создаем граф на основе близости (k-nearest neighbors).
        from scipy.spatial import KDTree
        tree = KDTree(nodes_coords)

  # Определяем среднее количество соседей на узел.
        target_edges = spec['edges']
        avg_degree = 2 * target_edges / spec['nodes']
        k = max(2, int(avg_degree * 1.5))  # С запасом для связности.

        edges_set = set()
        for i in range(spec['nodes']):
  # Находим k ближайших соседей.
            distances, indices = tree.query(nodes_coords[i], k=min(k + 1, spec['nodes']))

  # Соединяем с ближайшими соседями (исключая самого себя).
            for j in indices[1:]:
                if len(edges_set) < target_edges and i != j:
                    edges_set.add((min(i, j), max(i, j)))

        edges_list = list(edges_set)

  # Генерируем OD матрицу.
        od_matrix = np.zeros((spec['nodes'], spec['nodes']))
        for i in range(spec['nodes']):
            for j in range(spec['nodes']):
                if i != j:
  # Спрос обратно пропорционален расстоянию.
                    distance = np.linalg.norm(nodes_coords[i] - nodes_coords[j])
                    max_distance = area_side * np.sqrt(2)
                    demand_factor = 1.0 - (distance / max_distance)

                    base_demand = np.random.randint(
                        config.data.demand_range[0],
                        config.data.demand_range[1]
                    )
                    od_matrix[i, j] = max(1, int(base_demand * demand_factor))

        constraints = {
            'S': spec['routes'],
            'MIN': spec['min_length'],
            'MAX': spec['max_length']
        }

  # Создаем и сохраняем бенчмарк.
        benchmark = BenchmarkCity(
            name=benchmark_name,
            nodes_coords=nodes_coords,
            edges_list=edges_list,
            od_matrix=od_matrix,
            constraints=constraints,
            area_km2=spec['area_km2']
        )

        self._save_benchmark(benchmark)
        return benchmark

    def _save_benchmark(self, benchmark: BenchmarkCity):
        """Сохранить бенчмарк в файл"""
        benchmark_file = f"{self.data_dir}/{benchmark.name}.pkl"

        data = {
            'name': benchmark.name,
            'nodes_coords': benchmark.nodes_coords,
            'edges_list': benchmark.edges_list,
            'od_matrix': benchmark.od_matrix,
            'constraints': benchmark.constraints,
            'area_km2': benchmark.area_km2
        }

        with open(benchmark_file, 'wb') as f:
            pickle.dump(data, f)

        logger.info(f"Бенчмарк {benchmark.name} сохранен в {benchmark_file}")

    def load_all_benchmarks(self) -> Dict[str, BenchmarkCity]:
        """Загрузить все доступные бенчмарки"""
        benchmarks = {}

        for benchmark_name in config.evaluation.benchmark_datasets:
            try:
                benchmarks[benchmark_name] = self.load_benchmark(benchmark_name)
            except Exception as e:
                logger.error(f"Ошибка загрузки бенчмарка {benchmark_name}: {e}")

        return benchmarks

class BenchmarkEvaluator:
    """Основной класс для оценки алгоритмов на бенчмарках"""

    def __init__(self,
                 neural_planner: NeuralPlanner = None,
                 experiment_dir: str = None):
        """
        Args:
            neural_planner: обученная нейронная сеть
            experiment_dir: директория для сохранения результатов
        """
        self.neural_planner = neural_planner
        self.experiment_dir = experiment_dir or f"{config.files.results_dir}/evaluation"
        os.makedirs(self.experiment_dir, exist_ok=True)

  # Загрузчик бенчмарков.
        self.loader = BenchmarkLoader()

  # Результаты оценки.
        self.evaluation_results = {}

        logger.info(f"BenchmarkEvaluator инициализирован, результаты в {self.experiment_dir}")

    def evaluate_all_benchmarks(self,
                                alpha_values: List[float] = None,
                                num_seeds: int = None) -> Dict[str, Any]:
        """
        Оценить все алгоритмы на всех бенчмарках

        Args:
            alpha_values: значения α для тестирования
            num_seeds: количество случайных семян

        Returns:
            Словарь с результатами оценки
        """
  # Проверяем, нужно ли пропустить оценку на бенчмарках.
        if config.evaluation.skip_benchmark_evaluation:
            logger.info("Оценка на бенчмарках пропущена согласно конфигурации")
            return {'status': 'skipped', 'reason': 'skip_benchmark_evaluation=True'}

        if alpha_values is None:
            alpha_values = [0.0, 0.5, 1.0]

        if num_seeds is None:
            num_seeds = config.evaluation.num_evaluation_seeds

  # Загружаем все бенчмарки.
        benchmarks = self.loader.load_all_benchmarks()

        if not benchmarks:
            logger.error("Не удалось загрузить бенчмарки")
            return {}

  # Логируем режим работы.
        comparison_mode = "с классическим BCO" if config.evaluation.enable_classical_comparison else "только Neural BCO"
        logger.info(f"Оценка на {len(benchmarks)} бенчмарках с α={alpha_values} ({comparison_mode})")

        all_results = {}

        for benchmark_name, benchmark in benchmarks.items():
            logger.info(f"Оценка на бенчмарке {benchmark_name}")

            try:
                benchmark_results = self.evaluate_single_benchmark(
                    benchmark, alpha_values, num_seeds
                )
                all_results[benchmark_name] = benchmark_results

  # Сохраняем промежуточные результаты.
                self._save_intermediate_results(benchmark_name, benchmark_results)

            except Exception as e:
                logger.error(f"Ошибка оценки бенчмарка {benchmark_name}: {e}")
                all_results[benchmark_name] = {'error': str(e)}

  # Сохраняем общие результаты.
        self.evaluation_results = all_results
        self._save_evaluation_results()

        return all_results

    def evaluate_single_benchmark(self,
                                  benchmark: BenchmarkCity,
                                  alpha_values: List[float],
                                  num_seeds: int) -> Dict[str, Any]:
        """Оценить алгоритмы на одном бенчмарке"""
  # Временно переопределяем ограничения для этого бенчмарка.
        original_constraints = {
            'num_routes': config.mdp.num_routes,
            'min_length': config.mdp.min_route_length,
            'max_length': config.mdp.max_route_length
        }

        config.mdp.num_routes = benchmark.constraints['S']
        config.mdp.min_route_length = benchmark.constraints['MIN']
        config.mdp.max_route_length = benchmark.constraints['MAX']

        try:
            results = {}

            for alpha in alpha_values:
                logger.info(f"  Тестирование с α={alpha}")

                alpha_results = {
                    'alpha': alpha,
                    'seeds': {}
                }

                for seed in range(num_seeds):
                    logger.debug(f"    Семя {seed + 1}/{num_seeds}")

                    np.random.seed(seed)

                    seed_results = self._evaluate_algorithms_single_run(
                        benchmark, alpha, seed
                    )

                    alpha_results['seeds'][seed] = seed_results

  # Агрегируем результаты по семенам.
                alpha_results['aggregated'] = self._aggregate_seed_results(
                    alpha_results['seeds']
                )

                results[f'alpha_{alpha}'] = alpha_results

            return {
                'benchmark_info': {
                    'name': benchmark.name,
                    'nodes': len(benchmark.nodes_coords),
                    'edges': len(benchmark.edges_list),
                    'constraints': benchmark.constraints,
                    'area_km2': benchmark.area_km2
                },
                'results': results
            }

        finally:
  # Восстанавливаем оригинальные ограничения.
            config.mdp.num_routes = original_constraints['num_routes']
            config.mdp.min_route_length = original_constraints['min_length']
            config.mdp.max_route_length = original_constraints['max_length']

    def _evaluate_algorithms_single_run(self,
                                        benchmark: BenchmarkCity,
                                        alpha: float,
                                        seed: int) -> Dict[str, Any]:
        """Один запуск оценки всех алгоритмов"""
        results = {
            'alpha': alpha,
            'seed': seed,
            'algorithms': {}
        }

  # Уменьшаем количество итераций для ускорения оценки.
        original_iterations = config.bco.num_iterations
        config.bco.num_iterations = min(200, original_iterations)

        try:
  # 1. Classical BCO (только если включено в конфигурации).
            if config.evaluation.enable_classical_comparison:
                try:
                    start_time = time.time()

                    cost_calculator = TransitCostCalculator(alpha=alpha)
                    classical_bco = BeeColonyOptimization(
                        city_graph=benchmark.city_graph,
                        od_matrix=benchmark.od_matrix,
                        cost_calculator=cost_calculator,
                        alpha=alpha
                    )

                    classical_result = classical_bco.optimize()

                    results['algorithms']['classical_bco'] = {
                        'cost': classical_result.cost,
                        'routes': classical_result.routes,
                        'feasible': classical_result.is_feasible,
                        'duration': time.time() - start_time,
                        'constraint_violations': classical_result.constraint_violations
                    }

                except Exception as e:
                    logger.error(f"Ошибка Classical BCO: {e}")
                    results['algorithms']['classical_bco'] = {'error': str(e)}
            else:
  # Добавляем заглушку для отключенного классического BCO.
                results['algorithms']['classical_bco'] = {
                    'cost': float('inf'),
                    'routes': [],
                    'feasible': False,
                    'duration': 0.0,
                    'note': 'Classical BCO disabled in config'
                }

  # 2. Neural BCO (если есть обученная модель).
            if self.neural_planner is not None:
                try:
                    start_time = time.time()

                    neural_bco = NeuralBeeColonyOptimization(
                        city_graph=benchmark.city_graph,
                        od_matrix=benchmark.od_matrix,
                        neural_planner=self.neural_planner,
                        alpha=alpha
                    )

                    neural_result = neural_bco.optimize()

  # Детальный анализ Neural BCO.
                    neural_stats = neural_bco._get_neural_bee_statistics()
                    route_analysis = self._analyze_neural_routes(neural_result.routes, benchmark)

                    results['algorithms']['neural_bco'] = {
                        'cost': neural_result.cost,
                        'routes': neural_result.routes,
                        'feasible': neural_result.is_feasible,
                        'duration': time.time() - start_time,
                        'constraint_violations': neural_result.constraint_violations,
                        'neural_statistics': neural_stats,
                        'route_analysis': route_analysis,
                        'detailed_metrics': {
                            'neural_bee_success_rate': neural_stats.get('avg_success_rate', 0),
                            'policy_call_time': neural_stats.get('total_policy_time', 0),
                            'route_quality_score': route_analysis.get('quality_score', 0)
                        }
                    }

                except Exception as e:
                    logger.error(f"Ошибка Neural BCO: {e}")
                    results['algorithms']['neural_bco'] = {'error': str(e)}

  # 3. Neural Planner Only (LP).
            if self.neural_planner is not None:
                try:
                    start_time = time.time()

  # Создаем trainer для rollout.
                    trainer = NeuralPlannerTrainer()
                    trainer.neural_planner = self.neural_planner

  # Выполняем rollout эпизода.
                    states, actions, log_probs, reward = trainer.rollout_episode(
                        benchmark.city_instance, alpha, max_steps=200
                    )

                    if states:
                        final_state = states[-1]
                        routes = final_state.get_all_routes()
                        lp_cost = -reward
                    else:
                        routes = []
                        lp_cost = float('inf')

                    results['algorithms']['neural_planner_only'] = {
                        'cost': lp_cost,
                        'routes': routes,
                        'feasible': len(routes) > 0,
                        'duration': time.time() - start_time,
                        'num_steps': len(states)
                    }

                except Exception as e:
                    logger.error(f"Ошибка Neural Planner: {e}")
                    results['algorithms']['neural_planner_only'] = {'error': str(e)}

        finally:
            config.bco.num_iterations = original_iterations

        return results

    def _aggregate_seed_results(self, seed_results: Dict[int, Dict]) -> Dict[str, Any]:
        """Агрегировать результаты по семенам"""
        algorithms = set()
        for seed_data in seed_results.values():
            algorithms.update(seed_data.get('algorithms', {}).keys())

        aggregated = {}

        for algorithm in algorithms:
            costs = []
            durations = []
            feasible_count = 0
            total_count = 0

  # Специальные метрики для Neural BCO.
            neural_success_rates = []
            route_quality_scores = []

            for seed_data in seed_results.values():
                alg_data = seed_data.get('algorithms', {}).get(algorithm, {})

                if 'error' not in alg_data:
                    costs.append(alg_data.get('cost', float('inf')))
                    durations.append(alg_data.get('duration', 0))

                    if alg_data.get('feasible', False):
                        feasible_count += 1

                    total_count += 1

  # Собираем метрики Neural BCO.
                    if algorithm == 'neural_bco':
                        detailed_metrics = alg_data.get('detailed_metrics', {})
                        neural_success_rates.append(detailed_metrics.get('neural_bee_success_rate', 0))
                        route_quality_scores.append(detailed_metrics.get('route_quality_score', 0))

            if costs:
                result = {
                    'mean_cost': np.mean(costs),
                    'std_cost': np.std(costs),
                    'min_cost': np.min(costs),
                    'max_cost': np.max(costs),
                    'mean_duration': np.mean(durations),
                    'feasible_ratio': feasible_count / total_count if total_count > 0 else 0,
                    'num_runs': total_count
                }

  # Добавляем специальные метрики для Neural BCO.
                if algorithm == 'neural_bco' and neural_success_rates:
                    result['neural_metrics'] = {
                        'avg_neural_success_rate': np.mean(neural_success_rates),
                        'avg_route_quality_score': np.mean(route_quality_scores),
                        'neural_success_rate_std': np.std(neural_success_rates)
                    }

                aggregated[algorithm] = result
            else:
                aggregated[algorithm] = {
                    'mean_cost': float('inf'),
                    'num_runs': 0,
                    'error': 'All runs failed'
                }

        return aggregated

    def _save_intermediate_results(self, benchmark_name: str, results: Dict[str, Any]):
        """Сохранить промежуточные результаты"""
        filepath = f"{self.experiment_dir}/{benchmark_name}_results.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    def _save_evaluation_results(self):
        """Сохранить общие результаты оценки"""
        filepath = f"{self.experiment_dir}/evaluation_results.json"

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.evaluation_results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Результаты оценки сохранены в {filepath}")

    def generate_neural_bco_report(self) -> str:
        """Сгенерировать детальный отчет только по Neural BCO"""
        if not self.evaluation_results:
            return "Нет результатов для генерации отчета Neural BCO"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ДЕТАЛЬНЫЙ ОТЧЕТ NEURAL BCO")
        report_lines.append("=" * 80)
        report_lines.append("")

        for benchmark_name, benchmark_data in self.evaluation_results.items():
            if 'error' in benchmark_data:
                continue

            report_lines.append(f"Бенчмарк: {benchmark_name.upper()}")
            report_lines.append("-" * 40)

            benchmark_info = benchmark_data.get('benchmark_info', {})
            report_lines.append(f"Узлов: {benchmark_info.get('nodes', 'N/A')}")
            report_lines.append(f"Ребер: {benchmark_info.get('edges', 'N/A')}")
            report_lines.append(f"Маршрутов: {benchmark_info.get('constraints', {}).get('S', 'N/A')}")
            report_lines.append("")

  # Результаты Neural BCO по каждому α.
            for alpha_key, alpha_data in benchmark_data.get('results', {}).items():
                alpha = alpha_data.get('alpha', alpha_key)
                report_lines.append(f"α = {alpha}:")

                aggregated = alpha_data.get('aggregated', {})
                neural_data = aggregated.get('neural_bco', {})

                if 'error' not in neural_data:
                    mean_cost = neural_data.get('mean_cost', float('inf'))
                    std_cost = neural_data.get('std_cost', 0)
                    feasible_ratio = neural_data.get('feasible_ratio', 0)

                    report_lines.append(f"  Стоимость: {mean_cost:.3f} ± {std_cost:.3f}")
                    report_lines.append(f"  Допустимость: {feasible_ratio:.1%}")

  # Специальные метрики Neural BCO.
                    neural_metrics = neural_data.get('neural_metrics', {})
                    if neural_metrics:
                        success_rate = neural_metrics.get('avg_neural_success_rate', 0)
                        quality_score = neural_metrics.get('avg_route_quality_score', 0)

                        report_lines.append(f"  Успешность Neural Bees: {success_rate:.1%}")
                        report_lines.append(f"  Качество маршрутов: {quality_score:.3f}")

                report_lines.append("")

            report_lines.append("")

        report_text = "\n".join(report_lines)

  # Сохраняем отчет.
        report_path = f"{self.experiment_dir}/neural_bco_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Отчет Neural BCO сохранен в {report_path}")

        return report_text

    def _analyze_neural_routes(self, routes: List[List[int]], benchmark: BenchmarkCity) -> Dict[str, Any]:
        """Детальный анализ маршрутов, созданных Neural BCO"""
        if not routes:
            return {
                'quality_score': 0.0,
                'coverage_metrics': {},
                'route_metrics': {},
                'efficiency_metrics': {}
            }

  # Анализ покрытия.
        total_nodes = len(benchmark.city_graph.nodes())
        covered_nodes = set()
        for route in routes:
            covered_nodes.update(route)

        coverage_ratio = len(covered_nodes) / total_nodes if total_nodes > 0 else 0.0

  # Анализ качества маршрутов.
        route_lengths = [len(route) for route in routes]
        avg_length = np.mean(route_lengths) if route_lengths else 0

  # Анализ пересечений.
        overlaps = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                overlap = len(set(routes[i]).intersection(set(routes[j])))
                overlaps.append(overlap)

        avg_overlap = np.mean(overlaps) if overlaps else 0

  # Анализ эффективности (простая метрика).
        efficiency_score = coverage_ratio * (1.0 - avg_overlap / max(avg_length, 1))

  # Общий балл качества.
        quality_score = (coverage_ratio * 0.4 +
                         (1.0 - avg_overlap / max(avg_length, 1)) * 0.3 +
                         min(avg_length / 10.0, 1.0) * 0.3)

        return {
            'quality_score': quality_score,
            'coverage_metrics': {
                'coverage_ratio': coverage_ratio,
                'covered_nodes': len(covered_nodes),
                'total_nodes': total_nodes
            },
            'route_metrics': {
                'num_routes': len(routes),
                'avg_length': avg_length,
                'min_length': min(route_lengths) if route_lengths else 0,
                'max_length': max(route_lengths) if route_lengths else 0,
                'length_std': np.std(route_lengths) if route_lengths else 0
            },
            'efficiency_metrics': {
                'avg_overlap': avg_overlap,
                'efficiency_score': efficiency_score,
                'unique_nodes_ratio': len(covered_nodes) / sum(len(route) for route in routes) if routes else 0
            }
        }

    def generate_comparison_report(self) -> str:
        """Сгенерировать отчет сравнения алгоритмов"""
        if not self.evaluation_results:
            return "Нет результатов для генерации отчета"

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ОТЧЕТ СРАВНЕНИЯ АЛГОРИТМОВ")
        report_lines.append("=" * 80)
        report_lines.append("")

        for benchmark_name, benchmark_data in self.evaluation_results.items():
            if 'error' in benchmark_data:
                continue

            report_lines.append(f"Бенчмарк: {benchmark_name.upper()}")
            report_lines.append("-" * 40)

            benchmark_info = benchmark_data.get('benchmark_info', {})
            report_lines.append(f"Узлов: {benchmark_info.get('nodes', 'N/A')}")
            report_lines.append(f"Ребер: {benchmark_info.get('edges', 'N/A')}")
            report_lines.append(f"Маршрутов: {benchmark_info.get('constraints', {}).get('S', 'N/A')}")
            report_lines.append("")

  # Результаты по каждому α.
            for alpha_key, alpha_data in benchmark_data.get('results', {}).items():
                alpha = alpha_data.get('alpha', alpha_key)
                report_lines.append(f"α = {alpha}:")

                aggregated = alpha_data.get('aggregated', {})

                for algorithm, stats in aggregated.items():
                    if 'error' not in stats:
                        mean_cost = stats.get('mean_cost', float('inf'))
                        std_cost = stats.get('std_cost', 0)
                        feasible_ratio = stats.get('feasible_ratio', 0)

                        report_lines.append(f"  {algorithm}:")
                        report_lines.append(f"    Стоимость: {mean_cost:.3f} ± {std_cost:.3f}")
                        report_lines.append(f"    Допустимость: {feasible_ratio:.1%}")

  # Вычисляем улучшения.
                if ('neural_bco' in aggregated and 'classical_bco' in aggregated and
                        'error' not in aggregated['neural_bco'] and 'error' not in aggregated['classical_bco']):

                    neural_cost = aggregated['neural_bco']['mean_cost']
                    classical_cost = aggregated['classical_bco']['mean_cost']

                    if classical_cost > 0 and neural_cost < float('inf'):
                        improvement = (classical_cost - neural_cost) / classical_cost * 100
                        report_lines.append(f"  Улучшение Neural BCO vs Classical: {improvement:.1f}%")

                report_lines.append("")

            report_lines.append("")

        report_text = "\n".join(report_lines)

  # Сохраняем отчет.
        report_path = f"{self.experiment_dir}/comparison_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_text)

        logger.info(f"Отчет сравнения сохранен в {report_path}")

        return report_text

    def plot_algorithm_comparison(self):
        """Создать графики сравнения алгоритмов"""
        if not self.evaluation_results:
            logger.warning("Нет результатов для визуализации")
            return

  # Собираем данные для графиков.
        plot_data = []

        for benchmark_name, benchmark_data in self.evaluation_results.items():
            if 'error' in benchmark_data:
                continue

            for alpha_key, alpha_data in benchmark_data.get('results', {}).items():
                alpha = alpha_data.get('alpha', alpha_key)
                aggregated = alpha_data.get('aggregated', {})

                for algorithm, stats in aggregated.items():
                    if 'error' not in stats:
                        plot_data.append({
                            'benchmark': benchmark_name,
                            'alpha': alpha,
                            'algorithm': algorithm,
                            'mean_cost': stats.get('mean_cost', float('inf')),
                            'std_cost': stats.get('std_cost', 0),
                            'feasible_ratio': stats.get('feasible_ratio', 0)
                        })

        if not plot_data:
            logger.warning("Нет данных для построения графиков")
            return

        df = pd.DataFrame(plot_data)

  # Фильтруем бесконечные значения.
        df = df[df['mean_cost'] < float('inf')]

        if len(df) == 0:
            logger.warning("Все стоимости бесконечны, графики не могут быть построены")
            return

  # График 1: Сравнение стоимости по бенчмаркам.
        plt.figure(figsize=(15, 10))

  # Subplot 1: Средняя стоимость.
        plt.subplot(2, 2, 1)

        benchmarks = df['benchmark'].unique()
        algorithms = df['algorithm'].unique()
        alpha_values = sorted(df['alpha'].unique())

  # Выбираем α=0.5 для основного сравнения.
        df_main = df[df['alpha'] == 0.5] if 0.5 in alpha_values else df[df['alpha'] == alpha_values[0]]

        if len(df_main) > 0:
            pivot_data = df_main.pivot(index='benchmark', columns='algorithm', values='mean_cost')
            pivot_data.plot(kind='bar', ax=plt.gca())
            plt.title('Средняя стоимость по бенчмаркам (α=0.5)')
            plt.ylabel('Стоимость')
            plt.xticks(rotation=45)
            plt.legend(title='Алгоритм')

  # Subplot 2: Trade-off passenger vs operator cost.
        plt.subplot(2, 2, 2)

        for algorithm in algorithms:
            alg_data = df[df['algorithm'] == algorithm]
            if len(alg_data) > 0:
                plt.plot(alg_data['alpha'], alg_data['mean_cost'],
                         marker='o', label=algorithm, linewidth=2)

        plt.xlabel('α (passenger weight)')
        plt.ylabel('Стоимость')
        plt.title('Trade-off: пассажирская vs операторская стоимость')
        plt.legend()
        plt.grid(True, alpha=0.3)

  # Subplot 3: Feasibility ratio.
        plt.subplot(2, 2, 3)

        pivot_feasible = df_main.pivot(index='benchmark', columns='algorithm', values='feasible_ratio')
        pivot_feasible.plot(kind='bar', ax=plt.gca())
        plt.title('Доля допустимых решений')
        plt.ylabel('Доля допустимых решений')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)

  # Subplot 4: Improvement over Classical BCO.
        plt.subplot(2, 2, 4)

        improvements = []
        improvement_labels = []

        for benchmark in benchmarks:
            bench_data = df_main[df_main['benchmark'] == benchmark]

            classical_data = bench_data[bench_data['algorithm'] == 'classical_bco']
            neural_data = bench_data[bench_data['algorithm'] == 'neural_bco']

            if len(classical_data) > 0 and len(neural_data) > 0:
                classical_cost = classical_data['mean_cost'].iloc[0]
                neural_cost = neural_data['mean_cost'].iloc[0]

                if classical_cost > 0:
                    improvement = (classical_cost - neural_cost) / classical_cost * 100
                    improvements.append(improvement)
                    improvement_labels.append(benchmark)

        if improvements:
            plt.bar(improvement_labels, improvements, color='green', alpha=0.7)
            plt.title('Улучшение Neural BCO vs Classical BCO (%)')
            plt.ylabel('Улучшение (%)')
            plt.xticks(rotation=45)
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.experiment_dir}/algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Графики сравнения сохранены в {self.experiment_dir}/algorithm_comparison.png")

def run_benchmark_evaluation(neural_planner: NeuralPlanner = None,
                             benchmarks: List[str] = None,
                             alpha_values: List[float] = None,
                             num_seeds: int = 3) -> Dict[str, Any]:
    """
    Запустить полную оценку на бенчмарках

    Args:
        neural_planner: обученная нейронная сеть
        benchmarks: список бенчмарков для оценки
        alpha_values: значения α для тестирования
        num_seeds: количество случайных семян

    Returns:
        Результаты оценки
    """
    if benchmarks is None:
        benchmarks = config.evaluation.benchmark_datasets

    if alpha_values is None:
        alpha_values = [0.0, 0.5, 1.0]

  # Создаем evaluator.
    evaluator = BenchmarkEvaluator(neural_planner=neural_planner)

  # Ограничиваем оценку выбранными бенчмарками.
    original_benchmarks = config.evaluation.benchmark_datasets
    config.evaluation.benchmark_datasets = benchmarks

    try:
  # Запускаем оценку.
        results = evaluator.evaluate_all_benchmarks(alpha_values, num_seeds)

  # Генерируем отчет.
        report = evaluator.generate_comparison_report()
        logger.info(report)

  # Создаем графики.
        evaluator.plot_algorithm_comparison()

        return results

    finally:
  # Восстанавливаем оригинальный список.
        config.evaluation.benchmark_datasets = original_benchmarks

if __name__ == "__main__":
  # Демонстрация системы оценки.
    logger.info("Демонстрация системы оценки бенчмарков...")

  # Загружаем бенчмарки.
    loader = BenchmarkLoader()

  # Тестируем загрузку одного бенчмарка.
    try:
        mandl = loader.load_benchmark('mandl')
        logger.info(f"Бенчмарк Mandl загружен: {len(mandl.nodes_coords)} узлов, {len(mandl.edges_list)} ребер")
        logger.info(f"Ограничения: {mandl.constraints}")

  # Тестируем создание CityInstance.
        city_instance = mandl.city_instance
        logger.info(f"CityInstance создан: {len(city_instance.city_graph.nodes())} узлов в графе")

    except Exception as e:
        logger.info(f"Ошибка загрузки Mandl: {e}")

  # Тестируем загрузку всех бенчмарков.
    logger.info("\nЗагрузка всех бенчмарков...")
    benchmarks = loader.load_all_benchmarks()

    for name, benchmark in benchmarks.items():
        logger.info(f"  {name}: {benchmark.constraints}")

  # Быстрая оценка (без нейронной сети).
    logger.info("\nБыстрая оценка Classical BCO...")

    try:
        evaluator = BenchmarkEvaluator()

  # Уменьшаем параметры для быстрого тестирования.
        config.bco.num_iterations = 50
        config.evaluation.num_evaluation_seeds = 2

  # Тестируем только на Mandl.
        mandl = benchmarks.get('mandl')
        if mandl:
            result = evaluator.evaluate_single_benchmark(
                mandl, alpha_values=[0.5], num_seeds=2
            )

            logger.info("Результат оценки Mandl:")
            for alpha_key, alpha_data in result['results'].items():
                aggregated = alpha_data.get('aggregated', {})
                for algorithm, stats in aggregated.items():
                    if 'error' not in stats:
                        logger.info(f"  {algorithm}: {stats['mean_cost']:.3f}")

  # Генерируем отчет.
        evaluator.evaluation_results = {'mandl': result}
        report = evaluator.generate_comparison_report()
        logger.info("\nОтчет:")
        logger.info(report[:500] + "..." if len(report) > 500 else report)

    except Exception as e:
        logger.info(f"Ошибка оценки: {e}")

    logger.info("Демонстрация системы оценки завершена!")
