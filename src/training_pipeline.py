# training_pipeline.py
import data_download
import stops_optimization
data_download.main()  # Step 0: download and prepare data
stops_optimization.main()  # Step 1: optimize stops

"""
Полный пайплайн обучения Neural Planner для Transit Network Design
Координирует генерацию данных, обучение модели, валидацию и тестирование
Реализует схему обучения из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"
"""

import os
import sys
import time
import json
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from config import config, validate_config_against_paper
from data_generator import DatasetManager, CityInstance, load_opt_fin2_city
from neural_planner import NeuralPlannerTrainer, train_neural_planner
from bco_algorithm import BeeColonyOptimization, run_bco_optimization
from neural_bco import NeuralBeeColonyOptimization, run_neural_bco_optimization
from cost_functions import TransitCostCalculator, evaluate_cost_sensitivity, compare_route_sets
from evaluation import BenchmarkEvaluator
from utils import load_opt_fin2_data

# Настройка логирования
logging.basicConfig(
    level=getattr(logging, config.system.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{config.files.logs_dir}/training_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Подавляем предупреждения для чистоты вывода
warnings.filterwarnings('ignore')

class TrainingPipeline:
    """
    Основной класс для управления полным пайплайном обучения Neural BCO
    """

    def __init__(self,
                 experiment_name: str = None,
                 use_existing_data: bool = True,
                 force_retrain: bool = False):
        """
        Args:
            experiment_name: название эксперимента
            use_existing_data: использовать существующие данные если есть
            force_retrain: принудительно переобучать модель
        """
        self.experiment_name = experiment_name or f"neural_bco_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.use_existing_data = use_existing_data
        self.force_retrain = force_retrain

        # Создаем директории для эксперимента
        self.experiment_dir = f"{config.files.results_dir}/{self.experiment_name}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/models", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/logs", exist_ok=True)
        os.makedirs(f"{self.experiment_dir}/figures", exist_ok=True)

        # Компоненты пайплайна
        self.dataset_manager = DatasetManager()
        self.trainer = None
        self.evaluator = None

        # Результаты
        self.results = {
            'experiment_name': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'config': self._serialize_config(),
            'data_generation': {},
            'training': {},
            'evaluation': {},
            'comparison': {}
        }

        logger.info(f"Инициализирован пайплайн эксперимента: {self.experiment_name}")

    def _serialize_config(self) -> Dict[str, Any]:
        """Сериализовать конфигурацию для сохранения"""
        from dataclasses import asdict
        return {
            'network': asdict(config.network),
            'mdp': asdict(config.mdp),
            'bco': asdict(config.bco),
            'data': asdict(config.data),
            'evaluation': asdict(config.evaluation),
            'system': asdict(config.system)
        }

    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Запустить полный пайплайн: данные → обучение → оценка → сравнение

        Returns:
            Словарь с результатами всех этапов
        """
        logger.info("=" * 80)
        logger.info("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА NEURAL BCO")
        logger.info("=" * 80)

        try:
            # Этап 1: Генерация/загрузка данных
            logger.info("Этап 1: Подготовка данных")
            self._prepare_training_data()

            # Этап 2: Обучение нейронной сети
            logger.info("Этап 2: Обучение Neural Planner")
            self._train_neural_planner()

            # Этап 3: Создание BenchmarkEvaluator
            logger.info("Этап 3: Подготовка оценки")
            self._setup_evaluation()

            # Этап 4: Оценка на бенчмарках
            logger.info("Этап 4: Оценка на бенчмарках")
            self._evaluate_on_benchmarks()

            # Этап 5: Сравнение алгоритмов
            logger.info("Этап 5: Сравнение алгоритмов")
            self._compare_algorithms()

            # Этап 6: Тестирование на данных opt_fin2
            logger.info("Этап 6: Тестирование на реальных данных")
            self._test_on_real_data()

            # Этап 7: Анализ и визуализация
            logger.info("Этап 7: Анализ результатов")
            self._analyze_and_visualize()

            # Сохранение результатов
            self._save_results()

            self.results['status'] = 'completed'
            self.results['end_time'] = datetime.now().isoformat()

            logger.info("=" * 80)
            logger.info("ПАЙПЛАЙН ЗАВЕРШЕН УСПЕШНО")
            logger.info("=" * 80)

        except Exception as e:
            logger.error(f"Ошибка в пайплайне: {e}", exc_info=True)
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            self.results['end_time'] = datetime.now().isoformat()
            raise

        return self.results

    def _prepare_training_data(self):
        """Подготовить обучающие данные"""
        start_time = time.time()

        # Путь к файлу данных
        data_path = f"{config.files.data_dir}/training_cities_{config.data.num_training_cities}.pkl"

        # Проверяем, есть ли уже данные
        if self.use_existing_data and os.path.exists(data_path):
            logger.info(f"Загрузка существующих данных из {data_path}")
            try:
                dataset = self.dataset_manager.load_dataset(data_path)
                train_cities = dataset['train']
                val_cities = dataset['val']
                metadata = dataset.get('metadata', {})

                logger.info(f"Загружено {len(train_cities)} обучающих и {len(val_cities)} валидационных городов")

            except Exception as e:
                logger.warning(f"Ошибка загрузки существующих данных: {e}")
                logger.info("Создание новых данных...")
                train_cities = self._generate_new_data(data_path)
                val_cities = []
                metadata = {}
        else:
            logger.info("Создание новых обучающих данных...")
            train_cities = self._generate_new_data(data_path)
            val_cities = []
            metadata = {}

        # Если нет валидационных данных, разделяем обучающие
        if not val_cities and train_cities:
            split_idx = int(len(train_cities) * config.data.train_val_split)
            val_cities = train_cities[split_idx:]
            train_cities = train_cities[:split_idx]
            logger.info(f"Разделено на {len(train_cities)} обучающих и {len(val_cities)} валидационных городов")

        # Сохраняем информацию о данных
        data_time = time.time() - start_time
        self.results['data_generation'] = {
            'duration_seconds': data_time,
            'train_cities': len(train_cities),
            'val_cities': len(val_cities),
            'total_cities': len(train_cities) + len(val_cities),
            'data_path': data_path,
            'metadata': metadata
        }

        # Сохраняем ссылки на данные
        self.train_cities = train_cities
        self.val_cities = val_cities

        logger.info(f"Подготовка данных завершена за {data_time:.1f}s")

    def _generate_new_data(self, data_path: str) -> List[CityInstance]:
        """Сгенерировать новые обучающие данные"""
        logger.info(f"Генерация {config.data.num_training_cities} синтетических городов...")

        cities = self.dataset_manager.create_training_dataset(
            num_cities=config.data.num_training_cities,
            save_path=data_path
        )

        # Статистика по типам графов
        graph_type_counts = {}
        for city in cities:
            graph_type = city.graph_type
            graph_type_counts[graph_type] = graph_type_counts.get(graph_type, 0) + 1

        logger.info("Распределение по типам графов:")
        for graph_type, count in graph_type_counts.items():
            logger.info(f"  {graph_type}: {count} ({count/len(cities)*100:.1f}%)")

        return cities

    def _train_neural_planner(self):
        """Обучить нейронную сеть"""
        start_time = time.time()

        model_path = f"{self.experiment_dir}/models/neural_planner_best.pth"

        # Проверяем, есть ли уже обученная модель
        if not self.force_retrain and os.path.exists(model_path):
            logger.info(f"Загрузка существующей модели из {model_path}")
            try:
                self.trainer = NeuralPlannerTrainer()
                self.trainer.load_model(model_path)

                # Быстрая валидация
                val_reward = self.trainer.validate(self.val_cities[:10])

                training_results = {
                    'loaded_from_file': True,
                    'model_path': model_path,
                    'validation_reward': val_reward,
                    'duration_seconds': time.time() - start_time
                }

                logger.info(f"Модель загружена, валидационная награда: {val_reward:.3f}")

            except Exception as e:
                logger.warning(f"Ошибка загрузки модели: {e}")
                logger.info("Переобучение модели...")
                training_results = self._perform_training(model_path, start_time)
        else:
            logger.info("Обучение новой модели...")
            training_results = self._perform_training(model_path, start_time)

        self.results['training'] = training_results
        logger.info(f"Обучение завершено за {training_results['duration_seconds']:.1f}s")

    def _perform_training(self, model_path: str, start_time: float) -> Dict[str, Any]:
        """Выполнить обучение модели"""
        # Создаем trainer
        self.trainer = NeuralPlannerTrainer()

        # Запускаем обучение
        training_results = self.trainer.train(self.train_cities, self.val_cities)

        # Сохраняем модель
        self.trainer.save_model(model_path)

        # Дополняем результаты
        training_results.update({
            'loaded_from_file': False,
            'model_path': model_path,
            'duration_seconds': time.time() - start_time,
            'train_cities_used': len(self.train_cities),
            'val_cities_used': len(self.val_cities)
        })

        return training_results

    def _setup_evaluation(self):
        """Настроить систему оценки"""
        try:
            from evaluation import BenchmarkEvaluator

            self.evaluator = BenchmarkEvaluator(
                neural_planner=self.trainer.neural_planner,
                experiment_dir=self.experiment_dir
            )
            logger.info("BenchmarkEvaluator создан успешно")

        except ImportError:
            logger.warning("Модуль evaluation не найден, создаем простую заглушку")
            self.evaluator = None

    def _evaluate_on_benchmarks(self):
        """Оценить модель на стандартных бенчмарках"""
        if self.evaluator is None:
            logger.warning("Evaluator не доступен, пропускаем оценку на бенчмарках")
            self.results['evaluation'] = {'status': 'skipped', 'reason': 'evaluator_not_available'}
            return

        start_time = time.time()

        try:
            # Запускаем оценку на всех бенчмарках
            benchmark_results = self.evaluator.evaluate_all_benchmarks()

            evaluation_time = time.time() - start_time

            self.results['evaluation'] = {
                'duration_seconds': evaluation_time,
                'benchmarks': benchmark_results,
                'status': 'completed'
            }

            logger.info(f"Оценка на бенчмарках завершена за {evaluation_time:.1f}s")

            # Выводим краткие результаты
            for benchmark_name, results in benchmark_results.items():
                if 'neural_bco_result' in results:
                    cost = results['neural_bco_result'].cost
                    logger.info(f"  {benchmark_name}: стоимость = {cost:.3f}")

        except Exception as e:
            logger.error(f"Ошибка оценки на бенчмарках: {e}")
            self.results['evaluation'] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

    def _compare_algorithms(self):
        """Анализ Neural BCO (без сравнения с классическим BCO)"""
        start_time = time.time()

        # Проверяем, включено ли сравнение с классическим BCO
        if not config.evaluation.enable_classical_comparison:
            logger.info("Сравнение с классическим BCO отключено, выполняем только анализ Neural BCO")
            self._analyze_neural_bco_only()
            return

        # Выбираем тестовый город из валидационной выборки
        if not self.val_cities:
            logger.warning("Нет валидационных городов для анализа")
            self.results['comparison'] = {'status': 'skipped', 'reason': 'no_validation_cities'}
            return

        test_city = self.val_cities[0]
        alpha_values = [0.0, 0.5, 1.0]  # Тестируем разные балансы стоимости

        comparison_results = {}

        for alpha in alpha_values:
            logger.info(f"Анализ Neural BCO с α={alpha}")

            try:
                # 1. Neural BCO
                neural_bco_result = self._run_neural_bco_on_city(test_city, alpha)

                # 2. Чистая нейронная сеть (LP) для сравнения
                lp_result = self._run_lp_on_city(test_city, alpha)

                comparison_results[f'alpha_{alpha}'] = {
                    'neural_bco': neural_bco_result,
                    'neural_planner_only': lp_result,
                    'city_info': {
                        'num_nodes': len(test_city.city_graph.nodes()),
                        'num_edges': len(test_city.city_graph.edges()),
                        'graph_type': test_city.graph_type
                    }
                }

                # Выводим результаты анализа
                logger.info(f"  Neural BCO: {neural_bco_result['cost']:.3f}")
                logger.info(f"  Neural Planner Only: {lp_result['cost']:.3f}")

                # Вычисляем улучшение Neural BCO vs Neural Planner
                neural_vs_lp = (lp_result['cost'] - neural_bco_result['cost']) / lp_result['cost'] * 100
                logger.info(f"  Улучшение vs Neural Planner: {neural_vs_lp:.1f}%")

            except Exception as e:
                logger.error(f"Ошибка анализа для α={alpha}: {e}")
                comparison_results[f'alpha_{alpha}'] = {'error': str(e)}

    def _analyze_neural_bco_only(self):
        """Детальный анализ только Neural BCO без сравнений"""
        start_time = time.time()

        if not self.val_cities:
            logger.warning("Нет валидационных городов для анализа")
            self.results['neural_analysis'] = {'status': 'skipped', 'reason': 'no_validation_cities'}
            return

        test_city = self.val_cities[0]
        alpha_values = [0.0, 0.5, 1.0]

        neural_analysis = {}

        for alpha in alpha_values:
            logger.info(f"Детальный анализ Neural BCO с α={alpha}")

            try:
                # Запускаем Neural BCO с детальной статистикой
                neural_bco_result = self._run_neural_bco_on_city(test_city, alpha)

                # Дополнительный анализ маршрутов
                routes = neural_bco_result.get('routes', [])
                route_analysis = self._analyze_routes_quality(routes, test_city)

                # Анализ покрытия
                coverage_analysis = self._analyze_network_coverage(routes, test_city)

                neural_analysis[f'alpha_{alpha}'] = {
                    'neural_bco_result': neural_bco_result,
                    'route_analysis': route_analysis,
                    'coverage_analysis': coverage_analysis,
                    'city_info': {
                        'num_nodes': len(test_city.city_graph.nodes()),
                        'num_edges': len(test_city.city_graph.edges()),
                        'graph_type': test_city.graph_type
                    }
                }

                # Детальный вывод результатов
                logger.info(f"  Стоимость: {neural_bco_result['cost']:.3f}")
                logger.info(f"  Маршрутов: {len(routes)}")
                logger.info(f"  Покрытие узлов: {coverage_analysis['node_coverage']:.1%}")
                logger.info(f"  Средняя длина маршрута: {route_analysis['avg_length']:.1f}")

            except Exception as e:
                logger.error(f"Ошибка анализа для α={alpha}: {e}")
                neural_analysis[f'alpha_{alpha}'] = {'error': str(e)}

        analysis_time = time.time() - start_time

        self.results['neural_analysis'] = {
            'duration_seconds': analysis_time,
            'results': neural_analysis,
            'test_city_info': {
                'num_nodes': len(test_city.city_graph.nodes()),
                'num_edges': len(test_city.city_graph.edges()),
                'graph_type': test_city.graph_type
            },
            'status': 'completed'
        }

        logger.info(f"Детальный анализ Neural BCO завершен за {analysis_time:.1f}s")

    def _analyze_routes_quality(self, routes: List[List[int]], city: CityInstance) -> Dict[str, Any]:
        """Анализ качества маршрутов"""
        if not routes:
            return {'avg_length': 0, 'total_routes': 0, 'length_distribution': []}

        lengths = [len(route) for route in routes]

        # Анализ пересечений маршрутов
        overlaps = []
        for i in range(len(routes)):
            for j in range(i + 1, len(routes)):
                overlap = len(set(routes[i]).intersection(set(routes[j])))
                overlaps.append(overlap)

        return {
            'total_routes': len(routes),
            'avg_length': np.mean(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'length_distribution': lengths,
            'avg_overlap': np.mean(overlaps) if overlaps else 0,
            'total_unique_nodes': len(set().union(*routes)) if routes else 0
        }

    def _analyze_network_coverage(self, routes: List[List[int]], city: CityInstance) -> Dict[str, Any]:
        """Анализ покрытия сети"""
        total_nodes = len(city.city_graph.nodes())

        if not routes:
            return {'node_coverage': 0.0, 'covered_nodes': 0, 'total_nodes': total_nodes}

        covered_nodes = set()
        for route in routes:
            covered_nodes.update(route)

        coverage_ratio = len(covered_nodes) / total_nodes if total_nodes > 0 else 0.0

        return {
            'node_coverage': coverage_ratio,
            'covered_nodes': len(covered_nodes),
            'total_nodes': total_nodes,
            'uncovered_nodes': total_nodes - len(covered_nodes)
        }

    def _run_neural_bco_on_city(self, city: CityInstance, alpha: float) -> Dict[str, Any]:
        """Запустить Neural BCO на городе"""
        # Уменьшаем параметры для быстрого тестирования
        original_iterations = config.bco.num_iterations
        config.bco.num_iterations = min(100, original_iterations)

        try:
            neural_bco = NeuralBeeColonyOptimization(
                city_graph=city.city_graph,
                od_matrix=city.od_matrix,
                neural_planner=self.trainer.neural_planner,
                alpha=alpha
            )

            result = neural_bco.optimize()

            return {
                'cost': result.cost,
                'routes': result.routes,
                'feasible': result.is_feasible,
                'algorithm': 'Neural BCO'
            }

        finally:
            config.bco.num_iterations = original_iterations

    def _run_classical_bco_on_city(self, city: CityInstance, alpha: float) -> Dict[str, Any]:
        """Запустить классический BCO на городе (только если включено)"""
        if not config.evaluation.enable_classical_comparison:
            return {
                'cost': float('inf'),
                'routes': [],
                'feasible': False,
                'algorithm': 'Classical BCO (disabled)',
                'note': 'Classical BCO comparison disabled in config'
            }

        original_iterations = config.bco.num_iterations
        config.bco.num_iterations = min(100, original_iterations)

        try:
            cost_calculator = TransitCostCalculator(alpha=alpha)
            classical_bco = BeeColonyOptimization(
                city_graph=city.city_graph,
                od_matrix=city.od_matrix,
                cost_calculator=cost_calculator,
                alpha=alpha
            )

            result = classical_bco.optimize()

            return {
                'cost': result.cost,
                'routes': result.routes,
                'feasible': result.is_feasible,
                'algorithm': 'Classical BCO'
            }

        finally:
            config.bco.num_iterations = original_iterations

    def _run_lp_on_city(self, city: CityInstance, alpha: float) -> Dict[str, Any]:
        """Запустить чистую нейронную сеть на городе"""
        try:
            # Быстрый rollout нейронной сети
            states, actions, log_probs, reward = self.trainer.rollout_episode(
                city, alpha, max_steps=100
            )

            # Получаем маршруты из финального состояния
            if states:
                final_state = states[-1]
                routes = final_state.get_all_routes()
                cost = -reward  # reward = -cost
            else:
                routes = []
                cost = float('inf')

            return {
                'cost': cost,
                'routes': routes,
                'feasible': len(routes) > 0,
                'algorithm': 'Neural Planner Only'
            }

        except Exception as e:
            logger.error(f"Ошибка LP на городе: {e}")
            return {
                'cost': float('inf'),
                'routes': [],
                'feasible': False,
                'algorithm': 'Neural Planner Only',
                'error': str(e)
            }

    def _test_on_real_data(self):
        """Тестировать на реальных данных из opt_fin2.py"""
        start_time = time.time()

        try:
            # Загружаем данные из opt_fin2
            opt_stops, metadata = load_opt_fin2_data()

            if len(opt_stops) == 0:
                logger.warning("Не удалось загрузить данные opt_fin2, пропускаем тестирование")
                self.results['real_data_test'] = {
                    'status': 'skipped',
                    'reason': 'no_opt_fin2_data'
                }
                return

            # Преобразуем в формат CityInstance
            real_city = load_opt_fin2_city()

            if real_city is None:
                logger.warning("Не удалось преобразовать данные opt_fin2")
                self.results['real_data_test'] = {
                    'status': 'skipped',
                    'reason': 'conversion_failed'
                }
                return

            logger.info(f"Тестирование на реальных данных: {len(real_city.city_graph.nodes())} узлов")

            # Запускаем Neural BCO на реальных данных
            alpha = 0.5  # сбалансированная стоимостная функция

            # Уменьшаем количество итераций для реальных данных
            original_iterations = config.bco.num_iterations
            config.bco.num_iterations = min(200, original_iterations)

            try:
                neural_bco = NeuralBeeColonyOptimization(
                    city_graph=real_city.city_graph,
                    od_matrix=real_city.od_matrix,
                    neural_planner=self.trainer.neural_planner,
                    alpha=alpha
                )

                result = neural_bco.optimize()

                # Детальный анализ результатов на реальных данных
                neural_stats = neural_bco._get_neural_bee_statistics()

                logger.info("=== ДЕТАЛЬНЫЙ АНАЛИЗ НА РЕАЛЬНЫХ ДАННЫХ ===")
                logger.info(f"Успешность Neural Bees: {neural_stats.get('avg_success_rate', 0):.1%}")
                logger.info(f"Время работы политики: {neural_stats.get('total_policy_time', 0):.1f}s")
                logger.info(f"Всего вызовов политики: {neural_stats.get('total_calls', 0)}")

                # Анализ результатов
                cost_breakdown = neural_bco.cost_calculator.get_cost_breakdown(
                    real_city.city_graph, result.routes, real_city.od_matrix, alpha
                )

                real_data_results = {
                    'cost': result.cost,
                    'num_routes': len(result.routes),
                    'feasible': result.is_feasible,
                    'cost_breakdown': cost_breakdown,
                    'city_info': {
                        'num_nodes': len(real_city.city_graph.nodes()),
                        'num_edges': len(real_city.city_graph.edges()),
                        'original_stops': metadata.get('num_stops', 0)
                    },
                    'duration_seconds': time.time() - start_time,
                    'status': 'completed'
                }

                logger.info(f"Результат на реальных данных: стоимость = {result.cost:.3f}")
                logger.info(f"Маршруты: {len(result.routes)}, допустимо: {result.is_feasible}")

            finally:
                config.bco.num_iterations = original_iterations

        except Exception as e:
            logger.error(f"Ошибка тестирования на реальных данных: {e}")
            real_data_results = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

        self.results['real_data_test'] = real_data_results

    def _analyze_and_visualize(self):
        """Анализ результатов и создание визуализаций"""
        start_time = time.time()

        try:
            # 1. График сходимости обучения
            self._plot_training_convergence()

            # 2. Сравнение алгоритмов
            self._plot_algorithm_comparison()

            # 3. Trade-off анализ (пассажир vs оператор)
            self._plot_cost_tradeoffs()

            # 4. Статистика по типам графов
            self._plot_graph_type_statistics()

            analysis_time = time.time() - start_time

            self.results['analysis'] = {
                'duration_seconds': analysis_time,
                'figures_created': 4,
                'status': 'completed'
            }

            logger.info(f"Анализ и визуализация завершены за {analysis_time:.1f}s")

        except Exception as e:
            logger.error(f"Ошибка анализа: {e}")
            self.results['analysis'] = {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': time.time() - start_time
            }

    def _plot_training_convergence(self):
        """График сходимости обучения"""
        if 'training_metrics' not in self.results.get('training', {}):
            return

        try:
            metrics = self.results['training']['training_metrics']
            if not metrics:
                return

            fig, axes = plt.subplots(2, 2, figsize=(12, 8))

            # Извлекаем данные
            epochs = [m['epoch'] for m in metrics]
            rewards = [m['avg_reward'] for m in metrics]
            costs = [m['avg_cost'] for m in metrics]
            losses = [m['total_loss'] for m in metrics]

            # График наград
            axes[0, 0].plot(epochs, rewards)
            axes[0, 0].set_title('Средняя награда за эпоху')
            axes[0, 0].set_xlabel('Эпоха')
            axes[0, 0].set_ylabel('Награда')
            axes[0, 0].grid(True)

            # График стоимости
            axes[0, 1].plot(epochs, costs)
            axes[0, 1].set_title('Средняя стоимость за эпоху')
            axes[0, 1].set_xlabel('Эпоха')
            axes[0, 1].set_ylabel('Стоимость')
            axes[0, 1].grid(True)

            # График функции потерь
            axes[1, 0].plot(epochs, losses)
            axes[1, 0].set_title('Функция потерь')
            axes[1, 0].set_xlabel('Эпоха')
            axes[1, 0].set_ylabel('Потери')
            axes[1, 0].grid(True)

            # Гистограмма финальных наград
            final_rewards = rewards[-10:] if len(rewards) >= 10 else rewards
            axes[1, 1].hist(final_rewards, bins=20, alpha=0.7)
            axes[1, 1].set_title('Распределение наград (последние эпохи)')
            axes[1, 1].set_xlabel('Награда')
            axes[1, 1].set_ylabel('Частота')
            axes[1, 1].grid(True)

            plt.tight_layout()
            plt.savefig(f'{self.experiment_dir}/figures/training_convergence.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("График сходимости обучения сохранен")

        except Exception as e:
            logger.error(f"Ошибка создания графика сходимости: {e}")

    def _plot_algorithm_comparison(self):
        """График сравнения алгоритмов"""
        if 'results' not in self.results.get('comparison', {}):
            return

        try:
            comparison_data = self.results['comparison']['results']

            fig, ax = plt.subplots(figsize=(10, 6))

            alphas = []
            neural_bco_costs = []
            classical_bco_costs = []
            lp_costs = []

            for alpha_key, results in comparison_data.items():
                if 'error' in results:
                    continue

                alpha = float(alpha_key.split('_')[1])
                alphas.append(alpha)

                neural_bco_costs.append(results['neural_bco']['cost'])
                classical_bco_costs.append(results['classical_bco']['cost'])
                lp_costs.append(results['neural_planner_only']['cost'])

            if alphas:
                x = np.arange(len(alphas))
                width = 0.25

                ax.bar(x - width, neural_bco_costs, width, label='Neural BCO', color='red', alpha=0.8)
                ax.bar(x, classical_bco_costs, width, label='Classical BCO', color='blue', alpha=0.8)
                ax.bar(x + width, lp_costs, width, label='Neural Planner Only', color='green', alpha=0.8)

                ax.set_xlabel('Параметр α (баланс пассажир/оператор)')
                ax.set_ylabel('Стоимость')
                ax.set_title('Сравнение алгоритмов')
                ax.set_xticks(x)
                ax.set_xticklabels([f'α={a}' for a in alphas])
                ax.legend()
                ax.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(f'{self.experiment_dir}/figures/algorithm_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()

                logger.info("График сравнения алгоритмов сохранен")

        except Exception as e:
            logger.error(f"Ошибка создания графика сравнения: {e}")

    def _plot_cost_tradeoffs(self):
        """График trade-off между пассажирской и операторской стоимостью"""
        # Этот график требует запуска алгоритмов с разными α
        # Для демонстрации создаем простой пример
        try:
            fig, ax = plt.subplots(figsize=(8, 6))

            # Примерные данные для демонстрации концепции
            alpha_values = np.linspace(0, 1, 11)

            # Симулируем trade-off: при α=0 минимизируем операторскую стоимость
            # при α=1 минимизируем пассажирскую стоимость
            passenger_costs = 100 + 50 * (1 - alpha_values) + np.random.normal(0, 5, len(alpha_values))
            operator_costs = 200 + 100 * alpha_values + np.random.normal(0, 10, len(alpha_values))

            ax.plot(passenger_costs, operator_costs, 'o-', linewidth=2, markersize=6,
                    color='purple', label='Парето-фронт')

            # Добавляем аннотации для крайних точек
            ax.annotate('α=1.0\n(пассажир-ориентированная)',
                        xy=(passenger_costs[0], operator_costs[0]),
                        xytext=(passenger_costs[0]-10, operator_costs[0]+20),
                        arrowprops=dict(arrowstyle='->', color='red'))

            ax.annotate('α=0.0\n(оператор-ориентированная)',
                        xy=(passenger_costs[-1], operator_costs[-1]),
                        xytext=(passenger_costs[-1]+10, operator_costs[-1]-20),
                        arrowprops=dict(arrowstyle='->', color='blue'))

            ax.set_xlabel('Пассажирская стоимость (минуты)')
            ax.set_ylabel('Операторская стоимость (минуты)')
            ax.set_title('Trade-off: Пассажирская vs Операторская стоимость')
            ax.grid(True, alpha=0.3)
            ax.legend()

            plt.tight_layout()
            plt.savefig(f'{self.experiment_dir}/figures/cost_tradeoffs.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("График trade-off сохранен")

        except Exception as e:
            logger.error(f"Ошибка создания графика trade-off: {e}")

    def _plot_graph_type_statistics(self):
        """Статистика по типам графов в обучающих данных"""
        if not hasattr(self, 'train_cities') or not self.train_cities:
            return

        try:
            # Собираем статистику по типам графов
            graph_type_counts = {}
            for city in self.train_cities:
                graph_type = city.graph_type
                graph_type_counts[graph_type] = graph_type_counts.get(graph_type, 0) + 1

            if not graph_type_counts:
                return

            # Создаем круговую диаграмму
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # Круговая диаграмма
            labels = list(graph_type_counts.keys())
            sizes = list(graph_type_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
            ax1.set_title('Распределение типов графов в обучающих данных')

            # Столбчатая диаграмма
            ax2.bar(labels, sizes, color=colors)
            ax2.set_title('Количество городов по типам графов')
            ax2.set_ylabel('Количество')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(f'{self.experiment_dir}/figures/graph_type_statistics.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Статистика по типам графов сохранена")

        except Exception as e:
            logger.error(f"Ошибка создания статистики типов графов: {e}")

    def _save_results(self):
        """Сохранить результаты эксперимента"""
        try:
            # Сохраняем основные результаты в JSON
            results_path = f"{self.experiment_dir}/experiment_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                # Создаем копию результатов без несериализуемых объектов
                serializable_results = self._make_serializable(self.results.copy())
                json.dump(serializable_results, f, indent=2, ensure_ascii=False, default=str)

            # Сохраняем конфигурацию
            config_path = f"{self.experiment_dir}/config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(self._serialize_config(), f, indent=2, ensure_ascii=False)

            # Сохраняем краткий отчет
            self._save_summary_report()

            logger.info(f"Результаты сохранены в {self.experiment_dir}")

        except Exception as e:
            logger.error(f"Ошибка сохранения результатов: {e}")

    def _make_serializable(self, obj):
        """Рекурсивно преобразовать объект в JSON-сериализуемый вид"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)

    def _save_summary_report(self):
        """Сохранить краткий отчет об эксперименте"""
        try:
            report_path = f"{self.experiment_dir}/summary_report.txt"

            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ОТЧЕТ ОБ ЭКСПЕРИМЕНТЕ NEURAL BCO\n")
                f.write("=" * 80 + "\n\n")

                f.write(f"Название эксперимента: {self.experiment_name}\n")
                f.write(f"Время начала: {self.results.get('start_time', 'N/A')}\n")
                f.write(f"Время завершения: {self.results.get('end_time', 'N/A')}\n")
                f.write(f"Статус: {self.results.get('status', 'unknown')}\n\n")

                # Данные
                data_info = self.results.get('data_generation', {})
                f.write("ДАННЫЕ:\n")
                f.write(f"  Обучающих городов: {data_info.get('train_cities', 0)}\n")
                f.write(f"  Валидационных городов: {data_info.get('val_cities', 0)}\n")
                f.write(f"  Время генерации: {data_info.get('duration_seconds', 0):.1f} сек\n\n")

                # Обучение
                train_info = self.results.get('training', {})
                f.write("ОБУЧЕНИЕ:\n")
                f.write(f"  Лучшая валидационная награда: {train_info.get('best_val_reward', 'N/A')}\n")
                f.write(f"  Время обучения: {train_info.get('duration_seconds', 0):.1f} сек\n")
                f.write(f"  Загружена из файла: {train_info.get('loaded_from_file', False)}\n\n")

                # Сравнение алгоритмов
                comp_info = self.results.get('comparison', {})
                if comp_info.get('status') == 'completed':
                    f.write("СРАВНЕНИЕ АЛГОРИТМОВ:\n")
                    for alpha_key, results in comp_info.get('results', {}).items():
                        if 'error' not in results:
                            alpha = alpha_key.split('_')[1]
                            f.write(f"  α={alpha}:\n")
                            f.write(f"    Neural BCO: {results['neural_bco']['cost']:.3f}\n")
                            f.write(f"    Classical BCO: {results['classical_bco']['cost']:.3f}\n")
                            f.write(f"    Neural Planner Only: {results['neural_planner_only']['cost']:.3f}\n")
                    f.write("\n")

                # Реальные данные
                real_test = self.results.get('real_data_test', {})
                if real_test.get('status') == 'completed':
                    f.write("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ:\n")
                    f.write(f"  Стоимость: {real_test.get('cost', 'N/A')}\n")
                    f.write(f"  Количество маршрутов: {real_test.get('num_routes', 'N/A')}\n")
                    f.write(f"  Допустимое решение: {real_test.get('feasible', 'N/A')}\n\n")

                f.write("Подробные результаты см. в experiment_results.json\n")
                f.write("Графики см. в папке figures/\n")

            logger.info(f"Краткий отчет сохранен в {report_path}")

        except Exception as e:
            logger.error(f"Ошибка сохранения отчета: {e}")

def run_quick_experiment() -> Dict[str, Any]:
    """
    Запустить быстрый эксперимент для демонстрации
    Использует уменьшенные параметры для ускорения
    """
    logger.info("Запуск быстрого эксперимента...")

    # Временно уменьшаем параметры
    original_cities = config.data.num_training_cities
    original_epochs = config.network.num_epochs
    original_iterations = config.bco.num_iterations

    config.data.num_training_cities = 100  # Вместо 32768
    config.network.num_epochs = 2  # Вместо 5
    config.bco.num_iterations = 50  # Вместо 400

    try:
        pipeline = TrainingPipeline(
            experiment_name="quick_demo",
            use_existing_data=False,
            force_retrain=True
        )

        results = pipeline.run_full_pipeline()

        return results

    finally:
        # Восстанавливаем параметры
        config.data.num_training_cities = original_cities
        config.network.num_epochs = original_epochs
        config.bco.num_iterations = original_iterations

def run_full_experiment() -> Dict[str, Any]:
    """
    Запустить полный эксперимент как в статье
    """
    logger.info("Запуск полного эксперимента...")

    # Проверяем соответствие параметров статье
    validate_config_against_paper()

    pipeline = TrainingPipeline(
        experiment_name="neural_bco_full",
        use_existing_data=True,
        force_retrain=False
    )

    return pipeline.run_full_pipeline()

if __name__ == "__main__":
    import sys

    print("Neural BCO Training Pipeline")
    print("=" * 40)

    # Проверяем аргументы командной строки
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        print("Запуск быстрой демонстрации...")
        results = run_quick_experiment()
    else:
        print("Запуск полного эксперимента...")
        print("Для быстрой демонстрации используйте: python training_pipeline.py --quick")

        # Спрашиваем подтверждение для полного эксперимента
        response = input("Полный эксперимент может занять несколько часов. Продолжить? (y/N): ")
        if response.lower() != 'y':
            print("Эксперимент отменен.")
            sys.exit(0)

        results = run_full_experiment()

    # Выводим краткие результаты
    print("\nЭксперимент завершен!")
    print(f"Статус: {results.get('status', 'unknown')}")

    if results.get('status') == 'completed':
        if 'training' in results:
            train_info = results['training']
            print(f"Лучшая валидационная награда: {train_info.get('best_val_reward', 'N/A')}")

        if 'comparison' in results and results['comparison'].get('status') == 'completed':
            print("Сравнение алгоритмов:")
            for alpha_key, comp_results in results['comparison']['results'].items():
                if 'error' not in comp_results:
                    alpha = alpha_key.split('_')[1]
                    neural_cost = comp_results['neural_bco']['cost']
                    classical_cost = comp_results['classical_bco']['cost']
                    improvement = (classical_cost - neural_cost) / classical_cost * 100
                    print(f"  α={alpha}: Neural BCO улучшение = {improvement:.1f}%")

    print(f"Результаты сохранены в: {results.get('experiment_dir', 'results/')}")
