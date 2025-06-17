"""
Главный файл для запуска Neural Bee Colony Optimization системы
Предоставляет единую точку входа для всех компонентов системы
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import warnings
import time
import networkx as nx
from cost_functions import TransitCostCalculator
import numpy as np

  # Подавляем предупреждения для чистоты вывода.
warnings.filterwarnings('ignore')

  # Добавляем текущую директорию в путь для импорта модулей.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config, validate_config_against_paper
from training_pipeline import TrainingPipeline, run_quick_experiment, run_full_experiment
from neural_planner import train_neural_planner, NeuralPlannerTrainer
from data_generator import DatasetManager, visualize_city_examples, load_opt_fin2_city
from bco_algorithm import run_bco_optimization
from neural_bco import run_neural_bco_optimization, load_trained_neural_planner
from evaluation import run_benchmark_evaluation
from visualization import ResultsVisualizer
from utils import load_opt_fin2_data, RouteUtils

  # Настройка логирования.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def print_help():
    """Вывести справку по использованию"""
    help_text = """
    Доступные команды:

     ОСНОВНЫЕ КОМАНДЫ:
      train          - Обучить нейронную сеть с нуля
      evaluate       - Оценить обученную модель на бенчмарках
      optimize       - Запустить Neural BCO на реальных данных
      pipeline       - Полный пайплайн: данные → обучение → оценка
      pipeline-quick - Быстрая демонстрация пайплайна

     АНАЛИЗ И ДАННЫЕ:
      generate-data  - Создать синтетические города для обучения
      show-config    - Показать текущую конфигурацию
      validate       - Проверить соответствие параметров статье

     ВИЗУАЛИЗАЦИЯ:
      visualize      - Создать графики и отчеты
      demo-cities    - Показать примеры типов городов

     ДОПОЛНИТЕЛЬНЫЕ:
      compare        - Сравнить алгоритмы (BCO vs Neural BCO)
      test-data      - Проверить загрузку данных из opt_fin2.py
      benchmark      - Запустить только оценку на бенчмарках

    Примеры использования:
      python main.py pipeline-quick  # Быстрая демонстрация.
      python main.py train  # Обучение модели.
      python main.py optimize  # Оптимизация на реальных данных.
      python main.py visualize --results results/experiment_name
    """
    logger.info(help_text)

def cmd_train(args):
    """Обучить нейронную сеть"""
    logger.info("Запуск обучения нейронной сети...")

  # Создаем trainer и запускаем обучение.
    trainer = train_neural_planner()

    logger.info("Обучение завершено!")
    logger.info(f"Модель сохранена в {config.files.models_dir}")

    return {"status": "success", "trainer": trainer}

def cmd_evaluate(args):
    """Оценить обученную модель на бенчмарках"""
    logger.info("Запуск оценки на бенчмарках...")

    try:
  # Загружаем обученную модель.
        neural_planner = load_trained_neural_planner()

  # Запускаем оценку.
        results = run_benchmark_evaluation(
            neural_planner=neural_planner,
            benchmarks=['mandl', 'mumford0', 'mumford1'],  # Ограничиваем для скорости.
            num_seeds=3
        )

        logger.info("Оценка завершена!")
        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Ошибка при оценке: {e}")
        return {"status": "error", "error": str(e)}

def cmd_optimize(args):
    """Запустить Neural BCO на реальных данных"""
    logger.info(" === ЗАПУСК ОПТИМИЗАЦИИ NEURAL BCO ===")

    optimization_start = time.time()

    try:
  # Проверяем доступность устройства.
        device_info = "MPS" if config.system.device == "mps" else config.system.device.upper()
        logger.info(f" Устройство для вычислений: {device_info}")

  # Проверка файла данных с детальным логированием.
        opt_stops_file = config.files.opt_stops_file
        if not os.path.exists(opt_stops_file):
            logger.error(f" Файл {opt_stops_file} не найден в корневой директории")
            logger.info(" Поместите файл opt_stops.pkl в корневую папку проекта")
            return {"status": "error", "error": "opt_stops.pkl not found"}

        file_size = os.path.getsize(opt_stops_file)
        logger.info(f" Загрузка данных из {opt_stops_file} (размер: {file_size / 1024:.1f} KB)")

  # Загрузка данных с валидацией.
        data_start = time.time()
        opt_stops, metadata = load_opt_fin2_data()
        data_load_time = time.time() - data_start

        if len(opt_stops) == 0:
            logger.error(f" Файл {opt_stops_file} пуст или поврежден")
            return {"status": "error", "error": "Invalid opt_stops.pkl"}

        logger.info(f" Данные загружены за {data_load_time:.2f}s:")
        logger.info(f"    Остановок: {len(opt_stops)}")
        logger.info(f"    Колонки: {list(opt_stops.columns)}")

  # Анализ типов остановок.
        if 'type' in opt_stops.columns:
            type_counts = opt_stops['type'].value_counts()
            logger.info(f"   ️  Типы остановок:")
            for stop_type, count in type_counts.items():
                logger.info(f"      {stop_type}: {count} ({count / len(opt_stops) * 100:.1f}%)")

  # Создание города с валидацией.
        city_start = time.time()
        logger.info("️  Создание графа города...")

        from data_generator import load_opt_fin2_city
        city = load_opt_fin2_city()
        city_creation_time = time.time() - city_start

        if city is None:
            logger.error(" Не удалось создать граф города из данных opt_stops.pkl")
            return {"status": "error", "error": "City creation failed"}

        logger.info(f" Граф города создан за {city_creation_time:.2f}s:")
        logger.info(f"   ️  Узлов: {len(city.city_graph.nodes())}")
        logger.info(f"   ️  Ребер: {len(city.city_graph.edges())}")
        logger.info(f"    Общий спрос: {city.od_matrix.sum():.0f} поездок/день")
        logger.info(f"    Размер OD матрицы: {city.od_matrix.shape}")

  # Детальный анализ входного графа.
        logger.info(" === АНАЛИЗ ВХОДНОГО ГРАФА ===")

  # Анализ связности.
        if nx.is_connected(city.city_graph):
            logger.info("    Граф связный: ")
        else:
            components = list(nx.connected_components(city.city_graph))
            largest_cc = max(components, key=len)
            logger.warning(f"    Граф несвязный: {len(components)} компонентов")
            logger.warning(f"    Крупнейший компонент: {len(largest_cc)} узлов")

  # Анализ степеней узлов.
        degrees = [city.city_graph.degree(n) for n in city.city_graph.nodes()]
        logger.info(f"    Степени узлов: мин={min(degrees)}, макс={max(degrees)}, средняя={np.mean(degrees):.1f}")

  # Проверяем изолированные узлы.
        isolated = [n for n in city.city_graph.nodes() if city.city_graph.degree(n) == 0]
        if isolated:
            logger.warning(f"   ️ Изолированных узлов: {len(isolated)}")

  # Предварительная оценка сложности.
        avg_degree = np.mean(degrees)
        complexity_score = len(city.city_graph.nodes()) * avg_degree / 100
        logger.info(f"    Оценка сложности: {complexity_score:.1f} (чем больше, тем сложнее)")

  # Анализ сложности задачи.
        logger.info(" Анализ параметров задачи:")
        logger.info(f"    Целевое количество маршрутов: {config.mdp.num_routes}")
        logger.info(f"    Длина маршрута: {config.mdp.min_route_length}-{config.mdp.max_route_length} остановок")
        logger.info(f"    Параметр α: {0.5} (баланс пассажир/оператор)")

  # Оценка времени выполнения.
        estimated_time = (config.bco.num_iterations * len(city.city_graph.nodes())) / 1000
        logger.info(f"   ️  Ожидаемое время: ~{estimated_time:.0f} секунд ({estimated_time / 60:.1f} мин)")

  # Запуск оптимизации с мониторингом.
        logger.info(" === ЗАПУСК NEURAL BCO ОПТИМИЗАЦИИ ===")
        optimization_start = time.time()

        save_raw = getattr(args, 'raw_routes', None)
        postproc = not getattr(args, 'no_smoothing', False)

        results = run_neural_bco_optimization(
            city.city_graph,
            city.od_matrix,
            alpha=0.5,
            save_raw_routes=save_raw,  # ← новое.
            post_process=postproc  # ← новое.
        )

        optimization_time = time.time() - optimization_start

  # Детальный анализ результатов.
        logger.info(" === АНАЛИЗ РЕЗУЛЬТАТОВ ===")

        neural_result = results['neural_bco_result']
        neural_stats = results.get('neural_statistics', {})

        logger.info(f"️  Общее время оптимизации: {optimization_time:.1f}s ({optimization_time / 60:.1f} мин)")
        logger.info(f" Финальная стоимость: {neural_result.cost:.3f}")
        logger.info(f"️  Создано маршрутов: {len(neural_result.routes)}")
        logger.info(f" Решение допустимо: {neural_result.is_feasible}")

  # Анализ качества маршрутов.
        if neural_result.routes:
            route_lengths = [len(route) for route in neural_result.routes]
            covered_nodes = set()
            for route in neural_result.routes:
                covered_nodes.update(route)

            logger.info(" === АНАЛИЗ КАЧЕСТВА МАРШРУТОВ ===")
            logger.info(f" Длины маршрутов:")
            logger.info(f"   Минимальная: {min(route_lengths)} остановок")
            logger.info(f"   Максимальная: {max(route_lengths)} остановок")
            logger.info(f"   Средняя: {np.mean(route_lengths):.1f} остановок")
            logger.info(f"   Стандартное отклонение: {np.std(route_lengths):.1f}")

            logger.info(f"️  Покрытие сети:")
            coverage_percent = len(covered_nodes) / len(city.city_graph.nodes()) * 100
            logger.info(
                f"   Покрыто узлов: {len(covered_nodes)}/{len(city.city_graph.nodes())} ({coverage_percent:.1f}%)")

  # Анализ пересечений маршрутов.
            overlaps = []
            for i in range(len(neural_result.routes)):
                for j in range(i + 1, len(neural_result.routes)):
                    overlap = len(set(neural_result.routes[i]).intersection(set(neural_result.routes[j])))
                    overlaps.append(overlap)

            if overlaps:
                logger.info(f" Пересечения маршрутов:")
                logger.info(f"   Среднее пересечение: {np.mean(overlaps):.1f} узлов")
                logger.info(f"   Максимальное пересечение: {max(overlaps)} узлов")

  # Детальный вывод маршрутов (только для отладки).
            logger.debug(f" НАЙДЕННЫЕ МАРШРУТЫ:")
            for i, route in enumerate(neural_result.routes):
                logger.debug(f"  Маршрут {i + 1}: {route} (длина: {len(route)} остановок)")

  # Анализ эффективности Neural Bees.
        logger.info(" === АНАЛИЗ NEURAL BEES ===")
        logger.info(f" Успешность Neural Bees: {neural_stats.get('avg_success_rate', 0):.1%}")
        logger.info(f" Всего вызовов политики: {neural_stats.get('total_calls', 0)}")
        logger.info(f"️  Время работы политики: {neural_stats.get('total_policy_time', 0):.1f}s")

        if neural_stats.get('total_calls', 0) > 0:
            avg_time = neural_stats.get('total_policy_time', 0) / neural_stats.get('total_calls', 1)
            logger.info(f" Среднее время на вызов: {avg_time:.4f}s")

  # Детальная декомпозиция стоимости.
        logger.info(" === ДЕТАЛЬНЫЙ АНАЛИЗ СТОИМОСТИ ===")
        from cost_functions import TransitCostCalculator
        cost_calculator = TransitCostCalculator(alpha=0.5, beta=config.mdp.constraint_penalty_weight)
        cost_breakdown = cost_calculator.get_cost_breakdown(
            city.city_graph, neural_result.routes, city.od_matrix, 0.5
        )

        logger.info(f" Пассажирская стоимость: {cost_breakdown['passenger_cost']:.1f} мин")
        logger.info(f" Операторская стоимость: {cost_breakdown['operator_cost']:.1f} мин")
        logger.info(f"️  Штрафы за ограничения: {cost_breakdown['constraint_cost']:.3f}")
        logger.info(f" Связность сети: {cost_breakdown['connectivity_ratio']:.1%}")
        logger.info(f" Нарушения длины: {cost_breakdown['length_violations']}")
        logger.info(f" Количество пересадок: {cost_breakdown['transfer_count']}")

  # Сводная таблица производительности.
        total_time = time.time() - optimization_start
        logger.info(" === СВОДКА ПРОИЗВОДИТЕЛЬНОСТИ ===")
        logger.info(f" Загрузка данных: {data_load_time:.2f}s")
        logger.info(f"️  Создание города: {city_creation_time:.2f}s")
        logger.info(f" Оптимизация: {optimization_time:.2f}s")
        logger.info(f" Общее время: {total_time:.2f}s")

  # Сохранение подробных результатов.
        detailed_results = {
            'neural_bco_result': neural_result,
            'cost_breakdown': cost_breakdown,
            'neural_statistics': neural_stats,
            'performance_metrics': {
                'data_load_time': data_load_time,
                'city_creation_time': city_creation_time,
                'optimization_time': optimization_time,
                'total_time': total_time
            },
            'data_info': {
                'num_stops': len(opt_stops),
                'num_nodes': len(city.city_graph.nodes()),
                'num_edges': len(city.city_graph.edges()),
                'total_demand': city.od_matrix.sum()
            }
        }

  # НОВОЕ: Сохранение результатов для визуализации.
        experiment_name = f"neural_bco_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        experiment_dir = os.path.join(config.files.results_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(os.path.join(experiment_dir, 'figures'), exist_ok=True)

  # Подготавливаем результаты для сохранения.
  # В функции cmd_optimize добавить после optimization_time = time.time() - optimization_start.

  # НОВОЕ: Извлекаем детальные данные из результатов оптимизации.
        iteration_history = results.get('iteration_history', [])
        neural_performance = results.get('neural_performance', [])
        route_analysis = results.get('route_analysis', {})

  # Дополняем анализ маршрутов если данных недостаточно.
        if neural_result.routes and not route_analysis:
            route_lengths = [len(route) for route in neural_result.routes]
            covered_nodes = set()
            for route in neural_result.routes:
                covered_nodes.update(route)

  # Пересечения между маршрутами.
            route_overlaps = []
            for i in range(len(neural_result.routes)):
                for j in range(i + 1, len(neural_result.routes)):
                    overlap = len(set(neural_result.routes[i]).intersection(set(neural_result.routes[j])))
                    route_overlaps.append({
                        'route_1': i,
                        'route_2': j,
                        'overlap_count': overlap,
                        'overlap_ratio': overlap / min(len(neural_result.routes[i]),
                                                       len(neural_result.routes[j])) if min(
                            len(neural_result.routes[i]), len(neural_result.routes[j])) > 0 else 0
                    })

  # Покрытие по типам остановок.
            coverage_by_type = {}
            if 'type' in opt_stops.columns:
                for stop_type in ['key', 'connection', 'ordinary']:
                    type_stops = opt_stops[opt_stops['type'] == stop_type]
                    if len(type_stops) > 0:
                        type_covered = sum(1 for _, stop in type_stops.iterrows() if stop['node_id'] in covered_nodes)
                        coverage_by_type[stop_type] = {
                            'total': len(type_stops),
                            'covered': type_covered,
                            'coverage_ratio': type_covered / len(type_stops)
                        }

            route_analysis = {
                'route_lengths': route_lengths,
                'covered_nodes': list(covered_nodes),
                'total_route_nodes': sum(route_lengths),  # С повторениями.
                'route_overlaps': route_overlaps,
                'coverage_by_type': coverage_by_type,
                'avg_route_length': np.mean(route_lengths),
                'route_length_std': np.std(route_lengths)
            }

        experiment_results = {
            'experiment_name': experiment_name,
            'start_time': datetime.now().isoformat(),
            'status': 'completed',
            'neural_bco_result': {
                'cost': neural_result.cost,
                'routes': neural_result.routes,
                'is_feasible': neural_result.is_feasible,
                'num_routes': len(neural_result.routes)
            },
  # НОВОЕ: Расширенные данные для визуализации.
            'iteration_history': iteration_history,
            'neural_performance': neural_performance,
            'route_analysis': route_analysis,
            'neural_statistics': neural_stats,
            'cost_breakdown': cost_breakdown,
            'performance_metrics': detailed_results['performance_metrics'],
            'data_info': detailed_results['data_info'],
            'config': {
                'bco_iterations': config.bco.num_iterations,
                'bco_bees': config.bco.num_bees,
                'neural_bee_ratio': config.bco.neural_bee_ratio,
                'alpha': 0.5,
                'device': config.system.device
            }
        }

  # Сохраняем в JSON.
        results_file = os.path.join(experiment_dir, 'experiment_results.json')
        try:
            import json
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_results, f, indent=2, ensure_ascii=False, default=str)

            logger.info(f" Результаты сохранены в {experiment_dir}")
            logger.info(f" Для создания визуализаций запустите:")
            logger.info(f"    python main.py visualize --results-dir {experiment_dir}")

        except Exception as save_error:
            logger.warning(f"️  Не удалось сохранить результаты: {save_error}")

  # Автоматически создаем визуализации.
        try:
            logger.info(" Создание визуализаций...")
            from visualization import ResultsVisualizer

            visualizer = ResultsVisualizer(experiment_dir)
            if config.evaluation.focus_neural_only:
                visualizer.plot_neural_bco_detailed_analysis()
                visualizer.plot_route_quality_metrics()
                visualizer.plot_neural_bee_performance()
                visualizer.plot_network_coverage_analysis()
                visualizer.create_neural_summary_report()
                logger.info(" Визуализации Neural BCO созданы")
            else:
                visualizer.create_all_visualizations()
                logger.info(" Все визуализации созданы")

        except Exception as viz_error:
            logger.warning(f"️  Не удалось создать визуализации: {viz_error}")
            logger.info(" Попробуйте запустить: python main.py visualize")

        logger.info(" === ОПТИМИЗАЦИЯ ЗАВЕРШЕНА УСПЕШНО ===")
        return {"status": "success", "results": results, "detailed_analysis": detailed_results,
                "experiment_dir": experiment_dir}

    except Exception as e:
        logger.error(f" Критическая ошибка при оптимизации: {e}")
        return {"status": "error", "error": str(e)}

def cmd_pipeline(args):
    """Запустить полный пайплайн"""
    logger.info("Запуск полного пайплайна Neural BCO...")

  # Проверяем соответствие параметров статье.
    validate_config_against_paper()

    try:
        results = run_full_experiment()

        logger.info("Полный пайплайн завершен!")
        logger.info(f"Статус: {results.get('status', 'unknown')}")

        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Ошибка в пайплайне: {e}")
        return {"status": "error", "error": str(e)}

def cmd_pipeline_quick(args):
    """Запустить быструю демонстрацию пайплайна"""
    logger.info("Запуск быстрой демонстрации пайплайна...")

    try:
        results = run_quick_experiment()

        logger.info("Быстрая демонстрация завершена!")
        logger.info(f"Статус: {results.get('status', 'unknown')}")

        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Ошибка в быстрой демонстрации: {e}")
        return {"status": "error", "error": str(e)}

def cmd_generate_data(args):
    """Создать синтетические города для обучения"""
    logger.info("Создание синтетических данных...")

  # Применяем параметры из аргументов к конфигурации.
    if hasattr(args, 'num_cities') and args.num_cities:
        config.data.num_training_cities = args.num_cities

    if hasattr(args, 'num_nodes') and args.num_nodes:
        config.data.num_nodes = args.num_nodes

    if hasattr(args, 'graph_types') and args.graph_types:
        config.data.graph_types = args.graph_types

    if hasattr(args, 'city_area') and args.city_area:
        config.data.city_area_km = args.city_area / 1000  # Конвертируем м в км.

    if hasattr(args, 'demand_min') and args.demand_min:
        config.data.demand_range = (args.demand_min, config.data.demand_range[1])

    if hasattr(args, 'demand_max') and args.demand_max:
        config.data.demand_range = (config.data.demand_range[0], args.demand_max)

    logger.info(f"Параметры генерации:")
    logger.info(f"  Городов: {config.data.num_training_cities}")
    logger.info(f"  Узлов в городе: {config.data.num_nodes}")
    logger.info(f"  Типы графов: {config.data.graph_types}")
    logger.info(f"  Размер города: {config.data.city_area_km} км")

    dataset_manager = DatasetManager()
    cities = dataset_manager.create_training_dataset(num_cities=config.data.num_training_cities)

    logger.info(f"Создано {len(cities)} синтетических городов")

    return {"status": "success", "num_cities": len(cities)}

def cmd_show_config(args):
    """Показать текущую конфигурацию"""
    logger.info("Текущая конфигурация системы:")
    config.print_summary()

    return {"status": "success"}

def cmd_validate(args):
    """Проверить соответствие параметров статье"""
    logger.info("Проверка соответствия параметров статье...")
    validate_config_against_paper()

    return {"status": "success"}

def cmd_visualize(args):
    """Создать графики и отчеты"""
    logger.info("Создание визуализаций...")

    try:
  # Определяем директорию с результатами.
        results_dir = getattr(args, 'results_dir', None)
        if results_dir is None:
  # Ищем последний эксперимент.
            results_base = config.files.results_dir
            if os.path.exists(results_base):
                experiments = [d for d in os.listdir(results_base)
                               if os.path.isdir(os.path.join(results_base, d))]
                if experiments:
                    experiments.sort(key=lambda x: os.path.getctime(os.path.join(results_base, x)))
                    results_dir = os.path.join(results_base, experiments[-1])

        if results_dir and os.path.exists(results_dir):
            visualizer = ResultsVisualizer(results_dir)
            visualizer.create_all_visualizations()
            logger.info(f"Визуализации созданы в {results_dir}/figures/")
        else:
            logger.warning("Не найдены результаты для визуализации")
  # Создаем демонстрационные графики.
            visualizer = ResultsVisualizer()
            visualizer.create_demo_visualizations()
            logger.info("Созданы демонстрационные визуализации")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Ошибка при создании визуализаций: {e}")
        return {"status": "error", "error": str(e)}

def cmd_demo_cities(args):
    """Показать примеры типов городов"""
    logger.info("Создание примеров синтетических городов...")

    try:
        visualize_city_examples()
        logger.info("Примеры городов сохранены в synthetic_city_examples.png")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Ошибка при создании примеров: {e}")
        return {"status": "error", "error": str(e)}

def cmd_compare(args):
    """Сравнить алгоритмы"""
    if not config.evaluation.enable_classical_comparison:
        logger.info("Сравнение с классическим BCO отключено в конфигурации")
        logger.info("Для включения установите config.evaluation.enable_classical_comparison = True")
        return {"status": "skipped", "reason": "comparison_disabled"}

    logger.info("Сравнение алгоритмов BCO vs Neural BCO...")

    try:
  # Загружаем тестовый город.
        from data_generator import load_opt_fin2_city
        city = load_opt_fin2_city()

        if city is None:
            logger.warning("Не удалось загрузить данные opt_fin2, создаем тестовый город")
  # Создаем простой тестовый граф.
            import networkx as nx
            G = nx.grid_2d_graph(6, 6)
            G = nx.convert_node_labels_to_integers(G)

            for u, v in G.edges():
                G[u][v]['travel_time'] = 60.0

            od_matrix = np.random.randint(10, 100, size=(len(G.nodes()), len(G.nodes())))
            np.fill_diagonal(od_matrix, 0)
        else:
            G = city.city_graph
            od_matrix = city.od_matrix

        logger.info("Запуск классического BCO...")

  # Уменьшаем параметры для быстрого сравнения.
        original_iterations = config.bco.num_iterations
        config.bco.num_iterations = 100

        try:
  # Классический BCO.
            classical_result = run_bco_optimization(G, od_matrix, alpha=0.5,
                                                    save_raw_routes=getattr(args, 'raw_routes', None),
                                                    post_process=not getattr(args, 'no_smoothing', False))

  # Neural BCO (если есть обученная модель).
            try:
                save_raw = getattr(args, 'raw_routes', None)
                postproc = not getattr(args, 'no_smoothing', False)

                neural_result = run_neural_bco_optimization(
                    G, od_matrix, alpha=0.5, compare_with_classical=False, save_raw_routes=save_raw, post_process=postproc
                )

                logger.info("Результаты сравнения:")
                logger.info(f"Classical BCO: {classical_result.cost:.3f}")
                logger.info(f"Neural BCO: {neural_result['neural_bco_result'].cost:.3f}")

                improvement = (classical_result.cost - neural_result[
                    'neural_bco_result'].cost) / classical_result.cost * 100
                logger.info(f"Улучшение Neural BCO: {improvement:.1f}%")

            except Exception as e:
                logger.warning(f"Не удалось запустить Neural BCO: {e}")
                logger.info(f"Classical BCO результат: {classical_result.cost:.3f}")

        finally:
            config.bco.num_iterations = original_iterations

        return {"status": "success"}

    except Exception as e:

        logger.error(f"Ошибка при сравнении: {e}")

        return {"status": "error", "error": str(e)}

def cmd_smooth(args):
    """Запустить только 2-/3-opt над готовыми маршрутами"""
    logger.info(" Пост-обработка маршрутов…")
    try:
  # 1. Загружаем исходные маршруты.
        routes = RouteUtils.load_routes(args.input)
        logger.info(f"Загружено {len(routes)} маршрутов из {args.input}")

  # 2. Граф и OD-матрица.
        from data_generator import load_opt_fin2_city
        city = load_opt_fin2_city()
        if city is None:
            raise RuntimeError("load_opt_fin2_city() вернул None – проверьте, что opt_stops.pkl существует и корректен")

        G, od = city.city_graph, city.od_matrix

  # 3. Выбираем метод.
        if args.method == '3opt' and hasattr(RouteUtils, 'three_opt'):
            new_routes = RouteUtils.three_opt(routes, G)
        else:
            new_routes = RouteUtils.smooth_routes(routes, G)

  # 4. Сравниваем стоимость.
        calc = TransitCostCalculator(alpha=0.5)
        old_cost = calc.calculate_cost(G, routes, od)
        new_cost = calc.calculate_cost(G, new_routes, od)

        logger.info(f" Cost: {old_cost:.3f} → {new_cost:.3f}")

  # 5. Сохраняем.
        RouteUtils.save_routes(new_routes, args.output)
        logger.info(f" Сглаженные маршруты сохранены в {args.output}")

        return {"status": "success",
                "before": old_cost, "after": new_cost,
                "saved_to": args.output}

    except Exception as e:
        logger.error(f"Ошибка в cmd_smooth: {e}")
        return {"status": "error", "error": str(e)}

def cmd_test_data(args):
    """Проверить загрузку данных из opt_fin2.py"""
    logger.info("Проверка загрузки данных из opt_fin2.py...")

    try:
        opt_stops, metadata = load_opt_fin2_data()

        if len(opt_stops) > 0:
            logger.info(f" Успешно загружено {len(opt_stops)} остановок")
            logger.info(f"Метаданные: {metadata}")

  # Проверяем создание города.
            from data_generator import load_opt_fin2_city
            city = load_opt_fin2_city()

            if city:
                logger.info(
                    f" Город создан: {len(city.city_graph.nodes())} узлов, {len(city.city_graph.edges())} ребер")
            else:
                logger.warning(" Не удалось создать город из данных")
        else:
            logger.warning(" Данные opt_fin2.py не найдены или пусты")
            logger.info("Убедитесь, что файл opt_stops.pkl существует в текущей директории")

        return {"status": "success", "data_found": len(opt_stops) > 0}

    except Exception as e:
        logger.error(f"Ошибка при проверке данных: {e}")
        return {"status": "error", "error": str(e)}

def cmd_benchmark(args):
    """Запустить только оценку на бенчмарках"""
    logger.info("Запуск оценки на бенчмарках...")

    try:
  # Пытаемся загрузить обученную модель.
        try:
            neural_planner = load_trained_neural_planner()
            logger.info("Обученная модель загружена")
        except:
            logger.warning("Не удалось загрузить обученную модель, используем только классический BCO")
            neural_planner = None

        results = run_benchmark_evaluation(
            neural_planner=neural_planner,
            benchmarks=getattr(args, 'benchmarks', ['mandl', 'mumford0']),
            num_seeds=getattr(args, 'num_seeds', 3)
        )

        return {"status": "success", "results": results}

    except Exception as e:
        logger.error(f"Ошибка при оценке на бенчмарках: {e}")
        return {"status": "error", "error": str(e)}

def main():
    """Главная функция"""
  # Создаем парсер аргументов.
    parser = argparse.ArgumentParser(
        description="Neural Bee Colony Optimization для транзитных сетей",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

  # Добавляем подкоманды.
    subparsers = parser.add_subparsers(dest='command', help='Доступные команды')

    parser_smooth = subparsers.add_parser(
        'smooth', help='Применить 2-/3-opt к сохранённым маршрутам'
    )
    parser_smooth.add_argument('--input', required=True, help='Файл с исходными маршрутами')
    parser_smooth.add_argument('--output', required=True, help='Куда сохранить сглаженные маршруты')
    parser_smooth.add_argument('--method', choices=['2opt', '3opt'], default='2opt')

  # Основные команды.
    subparsers.add_parser('train', help='Обучить нейронную сеть')
    subparsers.add_parser('evaluate', help='Оценить модель на бенчмарках')
    parser_opt = subparsers.add_parser(
        'optimize',
        help = 'Запустить Neural/Classic BCO; можно сохранить маршруты до 2-opt'
    )
    parser_opt.add_argument('--raw-routes', help='Файл .pickle для сохранения маршрутов ДО 2-opt')
    parser_opt.add_argument('--no-smoothing', action='store_true',
        help = 'Не выполнять 2-opt в самом конце оптимизации')
    subparsers.add_parser('pipeline', help='Полный пайплайн')
    subparsers.add_parser('pipeline-quick', help='Быстрая демонстрация')

  # Анализ и данные.
    data_parser = subparsers.add_parser('generate-data', help='Создать синтетические данные')
    data_parser.add_argument('--num-cities', type=int, default=32768, help='Количество городов')
    data_parser.add_argument('--num-nodes', type=int, default=20, help='Узлов в городе')
    data_parser.add_argument('--graph-types', nargs='+',
                             default=['incoming_4nn', 'outgoing_4nn', 'voronoi', '4_grid', '8_grid'],
                             help='Типы графов')
    data_parser.add_argument('--city-area', type=float, default=30000, help='Размер города (м)')
    data_parser.add_argument('--demand-min', type=int, default=60, help='Минимальный спрос')
    data_parser.add_argument('--demand-max', type=int, default=800, help='Максимальный спрос')
    data_parser.add_argument('--enable-augmentation', action='store_true', help='Включить аугментацию')
    data_parser.add_argument('--output-dir', default='data', help='Директория для сохранения')

    subparsers.add_parser('show-config', help='Показать конфигурацию')
    subparsers.add_parser('validate', help='Проверить соответствие статье')

  # Визуализация.
    viz_parser = subparsers.add_parser('visualize', help='Создать визуализации')
    viz_parser.add_argument('--results-dir', help='Директория с результатами')

    subparsers.add_parser('demo-cities', help='Примеры типов городов')

  # Дополнительные.
    subparsers.add_parser('compare', help='Сравнить алгоритмы (если включено)')
    subparsers.add_parser('test-data', help='Проверить данные opt_fin2')

  # Новые команды для анализа Neural BCO.
    subparsers.add_parser('neural-metrics', help='Детальные метрики Neural BCO')
    subparsers.add_parser('route-analysis', help='Анализ качества маршрутов')
    subparsers.add_parser('coverage-analysis', help='Анализ покрытия сети')

    bench_parser = subparsers.add_parser('benchmark', help='Оценка на бенчмарках')
    bench_parser.add_argument('--benchmarks', nargs='+', default=['mandl', 'mumford0'])
    bench_parser.add_argument('--num-seeds', type=int, default=3)

  # Парсим аргументы.
    args = parser.parse_args()

  # Если команда не указана, показываем справку.
    if not args.command:
        print_help()
        return

  # Создаем словарь команд.
    commands = {
        'train': cmd_train,
        'evaluate': cmd_evaluate,
        'optimize': cmd_optimize,
        'pipeline': cmd_pipeline,
        'pipeline-quick': cmd_pipeline_quick,
        'generate-data': cmd_generate_data,
        'show-config': cmd_show_config,
        'smooth': cmd_smooth,
        'validate': cmd_validate,
        'visualize': cmd_visualize,
        'demo-cities': cmd_demo_cities,
        'compare': cmd_compare,
        'test-data': cmd_test_data,
        'benchmark': cmd_benchmark,
        'neural-metrics': cmd_neural_metrics,
        'route-analysis': cmd_route_analysis,
        'coverage-analysis': cmd_coverage_analysis,
    }

  # Выполняем команду.
    if args.command in commands:
        try:
            start_time = datetime.now()
            result = commands[args.command](args)
            end_time = datetime.now()

            duration = end_time - start_time
            logger.info(f"Команда '{args.command}' выполнена за {duration}")

            if result.get('status') == 'error':
                logger.error(f"Команда завершилась с ошибкой: {result.get('error')}")
                sys.exit(1)

        except KeyboardInterrupt:
            logger.info("Операция прервана пользователем")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}", exc_info=True)
            sys.exit(1)
    else:
        logger.error(f"Неизвестная команда: {args.command}")
        print_help()
        sys.exit(1)

    logger.info("Программа завершена успешно!")

def cmd_neural_metrics(args):
    """Показать детальные метрики Neural BCO"""
    logger.info("Анализ метрик Neural BCO...")

    try:
  # Загружаем данные и запускаем Neural BCO.
        city = load_opt_fin2_city()
        if city is None:
            return {"status": "error", "error": "No city data"}

        save_raw = getattr(args, 'raw_routes', None)
        postproc = not getattr(args, 'no_smoothing', False)

        results = run_neural_bco_optimization(
            city.city_graph, city.od_matrix, alpha=0.5, compare_with_classical=False, save_raw_routes=save_raw, post_process=postproc
        )

        neural_stats = results.get('neural_statistics', {})

        logger.info("=== МЕТРИКИ NEURAL BEES ===")
        logger.info(f"Общие вызовы: {neural_stats.get('total_calls', 0)}")
        logger.info(f"Успешные улучшения: {neural_stats.get('total_successes', 0)}")
        logger.info(f"Успешность: {neural_stats.get('avg_success_rate', 0):.1%}")
        logger.info(f"Среднее время на вызов: {neural_stats.get('avg_policy_time_per_call', 0):.3f}s")

        return {"status": "success", "metrics": neural_stats}

    except Exception as e:
        logger.error(f"Ошибка анализа метрик: {e}")
        return {"status": "error", "error": str(e)}

def cmd_route_analysis(args):
    """Анализ качества построенных маршрутов"""
    logger.info("Анализ качества маршрутов...")

    try:
        city = load_opt_fin2_city()
        if city is None:
            return {"status": "error", "error": "No city data"}

        save_raw = getattr(args, 'raw_routes', None)
        postproc = not getattr(args, 'no_smoothing', False)

        results = run_neural_bco_optimization(
            city.city_graph, city.od_matrix, alpha=0.5, compare_with_classical=False, save_raw_routes=save_raw, post_process=postproc
        )

        routes = results['neural_bco_result'].routes

        logger.info("=== АНАЛИЗ МАРШРУТОВ ===")
        logger.info(f"Количество маршрутов: {len(routes)}")

        if routes:
            lengths = [len(route) for route in routes]
            logger.info(f"Длины маршрутов: мин={min(lengths)}, макс={max(lengths)}, среднее={np.mean(lengths):.1f}")

  # Анализ покрытия.
            all_nodes = set(city.city_graph.nodes())
            covered_nodes = set()
            for route in routes:
                covered_nodes.update(route)

            logger.info(
                f"Покрытие узлов: {len(covered_nodes)}/{len(all_nodes)} ({len(covered_nodes) / len(all_nodes) * 100:.1f}%)")

  # Анализ пересечений маршрутов.
            overlaps = []
            for i in range(len(routes)):
                for j in range(i + 1, len(routes)):
                    overlap = len(set(routes[i]).intersection(set(routes[j])))
                    overlaps.append(overlap)

            if overlaps:
                logger.info(f"Пересечения маршрутов: среднее={np.mean(overlaps):.1f} узлов")

        return {"status": "success", "routes": routes}

    except Exception as e:
        logger.error(f"Ошибка анализа маршрутов: {e}")
        return {"status": "error", "error": str(e)}

def cmd_coverage_analysis(args):
    """Анализ покрытия населения и рабочих мест"""
    logger.info("Анализ покрытия сети...")

    try:
        opt_stops, metadata = load_opt_fin2_data()
        city = load_opt_fin2_city()

        if city is None:
            return {"status": "error", "error": "No city data"}

        save_raw = getattr(args, 'raw_routes', None)
        postproc = not getattr(args, 'no_smoothing', False)

        results = run_neural_bco_optimization(
            city.city_graph, city.od_matrix, alpha=0.5, compare_with_classical=False, save_raw_routes=save_raw, post_process=postproc
        )

        routes = results['neural_bco_result'].routes

        logger.info("=== АНАЛИЗ ПОКРЫТИЯ ===")

  # Анализ покрытия по типам остановок.
        if 'type' in opt_stops.columns:
            covered_nodes = set()
            for route in routes:
                covered_nodes.update(route)

            for stop_type in ['key', 'connection', 'ordinary']:
                type_stops = opt_stops[opt_stops['type'] == stop_type]
                covered_type = sum(1 for _, stop in type_stops.iterrows()
                                   if stop['node_id'] in covered_nodes)

                logger.info(f"Покрытие {stop_type} остановок: {covered_type}/{len(type_stops)} "
                            f"({covered_type / len(type_stops) * 100:.1f}%)")

  # Анализ покрытия населения (если данные есть).
        if 'population' in opt_stops.columns:
            total_pop = opt_stops['population'].sum()
            covered_pop = opt_stops[opt_stops['node_id'].isin(covered_nodes)]['population'].sum()
            logger.info(f"Покрытие населения: {covered_pop}/{total_pop} ({covered_pop / total_pop * 100:.1f}%)")

        return {"status": "success"}

    except Exception as e:
        logger.error(f"Ошибка анализа покрытия: {e}")
        return {"status": "error", "error": str(e)}

if __name__ == "__main__":
    main()
