  # Config.py (обновленная версия для Mac).
"""
Конфигурационный файл для Neural Bee Colony Optimization для транзитных сетей
Основан на статье "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"
Оптимизирован для работы на Mac (MPS/CPU)
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

# ---------------------------------------------------------------------------
# Core project directories (do NOT hard‑code strings in other modules!)
# ---------------------------------------------------------------------------

ROOT_DIR: Path = Path(__file__).resolve().parents[1]

DATA_DIR:     Path = ROOT_DIR / "data"
CLUSTER_DIR:  Path = DATA_DIR   / "clusters"
RESULTS_DIR:  Path = ROOT_DIR / "results"
MODELS_DIR:   Path = ROOT_DIR / "models"
EXP_DIR:      Path = ROOT_DIR / "experiments"
LOG_DIR:      Path = ROOT_DIR / "logs"

for _dir in (DATA_DIR, CLUSTER_DIR, RESULTS_DIR, MODELS_DIR, EXP_DIR, LOG_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Path helper‑functions
# ---------------------------------------------------------------------------

def data(*parts: str | os.PathLike) -> Path:
    """Return Path inside *data/*."""
    return DATA_DIR.joinpath(*parts)

def clusters(*parts: str | os.PathLike) -> Path:
    """Return Path inside *data/clusters/*."""
    return CLUSTER_DIR.joinpath(*parts)

def results(*parts: str | os.PathLike) -> Path:
    return RESULTS_DIR.joinpath(*parts)

def models(*parts: str | os.PathLike) -> Path:
    return MODELS_DIR.joinpath(*parts)

def exp(run_id: str, *parts: str | os.PathLike) -> Path:
    return EXP_DIR.joinpath(run_id, *parts)

# ---------------------------------------------------------------------------
# Global constants / defaults
# ---------------------------------------------------------------------------

TZ            = "Europe/Helsinki"
RANDOM_SEED   = 42
NUM_WORKERS   = os.cpu_count() or 1
__all__ = []

__all__ += [
    "ROOT_DIR", "DATA_DIR", "CLUSTER_DIR", "RESULTS_DIR",
    "MODELS_DIR", "EXP_DIR", "LOG_DIR",
    "data", "clusters", "results", "models", "exp",
    "TZ", "RANDOM_SEED", "NUM_WORKERS",
]

import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import os


@dataclass
class NetworkConfig:
    """Конфигурация нейронной сети"""
  # Graph Attention Network параметры.
    hidden_dim: int = 128
    num_attention_heads: int = 8
    num_gat_layers: int = 3
    dropout_rate: float = 0.1
    edge_feature_dim: int = 6  # Demand, distance, existing_transit, etc.
    node_feature_dim: int = 8  # Population, jobs, centrality, etc.

  # Policy network heads.
    extension_head_dim: int = 64
    halt_head_dim: int = 32

  # Training параметры.
    learning_rate: float = 3e-4
    batch_size: int = 64
    num_epochs: int = 5
    baseline_learning_rate: float = 1e-4
  # REINFORCE параметры.
    gamma: float = 1.0  # Discount factor.
    entropy_bonus: float = 0.01
    baseline_update_frequency: int = 1


@dataclass
class MDPConfig:
    """Конфигурация MDP для транзитного планирования"""
  # Ограничения маршрутов.
    min_route_length: int = 3
    max_route_length: int = 15
    num_routes: int = 10

  # Штрафы и веса в функции стоимости.
    transfer_penalty: float = 5.0  # Минуты штрафа за пересадку.
    constraint_penalty_weight: float = 5.0  # Β в статье.

  # Cost function weights (α параметр).
    cost_alpha_range: Tuple[float, float] = (0.0, 1.0)
    cost_alpha_default: float = 0.5

    apply_post_opt: bool = True


@dataclass
class BCOConfig:
    """Конфигурация Bee Colony Optimization"""
  # Основные параметры BCO.
    num_bees: int = 10  # B в статье.
    num_cycles: int = 2  # NC в статье.
    num_passes: int = 5  # NP в статье.
    num_iterations: int = 2  # I в статье.

  # Параметры для Neural BCO.
    neural_bee_ratio: float = 0.5  # Доля neural bees от общего числа.
    type2_bee_ratio: float = 0.5  # Доля type-2 bees.

  # Параметры type-2 bees.
    type2_delete_prob: float = 0.2
    type2_add_prob: float = 0.8


@dataclass
class DataConfig:
    """Конфигурация данных и генерации синтетических городов"""
  # Параметры синтетических городов.
    num_nodes: int = 20
    num_training_cities: int = 32768  # 2^15 как в статье.
    train_val_split: float = 0.9

  # Географические параметры.
    city_area_km: float = 30.0  # 30km x 30km square.
    vehicle_speed_ms: float = 15.0  # 15 m/s скорость автобуса.

  # Параметры генерации графов.
    graph_types: List[str] = None
    edge_deletion_prob: float = 0.1  # Ρ в статье.
    voronoi_seed_multiplier: float = 1.2  # Для получения нужного числа узлов.

  # OD matrix параметры.
    demand_range: Tuple[int, int] = (60, 800)

  # Data augmentation.
    scale_range: Tuple[float, float] = (0.4, 1.6)
    demand_scale_range: Tuple[float, float] = (0.8, 1.2)
    rotation_range: Tuple[float, float] = (0.0, 360.0)

    def __post_init__(self):
        if self.graph_types is None:
            self.graph_types = [
                'incoming_4nn', 'outgoing_4nn', 'voronoi',
                '4_grid', '8_grid'
            ]


@dataclass
class EvaluationConfig:
    """Конфигурация для оценки и бенчмарков"""
  # Benchmark datasets.
    benchmark_datasets: List[str] = None

  # Evaluation metrics.
    service_radius_m: int = 400  # Радиус покрытия остановки.
    max_assignment_distance_m: int = 3000

  # Multiple runs для статистики.
    num_evaluation_seeds: int = 10

  # Comparison algorithms.
    compare_lp_variants: List[int] = None  # [100, 40000] rollouts.

  # НОВЫЕ ПАРАМЕТРЫ для ваших требований.
    enable_classical_comparison: bool = False  # Отключить сравнение с классическим BCO.
    focus_neural_only: bool = True  # Сосредоточиться только на Neural BCO.
    skip_benchmark_evaluation: bool = False  # Пропустить долгую оценку на бенчмарках.
    detailed_neural_analysis: bool = True  # Детальный анализ Neural BCO.

  # Настройки для работы с готовыми данными.
    use_existing_opt_stops: bool = True  # Использовать готовый opt_stops.pkl.
    require_opt_fin2_run: bool = False  # Не требовать запуск opt_fin2.py.

    def __post_init__(self):
        if self.benchmark_datasets is None:
            self.benchmark_datasets = ['mandl', 'mumford0', 'mumford1', 'mumford2', 'mumford3']
        if self.compare_lp_variants is None:
            self.compare_lp_variants = [100, 40000]


@dataclass
class FileConfig:
    """Конфигурация файлов и путей"""
  # Директории.
    data_dir: str = "data"
    models_dir: str = "models"
    results_dir: str = "results"
    logs_dir: str = "logs"
    figures_dir: str = "figures"

  # Имена файлов.
    model_checkpoint_name: str = "neural_planner_best.pth"
    training_log_name: str = "training_log.json"
    evaluation_results_name: str = "evaluation_results.json"

  # Benchmark data files.
    benchmark_data_dir: str = "benchmark_data"

  # Файлы данных из opt_fin2.
    opt_stops_file: str = "opt_stops.pkl"  # Основной файл с остановками.
    border_stops_csv: str = "border_stops_direction_1.csv"


@dataclass
class SystemConfig:
    """Системные настройки"""
  # Hardware.
    device: str = "cpu"  # Будет определено автоматически для Mac.
    num_workers: int = 4

  # Reproducibility.
    random_seed: int = 42

  # Улучшенное логирование.
    log_level: str = "INFO"
    detailed_logging: bool = True
    log_interval: int = 5  # Логирование каждые 5 batches (было 100).
    diagnostic_interval: int = 10  # Детальная диагностика каждые 10 batches.
    convergence_window: int = 20  # Окно для анализа сходимости.

  # Performance.
    enable_profiling: bool = False
    memory_efficient: bool = True

  # Режимы работы.
    fast_mode: bool = False  # Быстрый режим с уменьшенными параметрами.
    neural_only_mode: bool = True  # Режим работы только с Neural BCO.

  # Расширенное логирование.
    detailed_optimization_logging: bool = True
    log_iteration_interval: int = 5  # Логировать каждые 5 итераций.
    log_detailed_stats_interval: int = 50  # Детальная статистика каждые 50 итераций.
    log_neural_bee_stats: bool = True  # Логировать статистику Neural Bees.
    log_convergence_analysis: bool = True  # Анализ сходимости.
    log_performance_metrics: bool = True  # Метрики производительности.
  # Расширенное логирование.
    log_bee_individual_stats: bool = False  # Логировать каждую пчелу отдельно.
    log_memory_usage: bool = True  # Логировать использование памяти.
    log_graph_analysis: bool = True  # Детальный анализ графа.
    log_convergence_details: bool = True  # Детальный анализ сходимости.
    log_neural_network_internals: bool = False  # Внутренности нейросети.
    save_intermediate_solutions: bool = True  # Сохранять промежуточные решения.
    profile_performance: bool = False  # Профилирование производительности.



class Config:
    """Главный конфигурационный класс, объединяющий все настройки"""

    def __init__(self):
        self.network = NetworkConfig()
        self.mdp = MDPConfig()
        self.bco = BCOConfig()
        self.data = DataConfig()
        self.evaluation = EvaluationConfig()
        self.files = FileConfig()
        self.system = SystemConfig()

  # Определяем устройство автоматически для Mac.
        self._setup_device()

  # Настраиваем reproducibility.
        self._setup_reproducibility()

    def _setup_device(self):
        """Автоматическое определение устройства для Mac"""
        try:
            import torch

  # Проверяем доступность MPS (Metal Performance Shaders) на Mac.
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.system.device = "mps"
                print("Используется Metal Performance Shaders (MPS) на Mac")
            elif torch.cuda.is_available():
                self.system.device = "cuda"
                print(f"Используется CUDA GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.system.device = "cpu"
                print("Используется CPU")

        except ImportError:
            self.system.device = "cpu"
            print("PyTorch не найден, используется CPU")

    def _setup_reproducibility(self):
        """Настройка воспроизводимости результатов"""
        import random
        random.seed(self.system.random_seed)
        np.random.seed(self.system.random_seed)

        try:
            import torch
            torch.manual_seed(self.system.random_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.system.random_seed)
                torch.cuda.manual_seed_all(self.system.random_seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

  # Настройки для MPS на Mac.
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
  # MPS не требует специальных настроек для детерминизма.
                pass

        except ImportError:
            pass

    def get_cost_weights(self, alpha: float) -> Tuple[float, float]:
        """
        Получить веса для пассажирской и операторской стоимости

        Args:
            alpha: вес пассажирской стоимости [0, 1]

        Returns:
            (passenger_weight, operator_weight)
        """
        return alpha, (1.0 - alpha)

    def get_normalization_weights(self, max_travel_time: float) -> Tuple[float, float]:
        """
        Получить веса нормализации для функции стоимости

        Args:
            max_travel_time: максимальное время поездки в сети

        Returns:
            (wp, wo) - веса нормализации из статьи
        """
        wp = 1.0 / max_travel_time
        wo = 1.0 / (3 * self.mdp.num_routes * max_travel_time)
        return wp, wo

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Обновить конфигурацию из словаря"""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        print(f"Warning: Unknown config parameter {section_name}.{key}")
            else:
                print(f"Warning: Unknown config section {section_name}")

    def enable_fast_mode(self):
        """Включить быстрый режим для тестирования"""
        self.system.fast_mode = True
        self.data.num_training_cities = 1000  # Вместо 32768.
        self.network.num_epochs = 2  # Вместо 5.
        self.bco.num_iterations = 100  # Вместо 400.
        self.evaluation.num_evaluation_seeds = 3  # Вместо 10.
        print("Включен быстрый режим для тестирования")

    def enable_neural_only_mode(self):
        """Включить режим работы только с Neural BCO"""
        self.system.neural_only_mode = True
        self.evaluation.enable_classical_comparison = False
        self.evaluation.focus_neural_only = True
        self.evaluation.detailed_neural_analysis = True
        print("Включен режим работы только с Neural BCO")

    def setup_for_existing_data(self):
        """Настроить для работы с готовыми данными opt_stops.pkl"""
        self.evaluation.use_existing_opt_stops = True
        self.evaluation.require_opt_fin2_run = False
        print("Настроен режим работы с готовыми данными")

    def save_to_file(self, filepath: str):
        """Сохранить конфигурацию в JSON файл"""
        import json
        from dataclasses import asdict

        config_dict = {
            'network': asdict(self.network),
            'mdp': asdict(self.mdp),
            'bco': asdict(self.bco),
            'data': asdict(self.data),
            'evaluation': asdict(self.evaluation),
            'files': asdict(self.files),
            'system': asdict(self.system)
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

    def load_from_file(self, filepath: str):
        """Загрузить конфигурацию из JSON файла"""
        import json

        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)

        self.update_from_dict(config_dict)

    def print_summary(self):
        """Вывести краткую сводку конфигурации"""
        print("=" * 60)
        print("NEURAL BEE COLONY OPTIMIZATION - КОНФИГУРАЦИЯ")
        print("=" * 60)
        print(f"Устройство: {self.system.device}")
        print(f"Узлов в синтетических городах: {self.data.num_nodes}")
        print(f"Размер обучающей выборки: {self.data.num_training_cities}")
        print(f"Архитектура сети: GAT с {self.network.num_gat_layers} слоями")
        print(f"Размер скрытого слоя: {self.network.hidden_dim}")
        print(f"Число пчел в BCO: {self.bco.num_bees}")
        print(f"Число итераций BCO: {self.bco.num_iterations}")
        print(f"Ограничения маршрутов: {self.mdp.min_route_length}-{self.mdp.max_route_length} остановок")
        print(f"Число маршрутов: {self.mdp.num_routes}")
        print("=" * 60)


  # Глобальный экземпляр конфигурации.
config = Config()

  # Применяем настройки по умолчанию для ваших требований.
config.enable_neural_only_mode()
config.setup_for_existing_data()


  # Константы из статьи для валидации.
PAPER_CONSTANTS = {
    'MANDL_NODES': 15,
    'MANDL_EDGES': 20,
    'MANDL_ROUTES': 6,
    'MANDL_MIN': 2,
    'MANDL_MAX': 8,
    'MUMFORD0_NODES': 30,
    'MUMFORD1_NODES': 70,
    'MUMFORD2_NODES': 110,
    'MUMFORD3_NODES': 127,
    'TRAINING_EPOCHS': 5,
    'BATCH_SIZE': 64,
    'BCO_BEES': 10,
    'BCO_NC': 2,
    'BCO_NP': 5,
    'BCO_ITERATIONS': 400,
    'SYNTHETIC_CITIES': 32768,
    'TRAINING_NODES': 20
}


def validate_config_against_paper():
    """Проверить, что конфигурация соответствует параметрам из статьи"""
    errors = []

    if config.network.batch_size != PAPER_CONSTANTS['BATCH_SIZE']:
        errors.append(f"Batch size: {config.network.batch_size} != {PAPER_CONSTANTS['BATCH_SIZE']}")

    if config.network.num_epochs != PAPER_CONSTANTS['TRAINING_EPOCHS']:
        errors.append(f"Epochs: {config.network.num_epochs} != {PAPER_CONSTANTS['TRAINING_EPOCHS']}")

    if config.data.num_training_cities != PAPER_CONSTANTS['SYNTHETIC_CITIES']:
        errors.append(f"Training cities: {config.data.num_training_cities} != {PAPER_CONSTANTS['SYNTHETIC_CITIES']}")

    if config.data.num_nodes != PAPER_CONSTANTS['TRAINING_NODES']:
        errors.append(f"Training nodes: {config.data.num_nodes} != {PAPER_CONSTANTS['TRAINING_NODES']}")

    if config.bco.num_bees != PAPER_CONSTANTS['BCO_BEES']:
        errors.append(f"BCO bees: {config.bco.num_bees} != {PAPER_CONSTANTS['BCO_BEES']}")

    if config.bco.num_iterations != PAPER_CONSTANTS['BCO_ITERATIONS']:
        errors.append(f"BCO iterations: {config.bco.num_iterations} != {PAPER_CONSTANTS['BCO_ITERATIONS']}")

    if errors:
        print("ПРЕДУПРЕЖДЕНИЯ о соответствии статье:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("✓ Конфигурация соответствует параметрам из статьи")


if __name__ == "__main__":
  # Демонстрация использования.
    config.print_summary()
    validate_config_against_paper()

  # Пример сохранения и загрузки конфигурации.
    config.save_to_file("config_default.json")
    print("\nКонфигурация сохранена в config_default.json")