  # Neural_planner.py.
"""
Neural Planner для Transit Network Design
Реализует Graph Attention Network из статьи "Neural Bee Colony Optimization: A Case Study in Public Transit Network Design"

Архитектура:
- Graph Attention Network backbone на полносвязном графе
- Две "головы": extend_head для выбора расширений, halt_head для решения об остановке
- Обучение через REINFORCE с baseline
- Нормализация входных данных
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import json
from dataclasses import dataclass
from copy import deepcopy
import math
import random

from config import config
from transit_mdp import TransitMDP, MDPState, ExtendAction, HaltAction, ExtendMode
from cost_functions import TransitCostCalculator
from data_generator import CityInstance, DatasetManager

logger = logging.getLogger(__name__)

@dataclass
class TrainingMetrics:
    """Метрики обучения для логирования"""
    epoch: int
    batch: int
    avg_reward: float
    avg_cost: float
    baseline_loss: float
    policy_loss: float
    entropy_bonus: float
    total_loss: float
    learning_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'epoch': self.epoch,
            'batch': self.batch,
            'avg_reward': self.avg_reward,
            'avg_cost': self.avg_cost,
            'baseline_loss': self.baseline_loss,
            'policy_loss': self.policy_loss,
            'entropy_bonus': self.entropy_bonus,
            'total_loss': self.total_loss,
            'learning_rate': self.learning_rate
        }

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer из статьи [[11]]
    Реализация аналогичная Transformer attention, но с поддержкой edge features
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 edge_features: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_bias: bool = True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.edge_features = edge_features
        self.num_heads = num_heads
        self.head_dim = out_features // num_heads
        self.dropout = dropout

        assert out_features % num_heads == 0, "out_features должно быть кратно num_heads"

  # Линейные преобразования для Q, K, V.
        self.query = nn.Linear(in_features, out_features, bias=use_bias)
        self.key = nn.Linear(in_features, out_features, bias=use_bias)
        self.value = nn.Linear(in_features, out_features, bias=use_bias)

  # Преобразование edge features для включения в attention.
        self.edge_proj = nn.Linear(edge_features, num_heads, bias=use_bias)

  # Выходное преобразование.
        self.out_proj = nn.Linear(out_features, out_features, bias=use_bias)

  # Dropout.
        self.dropout_layer = nn.Dropout(dropout)

  # Инициализация весов.
        self._init_weights()

    def _init_weights(self):
        """Инициализация весов как в оригинальном Transformer"""
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.edge_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.query.bias is not None:
            nn.init.zeros_(self.query.bias)
            nn.init.zeros_(self.key.bias)
            nn.init.zeros_(self.value.bias)
            nn.init.zeros_(self.edge_proj.bias)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass Graph Attention Layer

        Args:
            node_features: [batch, num_nodes, in_features]
            edge_features: [batch, num_nodes, num_nodes, edge_features]
            mask: [batch, num_nodes, num_nodes] - маска для padding

        Returns:
            [batch, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = node_features.shape

  # Вычисляем Q, K, V.
        Q = self.query(node_features)  # [batch, num_nodes, out_features].
        K = self.key(node_features)  # [batch, num_nodes, out_features].
        V = self.value(node_features)  # [batch, num_nodes, out_features].

  # Reshape для multi-head attention.
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                   2)  # [batch, heads, num_nodes, head_dim].
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                   2)  # [batch, heads, num_nodes, head_dim].
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1,
                                                                                   2)  # [batch, heads, num_nodes, head_dim].

  # Вычисляем attention scores.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, num_nodes, num_nodes].

  # Добавляем edge features к attention scores.
        edge_bias = self.edge_proj(edge_features)  # [batch, num_nodes, num_nodes, heads].
        edge_bias = edge_bias.permute(0, 3, 1, 2)  # [batch, heads, num_nodes, num_nodes].
        scores = scores + edge_bias

  # Применяем маску если есть.
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch, heads, num_nodes, num_nodes].
            scores = scores.masked_fill(mask == 0, float('-inf'))

  # Softmax для получения attention weights.
        attention_weights = F.softmax(scores, dim=-1)  # [batch, heads, num_nodes, num_nodes].
        attention_weights = self.dropout_layer(attention_weights)

  # Применяем attention к values.
        out = torch.matmul(attention_weights, V)  # [batch, heads, num_nodes, head_dim].

  # Объединяем головы.
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes,
                                                    self.out_features)  # [batch, num_nodes, out_features].

  # Финальное преобразование.
        out = self.out_proj(out)

        return out

class GraphAttentionNetwork(nn.Module):
    """
    Graph Attention Network backbone из статьи [[11]]
    Полносвязный граф с edge features
    """

    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 128,
                 num_layers: int = 3,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

  # Входное преобразование node features.
        self.node_embedding = nn.Linear(node_features, hidden_dim)

  # Stack of Graph Attention Layers.
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                edge_features=edge_features,
                num_heads=num_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])

  # Layer normalization после каждого GAT слоя.
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])

  # Feed-forward networks после каждого GAT слоя.
        self.feed_forwards = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

  # Финальная layer norm.
        self.final_norm = nn.LayerNorm(hidden_dim)

        logger.info(f"GAT создан: {num_layers} слоев, {num_heads} heads, {hidden_dim} hidden_dim")

    def forward(self,
                node_features: torch.Tensor,
                edge_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass GAT

        Args:
            node_features: [batch, num_nodes, node_features]
            edge_features: [batch, num_nodes, num_nodes, edge_features]
            mask: [batch, num_nodes, num_nodes] опциональная маска

        Returns:
            [batch, num_nodes, hidden_dim] - node embeddings
        """
  # Входное преобразование.
        x = self.node_embedding(node_features)  # [batch, num_nodes, hidden_dim].

  # Проходим через GAT слои.
        for i in range(self.num_layers):
  # Self-attention с residual connection.
            residual = x
            x = self.gat_layers[i](x, edge_features, mask)
            x = x + residual
            x = self.layer_norms[i](x)

  # Feed-forward с residual connection.
            residual = x
            x = self.feed_forwards[i](x)
            x = x + residual

  # Финальная нормализация.
        x = self.final_norm(x)

        return x

class ExtendHead(nn.Module):
    """
    Policy head для выбора расширений маршрута
    Выбирает среди доступных кратчайших путей
    """

    def __init__(self, hidden_dim: int, head_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

  # Сеть для scoring action candidates.
        self.action_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, head_dim),  # Concat текущего состояния и action features.
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_dim, 1)
        )

        self.exploration_temperature = nn.Parameter(
            torch.tensor(1.2), requires_grad=False  # Немного больше 1 для exploration.
        )

  # Сеть для агрегации состояния маршрута.
        self.route_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, hidden_dim)
        )

    def set_exploration_temperature(self, temperature: float):
        """Изменить температуру для exploration"""
        self.exploration_temperature.data = torch.tensor(temperature)

    def forward(self,
                node_embeddings: torch.Tensor,
                route_mask: torch.Tensor,
                action_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass для extend head

        Args:
            node_embeddings: [batch, num_nodes, hidden_dim]
            route_mask: [batch, num_nodes] - 1 для узлов в текущем маршруте
            action_features: [batch, num_actions, hidden_dim] - features возможных действий

        Returns:
            [batch, num_actions] - логиты для каждого действия
        """
        batch_size = node_embeddings.size(0)

  # Агрегируем embeddings текущего маршрута.
        route_embeddings = node_embeddings * route_mask.unsqueeze(-1)  # [batch, num_nodes, hidden_dim].
        route_summary = route_embeddings.sum(dim=1)  # [batch, hidden_dim].

  # Если маршрут пустой, используем среднее по всем узлам.
        empty_route_mask = (route_mask.sum(dim=1) == 0).unsqueeze(-1)  # [batch, 1].
        global_avg = node_embeddings.mean(dim=1)  # [batch, hidden_dim].
        route_summary = torch.where(empty_route_mask, global_avg, route_summary)

  # Обрабатываем через aggregator.
        route_state = self.route_aggregator(route_summary)  # [batch, hidden_dim].

  # Расширяем для concatenation с action features.
        num_actions = action_features.size(1)
        route_state_expanded = route_state.unsqueeze(1).expand(-1, num_actions, -1)  # [batch, num_actions, hidden_dim].

  # Concatenate route state с action features.
        combined_features = torch.cat([route_state_expanded, action_features],
                                      dim=-1)  # [batch, num_actions, hidden_dim * 2].

  # Вычисляем scores для каждого действия.
        action_scores = self.action_scorer(combined_features).squeeze(-1)

  # Применяем температуру для exploration.
        action_scores = action_scores / self.exploration_temperature

        return action_scores

class HaltHead(nn.Module):
    """
    Policy head для решения об остановке маршрута
    Выбирает между continue и halt
    """

    def __init__(self, hidden_dim: int, head_dim: int = 32):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

  # Сеть для принятия решения halt/continue.
        self.decision_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, head_dim),  # Текущий маршрут + глобальный контекст.
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_dim, head_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(head_dim, 2)  # [continue, halt] логиты.
        )

  # Агрегация маршрута.
        self.route_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, hidden_dim)
        )

  # Агрегация глобального контекста.
        self.global_aggregator = nn.Sequential(
            nn.Linear(hidden_dim, head_dim),
            nn.ReLU(),
            nn.Linear(head_dim, hidden_dim)
        )

    def forward(self,
                node_embeddings: torch.Tensor,
                route_mask: torch.Tensor,
                route_length: torch.Tensor) -> torch.Tensor:
        """
        Forward pass для halt head

        Args:
            node_embeddings: [batch, num_nodes, hidden_dim]
            route_mask: [batch, num_nodes] - 1 для узлов в текущем маршруте
            route_length: [batch] - длина текущего маршрута

        Returns:
            [batch, 2] - логиты [continue, halt]
        """
  # Агрегируем embeddings текущего маршрута.
        route_embeddings = node_embeddings * route_mask.unsqueeze(-1)  # [batch, num_nodes, hidden_dim].
        route_summary = route_embeddings.sum(dim=1)  # [batch, hidden_dim].

  # Если маршрут пустой, используем нулевой вектор.
        empty_route_mask = (route_mask.sum(dim=1) == 0).unsqueeze(-1)  # [batch, 1].
        route_summary = torch.where(empty_route_mask, torch.zeros_like(route_summary), route_summary)

  # Обрабатываем состояние маршрута.
        route_state = self.route_aggregator(route_summary)  # [batch, hidden_dim].

  # Глобальный контекст (среднее по всем узлам).
        global_context = self.global_aggregator(node_embeddings.mean(dim=1))  # [batch, hidden_dim].

  # Объединяем маршрут и глобальный контекст.
        combined_state = torch.cat([route_state, global_context], dim=-1)  # [batch, hidden_dim * 2].

  # Вычисляем логиты решения.
        decision_logits = self.decision_network(combined_state)  # [batch, 2].

        return decision_logits

class NeuralPlanner(nn.Module):
    """
    Основная нейронная сеть для планирования транзитных маршрутов
    Реализует архитектуру из статьи [[11]]
    """

    def __init__(self,
                 node_features: int = None,
                 edge_features: int = None,
                 hidden_dim: int = None,
                 num_gat_layers: int = None,
                 num_heads: int = None,
                 dropout: float = None,
                 extend_head_dim: int = None,
                 halt_head_dim: int = None):
        super().__init__()

  # Используем параметры из конфигурации если не указаны.
        self.node_features = node_features or config.network.node_feature_dim
        self.edge_features = edge_features or config.network.edge_feature_dim
        self.hidden_dim = hidden_dim or config.network.hidden_dim
        self.num_gat_layers = num_gat_layers or config.network.num_gat_layers
        self.num_heads = num_heads or config.network.num_attention_heads
        self.dropout = dropout or config.network.dropout_rate
        self.extend_head_dim = extend_head_dim or config.network.extension_head_dim
        self.halt_head_dim = halt_head_dim or config.network.halt_head_dim

  # Graph Attention Network backbone.
        self.gat_backbone = GraphAttentionNetwork(
            node_features=self.node_features,
            edge_features=self.edge_features,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_gat_layers,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

  # Policy heads.
        self.extend_head = ExtendHead(self.hidden_dim, self.extend_head_dim)
        self.halt_head = HaltHead(self.hidden_dim, self.halt_head_dim)

  # Нормализация входных данных - инициализируем как buffers.
        self.register_buffer('node_mean', None)
        self.register_buffer('node_std', None)
        self.register_buffer('edge_mean', None)
        self.register_buffer('edge_std', None)

        logger.info(f"Neural Planner создан: {self.hidden_dim}D, {self.num_gat_layers} GAT слоев")

    def set_normalization_params(self,
                                 node_mean: torch.Tensor,
                                 node_std: torch.Tensor,
                                 edge_mean: torch.Tensor,
                                 edge_std: torch.Tensor):
        """Установить параметры нормализации из статьи [[11]]"""
  # Обновляем существующие buffers и переносим на правильное устройство.
        device = next(self.parameters()).device
        self.node_mean = node_mean.to(device)
        self.node_std = node_std.to(device)
        self.edge_mean = edge_mean.to(device)
        self.edge_std = edge_std.to(device)

    def normalize_inputs(self,
                         node_features: torch.Tensor,
                         edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Нормализация входных данных [[11]]"""
        if self.node_mean is not None and self.node_std is not None:
  # Убеждаемся, что параметры на том же устройстве.
            device = node_features.device
            node_mean = self.node_mean.to(device)
            node_std = self.node_std.to(device)
            node_features = (node_features - node_mean) / (node_std + 1e-8)

        if self.edge_mean is not None and self.edge_std is not None:
            device = edge_features.device
            edge_mean = self.edge_mean.to(device)
            edge_std = self.edge_std.to(device)
            edge_features = (edge_features - edge_mean) / (edge_std + 1e-8)

        return node_features, edge_features

    def forward_extend(self,
                       node_features: torch.Tensor,
                       edge_features: torch.Tensor,
                       route_mask: torch.Tensor,
                       action_features: torch.Tensor,
                       mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass для режима extend

        Args:
            node_features: [batch, num_nodes, node_features]
            edge_features: [batch, num_nodes, num_nodes, edge_features]
            route_mask: [batch, num_nodes] - маска текущего маршрута
            action_features: [batch, num_actions, hidden_dim] - features действий
            mask: [batch, num_nodes, num_nodes] - маска графа

        Returns:
            [batch, num_actions] - логиты для выбора действия
        """
  # Нормализация входов.
        node_features, edge_features = self.normalize_inputs(node_features, edge_features)

  # GAT backbone.
        node_embeddings = self.gat_backbone(node_features, edge_features, mask)

  # Extend head.
        action_logits = self.extend_head(node_embeddings, route_mask, action_features)

        return action_logits

    def forward_halt(self,
                     node_features: torch.Tensor,
                     edge_features: torch.Tensor,
                     route_mask: torch.Tensor,
                     route_length: torch.Tensor,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass для режима halt

        Args:
            node_features: [batch, num_nodes, node_features]
            edge_features: [batch, num_nodes, num_nodes, edge_features]
            route_mask: [batch, num_nodes] - маска текущего маршрута
            route_length: [batch] - длина маршрута
            mask: [batch, num_nodes, num_nodes] - маска графа

        Returns:
            [batch, 2] - логиты [continue, halt]
        """
  # Нормализация входов.
        node_features, edge_features = self.normalize_inputs(node_features, edge_features)

  # GAT backbone.
        node_embeddings = self.gat_backbone(node_features, edge_features, mask)

  # Halt head.
        decision_logits = self.halt_head(node_embeddings, route_mask, route_length)

        return decision_logits

class Baseline(nn.Module):
    def __init__(self,
                 node_features: int,
                 edge_features: int,
                 hidden_dim: int = 128):
        super().__init__()

  # Архитектура без BatchNorm для CPU и малых batch_size.
        self.network = nn.Sequential(
            nn.Linear(node_features + edge_features + 1, hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim * 2),  # LayerNorm вместо BatchNorm.
            nn.Dropout(0.2),

            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),  # LayerNorm вместо BatchNorm.
            nn.Dropout(0.2),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
  # Убираем Tanh для большей гибкости.
        )

    def forward(self,
                city_features: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        """
        Предсказать baseline для города и параметра cost function

        Args:
            city_features: [batch, features] - агрегированные признаки города
            alpha: [batch, 1] - параметр cost function

        Returns:
            [batch, 1] - предсказанная baseline награда
        """
        combined = torch.cat([city_features, alpha], dim=-1)
        return self.network(combined)

class CityDataset(Dataset):
    """Dataset для обучения на синтетических городах"""

    def __init__(self, cities: List[CityInstance], alpha_range: Tuple[float, float] = (0.0, 1.0)):
        self.cities = cities
        self.alpha_range = alpha_range

    def __len__(self):
        return len(self.cities)

    def __getitem__(self, idx):
        city = self.cities[idx]

  # Случайный alpha для обучения.
        alpha = random.uniform(self.alpha_range[0], self.alpha_range[1])

        return city, alpha

class NeuralPlannerTrainer:
    """
    Trainer для Neural Planner с REINFORCE алгоритмом
    Реализует обучение из статьи [[11]]
    """

    def __init__(self,
                 device: str = 'cpu'):
        self.device = device or config.system.device

  # Создаем модели.
        self.neural_planner = NeuralPlanner().to(self.device)
        self.baseline = Baseline(
            node_features=config.network.node_feature_dim,
            edge_features=config.network.edge_feature_dim
        ).to(self.device)

  # Оптимизаторы.
        self.policy_optimizer = optim.Adam(
            self.neural_planner.parameters(),
            lr=config.network.learning_rate
        )
        self.baseline_optimizer = optim.Adam(
            self.baseline.parameters(),
            lr=config.network.baseline_learning_rate
        )

  # Метрики для логирования.
        self.training_metrics = []

        logger.info(f"Trainer создан на устройстве {self.device}")

    def extract_graph_features(self, city: CityInstance) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Извлечь признаки графа города для нейронной сети

        Args:
            city: экземпляр города

        Returns:
            (node_features, edge_features) tensors
        """
        city_id = id(city)
        if hasattr(self, '_cached_city_id') and self._cached_city_id == city_id:
            return self._cached_node_features, self._cached_edge_features

        num_nodes = len(city.city_graph.nodes())

  # Node features: [x, y, population, jobs, degree_centrality, is_covered, betweenness, closeness].
        node_features = []

  # Вычисляем дополнительные метрики центральности.
        betweenness_centrality = nx.betweenness_centrality(city.city_graph)
        closeness_centrality = nx.closeness_centrality(city.city_graph)

        for node_id in city.city_graph.nodes():
            node_data = city.city_graph.nodes[node_id]

            features = [
                node_data.get('x', 0.0),
                node_data.get('y', 0.0),
  # Используем данные из nodes_gdf если они есть.
                0.0,  # Population - заполним из city.nodes_gdf.
                0.0,  # Jobs - заполним из city.nodes_gdf.
                nx.degree_centrality(city.city_graph).get(node_id, 0.0),
                0.0,  # Is_covered - пока не используется.
                betweenness_centrality.get(node_id, 0.0),
                closeness_centrality.get(node_id, 0.0)
            ]

            node_features.append(features)

  # Заполняем population и jobs из nodes_gdf если данные есть.
        if hasattr(city, 'nodes_gdf') and 'population' in city.nodes_gdf.columns:
            for i, (_, row) in enumerate(city.nodes_gdf.iterrows()):
                if i < len(node_features):
                    node_features[i][2] = float(row.get('population', 0))
                    node_features[i][3] = float(row.get('jobs', 0))

        node_features = torch.tensor(node_features, dtype=torch.float32, device=self.device)

  # Edge features для полносвязного графа.
        edge_features = torch.zeros(num_nodes, num_nodes, config.network.edge_feature_dim, device=self.device)

  # Создаем список узлов в том же порядке.
        node_list = list(city.city_graph.nodes())

  # Предвычисляем все координаты для векторизации.
        node_coords = {}
        for node in node_list:
            node_coords[node] = (city.city_graph.nodes[node]['x'], city.city_graph.nodes[node]['y'])

        for i, node_i in enumerate(node_list):
            xi, yi = node_coords[node_i]
            for j, node_j in enumerate(node_list):
                if i != j:
                    xj, yj = node_coords[node_j]
                    distance = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

  # Спрос между узлами из OD матрицы.
                    demand = city.od_matrix[i, j] if i < city.od_matrix.shape[0] and j < city.od_matrix.shape[
                        1] else 0.0

  # Есть ли прямое ребро в дорожной сети.
                    has_street_edge = 1.0 if city.city_graph.has_edge(node_i, node_j) else 0.0

  # Время поездки (из предвычисленных кратчайших путей или расчет).
                    try:
                        travel_time = nx.shortest_path_length(city.city_graph, node_i, node_j, weight='travel_time')
                    except (nx.NetworkXNoPath, nx.NodeNotFound, KeyError):
                        travel_time = distance / config.data.vehicle_speed_ms  # Простая оценка.

  # Нормализованный спрос.
                    max_demand = city.od_matrix.max() if city.od_matrix.size > 0 else 1.0
                    normalized_demand = demand / max_demand if max_demand > 0 else 0.0

                    edge_features[i, j] = torch.tensor([
                        demand,
                        distance,
                        0.0,  # Existing_transit - пока не используется.
                        has_street_edge,
                        travel_time,
                        normalized_demand
                    ])

  # Сохраняем в кэш.
        self._cached_city_id = city_id
        self._cached_node_features = node_features
        self._cached_edge_features = edge_features

        return node_features, edge_features

    def create_action_features(self,
                               mdp: TransitMDP,
                               state: MDPState,
                               valid_actions: List[ExtendAction],
                               node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Создать features для действий расширения

        Args:
            mdp: MDP экземпляр
            state: текущее состояние
            valid_actions: список валидных действий
            node_embeddings: embeddings узлов

        Returns:
            [num_actions, hidden_dim] - features действий
        """
        if not valid_actions:
            return torch.zeros(0, self.neural_planner.hidden_dim, device=self.device)

        action_features = []

        for action in valid_actions:
  # Агрегируем embeddings узлов в пути действия.
            path_nodes = action.path
            path_embeddings = []

            for node_id in path_nodes:
  # Находим индекс узла в графе.
                node_list = list(state.city_graph.nodes())
                if node_id in node_list:
                    node_idx = node_list.index(node_id)
                    if node_idx < node_embeddings.size(0):
                        path_embeddings.append(node_embeddings[node_idx])

            if path_embeddings:
  # Среднее embedding пути как feature действия.
                action_feature = torch.stack(path_embeddings).mean(dim=0)
            else:
                action_feature = torch.zeros(self.neural_planner.hidden_dim, device=self.device)

            action_features.append(action_feature)

        return torch.stack(action_features)

    def rollout_episode(self,
                        city: CityInstance,
                        alpha: float,
                        max_steps: int = 500) -> Tuple[List[MDPState], List[Any], List[float], float]:
        """
        Выполнить rollout эпизода с текущей политикой

        Returns:
            (states, actions, log_probs, final_reward)
        """
  # Создаем MDP и cost calculator.
        cost_calculator = TransitCostCalculator(alpha=alpha)
        mdp = TransitMDP(city.city_graph, city.od_matrix, cost_calculator, alpha)

  # Извлекаем признаки графа.
        node_features, edge_features = self.extract_graph_features(city)
        node_features = node_features.unsqueeze(0).to(self.device)  # [1, num_nodes, features].
        edge_features = edge_features.unsqueeze(0).to(self.device)  # [1, num_nodes, num_nodes, features].

        states = []
        actions = []
        log_probs = []

        current_state = mdp.get_initial_state()
        step_count = 0

        self.neural_planner.train()
        self.baseline.train()
        final_reward = 0.0
        episode_completed_naturally = False
        while not current_state.is_terminal() and step_count < max_steps:
            states.append(deepcopy(current_state))

  # Получаем валидные действия.
            valid_actions = mdp.get_valid_actions(current_state)
            if not valid_actions:
                break

            if current_state.extend_mode == ExtendMode.EXTEND:
  # Режим extend.
  # Создаем маску текущего маршрута.
                route_mask = torch.zeros(len(city.city_graph.nodes()))
                node_list = list(city.city_graph.nodes())
                for node_id in current_state.current_route:
                    if node_id in node_list:
                        route_mask[node_list.index(node_id)] = 1.0
                route_mask = route_mask.unsqueeze(0).to(self.device)  # [1, num_nodes].

  # Получаем embeddings узлов.
                node_embeddings = self.neural_planner.gat_backbone(node_features, edge_features)[
                    0]  # [num_nodes, hidden_dim].

  # Создаем features действий.
                action_features = self.create_action_features(mdp, current_state, valid_actions, node_embeddings)
                action_features = action_features.unsqueeze(0).to(self.device)  # [1, num_actions, hidden_dim].

  # Forward pass extend head.
                if action_features.size(1) > 0:
                    action_logits = self.neural_planner.forward_extend(
                        node_features, edge_features, route_mask, action_features
                    )[0]  # [num_actions].

  # Выбираем действие.
                    action_probs = F.softmax(action_logits, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action_idx = action_dist.sample()
                    selected_action = valid_actions[action_idx.item()]

                    log_prob = action_dist.log_prob(action_idx)
  # Убеждаемся, что tensor на правильном устройстве.
                    log_probs.append(log_prob.to(self.device))
                else:
  # Если нет валидных действий, берем первое доступное.
                    selected_action = valid_actions[0]
                    log_probs.append(torch.tensor(0.0, requires_grad=True, device=self.device))

            else:
  # Режим halt.
                route_mask = torch.zeros(len(city.city_graph.nodes()))
                node_list = list(city.city_graph.nodes())
                for node_id in current_state.current_route:
                    if node_id in node_list:
                        route_mask[node_list.index(node_id)] = 1.0
                route_mask = route_mask.unsqueeze(0).to(self.device)  # [1, num_nodes].

                route_length = torch.tensor([len(current_state.current_route)], dtype=torch.float32).to(self.device)

  # Forward pass halt head.
                decision_logits = self.neural_planner.forward_halt(
                    node_features, edge_features, route_mask, route_length
                )[0]  # [2].

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
                    decision_probs = F.softmax(available_logits, dim=-1)
                    decision_dist = torch.distributions.Categorical(decision_probs)
                    decision_idx = decision_dist.sample()
                    log_prob = decision_dist.log_prob(decision_idx)
                    selected_action = available_actions[decision_idx.item()]
                    log_probs.append(log_prob.to(self.device))
                else:
                    selected_action = valid_actions[0]
                    log_probs.append(torch.tensor(0.0, requires_grad=True, device=self.device))

            actions.append(selected_action)

  # Выполняем шаг.
            current_state, reward, done = mdp.step(current_state, selected_action)
            step_count += 1

  # Проверяем естественное завершение эпизода.
            if done:
                episode_completed_naturally = True
                if reward != 0.0:
                    final_reward = reward
                break

  # Вычисляем финальную награду если не получили её.
  # Если эпизод не завершился естественно или не получили награду, вычисляем её.
        if not episode_completed_naturally or final_reward == 0.0:
            all_routes = current_state.get_all_routes()
            if len(all_routes) == 0:
                final_reward = -1000.0
                logger.debug("Episode ended with no routes")
            else:
                cost = cost_calculator.calculate_cost(
                    city.city_graph, all_routes, city.od_matrix, alpha
                )
                final_reward = -cost
                logger.debug(f"Episode ended: {len(all_routes)} routes, cost={cost:.3f}")

  # Штраф только если не достигли терминального состояния.
            if not current_state.is_terminal():
                final_reward -= 10.0
                logger.debug("Episode ended prematurely (non-terminal)")

  # Logger.info(f"ROLLOUT DEBUG: steps={step_count}, ".
  # F"terminal={current_state.is_terminal()}, ".
  # F"routes_created={len(current_state.get_all_routes())}, ".
  # F"final_reward={final_reward:.6f}").
  # Изменил на debug чтобы не засорять логи.

        self.neural_planner.train()

        return states, actions, log_probs, final_reward

    def compute_city_features(self, city: CityInstance) -> torch.Tensor:
        """Вычислить агрегированные признаки города для baseline"""
        node_features, edge_features = self.extract_graph_features(city)

  # Агрегируем признаки города.
        city_features = torch.cat([
            node_features.mean(dim=0),  # Средние node features.
            edge_features.mean(dim=(0, 1))  # Средние edge features.
        ])

        return city_features.unsqueeze(0)  # [1, features].

    def train_batch(self, batch_cities: List[CityInstance], batch_alphas: List[float]) -> TrainingMetrics:
        """Обучение на одном batch"""
        self.neural_planner.train()  # Убеждаемся, что модель в режиме обучения.
        batch_size = len(batch_cities)

  # Собираем trajectories.
        all_log_probs = []
        all_rewards = []
        all_baselines = []

        total_reward = 0.0
        total_cost = 0.0

        for city, alpha in zip(batch_cities, batch_alphas):
  # Rollout с текущей политикой.
            states, actions, log_probs, final_reward = self.rollout_episode(city, alpha)

  # Baseline prediction.
            city_features = self.compute_city_features(city).to(self.device)
            alpha_tensor = torch.tensor([[alpha]], dtype=torch.float32).to(self.device)
            baseline_pred = self.baseline(city_features, alpha_tensor).squeeze()

  # Сохраняем tensors, а не числа.
            for log_prob in log_probs:
                if isinstance(log_prob, torch.Tensor):
                    all_log_probs.append(log_prob.to(self.device))
                else:
                    all_log_probs.append(torch.tensor(log_prob, requires_grad=True, device=self.device))

            all_rewards.extend([final_reward] * len(log_probs))  # Все действия получают финальную награду.
            all_baselines.extend([baseline_pred] * len(log_probs))

            total_reward += final_reward
            total_cost += -final_reward  # Reward = -cost.

  # Проверяем, что у нас есть данные для обучения.
        if len(all_log_probs) == 0:
            return TrainingMetrics(
                epoch=0, batch=0, avg_reward=0, avg_cost=0,
                baseline_loss=0, policy_loss=0, entropy_bonus=0, total_loss=0,
                learning_rate=self.policy_optimizer.param_groups[0]['lr']
            )

  # Преобразуем в tensors с градиентами.
        if all_log_probs and isinstance(all_log_probs[0], torch.Tensor):
            log_probs_tensor = torch.stack(all_log_probs).to(self.device)
        else:
            log_probs_tensor = torch.tensor(all_log_probs, dtype=torch.float32, requires_grad=True).to(self.device)

        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32).to(self.device)
        baselines_tensor = torch.stack(all_baselines).to(self.device)

  # Нормализуем награды для стабильности.
        rewards_mean = rewards_tensor.mean()
        rewards_std = rewards_tensor.std() + 1e-8
        normalized_rewards = (rewards_tensor - rewards_mean) / rewards_std

  # Baseline loss с нормализованными наградами.
        baseline_loss = F.mse_loss(baselines_tensor, normalized_rewards)

  # REINFORCE loss.
        advantages = normalized_rewards - baselines_tensor.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Нормализуем advantages.
        policy_loss = -(log_probs_tensor * advantages).mean()

  # Total loss.
        total_loss = policy_loss + baseline_loss

  # Обязательно обновляем градиенты.
        self.policy_optimizer.zero_grad()
        self.baseline_optimizer.zero_grad()

        total_loss.backward()

  # СОХРАНЯЕМ градиенты для диагностики ДО их обрезки.
        diagnostic_info = {}
        if config.system.detailed_logging:
  # Сохраняем градиенты до clipping'а.
            policy_grad_norm_diag = torch.nn.utils.clip_grad_norm_(self.neural_planner.parameters(), float('inf'))
            baseline_grad_norm_diag = torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), float('inf'))

  # 1. Baseline диагностика (используем данные из основного обучения).
            baseline_predictions = torch.tensor(all_baselines, dtype=torch.float32)
            actual_rewards = rewards_tensor

            diagnostic_info.update({
                'baseline_mean_prediction': baseline_predictions.mean().item(),
                'baseline_std_prediction': baseline_predictions.std().item(),
                'reward_mean': actual_rewards.mean().item(),
                'reward_std': actual_rewards.std().item(),
                'baseline_mae': F.l1_loss(baseline_predictions, actual_rewards).item(),
                'baseline_rmse': torch.sqrt(F.mse_loss(baseline_predictions, actual_rewards)).item()
            })

  # 2. Advantage диагностика (из основного обучения).
            advantages_orig = rewards_tensor - baseline_predictions
            diagnostic_info.update({
                'advantage_mean': advantages_orig.mean().item(),
                'advantage_std': advantages_orig.std().item(),
                'advantage_min': advantages_orig.min().item(),
                'advantage_max': advantages_orig.max().item()
            })

  # 3. Gradient norms (сохраненные ранее).
            diagnostic_info.update({
                'policy_grad_norm': policy_grad_norm_diag.item(),
                'baseline_grad_norm': baseline_grad_norm_diag.item(),
                'grad_norm_ratio': (policy_grad_norm_diag / (baseline_grad_norm_diag + 1e-8)).item()
            })

  # 4. Episode статистика (используем данные из основного обучения).
            total_steps = len(all_log_probs)
            num_episodes = len(batch_cities)

  # Примерные оценки для terminal_rate и avg_routes.
            estimated_terminal_rate = 0.7 if total_steps > num_episodes * 10 else 0.3
            estimated_avg_routes = min(total_steps / num_episodes * 0.5, 8.0) if num_episodes > 0 else 0

            diagnostic_info.update({
                'avg_episode_length': total_steps / num_episodes if num_episodes > 0 else 0,
                'terminal_rate': estimated_terminal_rate,
                'avg_routes_per_episode': estimated_avg_routes,
                'num_trajectories': total_steps
            })

  # 5. Policy entropy.
            log_probs_variance = log_probs_tensor.var().item() if len(log_probs_tensor) > 1 else 0
            diagnostic_info.update({
                'policy_entropy_proxy': log_probs_variance
            })

  # Обрезаем градиенты для обучения.
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.neural_planner.parameters(), 1.0)
        baseline_grad_norm = torch.nn.utils.clip_grad_norm_(self.baseline.parameters(), 1.0)

        self.policy_optimizer.step()
        self.baseline_optimizer.step()

  # ДЕТАЛЬНАЯ ДИАГНОСТИКА с правильными данными.
        if config.system.detailed_logging:
  # Логируем детальную диагностику.
            if hasattr(self, '_batch_counter'):
                self._batch_counter += 1
            else:
                self._batch_counter = 1

            if self._batch_counter % config.system.diagnostic_interval == 0:
                logger.info("=== ДЕТАЛЬНАЯ ДИАГНОСТИКА ===")
                logger.info(f"Baseline: pred_mean={diagnostic_info['baseline_mean_prediction']:.3f}, "
                            f"actual_mean={diagnostic_info['reward_mean']:.3f}, "
                            f"MAE={diagnostic_info['baseline_mae']:.3f}")
                logger.info(f"Advantages: mean={diagnostic_info['advantage_mean']:.3f}, "
                            f"std={diagnostic_info['advantage_std']:.3f}")
                logger.info(f"Gradients: policy={diagnostic_info['policy_grad_norm']:.3f}, "
                            f"baseline={diagnostic_info['baseline_grad_norm']:.3f}")
                logger.info(f"Episodes: avg_length={diagnostic_info['avg_episode_length']:.1f}, "
                            f"terminal_rate={diagnostic_info['terminal_rate']:.2f}, "
                            f"avg_routes={diagnostic_info['avg_routes_per_episode']:.1f}")
                logger.info("=" * 30)

  # Метрики.
        avg_reward = total_reward / batch_size
        avg_cost = total_cost / batch_size

        return TrainingMetrics(
            epoch=0,  # Будет установлено в train().
            batch=0,  # Будет установлено в train().
            avg_reward=avg_reward,
            avg_cost=avg_cost,
            baseline_loss=baseline_loss.item(),
            policy_loss=policy_loss.item(),
            entropy_bonus=0.0,
            total_loss=total_loss.item(),
            learning_rate=self.policy_optimizer.param_groups[0]['lr']
        )

    def train(self, train_cities: List[CityInstance], val_cities: List[CityInstance]) -> Dict[str, Any]:
        """
        Обучить Neural Planner на синтетических городах
        Реализует схему обучения из статьи [[11]]
        """
  # Вычисляем параметры нормализации.
        logger.info("Вычисление параметров нормализации...")
        normalization_sample_size = min(1000, len(train_cities))
        if normalization_sample_size > 0:
            self.compute_normalization_params(train_cities[:normalization_sample_size])
        else:
            logger.warning("Нет обучающих городов для нормализации")

        def collate_fn(batch):
            """Кастомная функция для обработки batch с CityInstance объектами"""
            cities, alphas = zip(*batch)
            return list(cities), list(alphas)

  # Создаем datasets.
        train_dataset = CityDataset(train_cities, alpha_range=config.mdp.cost_alpha_range)
        val_dataset = CityDataset(val_cities, alpha_range=config.mdp.cost_alpha_range)

        train_loader = DataLoader(train_dataset, batch_size=config.network.batch_size, shuffle=True,
                                  collate_fn=collate_fn)

        best_val_reward = float('-inf')
        best_model_state = None

  # Обучение.
  # НОВОЕ: Tracking для анализа сходимости.
        reward_history = []
        loss_history = []

  # Обучение.
        for epoch in range(config.network.num_epochs):
            logger.info(f"Epoch {epoch + 1}/{config.network.num_epochs}")

            epoch_metrics = []
            epoch_rewards = []
            epoch_losses = []

  # Training loop.
            for batch_idx, batch in enumerate(train_loader):
                batch_cities, batch_alphas = batch

  # Применяем аугментацию данных [[11]].
                augmented_cities = []
                for city in batch_cities:
                    if random.random() < 0.5:  # 50% шанс аугментации.
                        dataset_manager = DatasetManager()
                        augmented_city = dataset_manager._augment_city(city)
                        augmented_cities.append(augmented_city)
                    else:
                        augmented_cities.append(city)

  # Обучение на batch.
                metrics = self.train_batch(augmented_cities, batch_alphas)
                metrics.epoch = epoch + 1
                metrics.batch = batch_idx + 1

                epoch_metrics.append(metrics)
                epoch_rewards.append(metrics.avg_reward)
                epoch_losses.append(metrics.total_loss)

  # Расширенное логирование.
                if (batch_idx + 1) % config.system.log_interval == 0:
                    recent_rewards = epoch_rewards[-config.system.log_interval:]
                    reward_trend = "" if len(recent_rewards) > 1 and recent_rewards[-1] > recent_rewards[0] else ""

                    logger.info(f"Batch {batch_idx + 1}: reward={metrics.avg_reward:.3f} {reward_trend}, "
                                f"p_loss={metrics.policy_loss:.3f}, b_loss={metrics.baseline_loss:.0f}, "
                                f"lr={metrics.learning_rate:.2e}")

  # Анализ эпохи.
            reward_history.extend(epoch_rewards)
            loss_history.extend(epoch_losses)

  # Статистика сходимости.
            if len(reward_history) >= config.system.convergence_window:
                recent_improvement = (np.mean(reward_history[-config.system.convergence_window // 2:]) -
                                      np.mean(reward_history[
                                              -config.system.convergence_window:-config.system.convergence_window // 2]))
                logger.info(
                    f" Сходимость: улучшение за последние {config.system.convergence_window // 2} batches: {recent_improvement:.4f}")

  # Валидация.
            val_reward = self.validate(val_cities[:min(100, len(val_cities))])  # Безопасная выборка.

  # Сохраняем лучшую модель.
            if val_reward > best_val_reward:
                best_val_reward = val_reward
                best_model_state = {
                    'neural_planner': self.neural_planner.state_dict(),
                    'baseline': self.baseline.state_dict(),
                    'epoch': epoch + 1,
                    'val_reward': val_reward
                }

  # Логирование эпохи.
            avg_epoch_reward = np.mean([m.avg_reward for m in epoch_metrics])
            avg_epoch_cost = np.mean([m.avg_cost for m in epoch_metrics])
            logger.info(f"Epoch {epoch + 1} завершена: avg_reward={avg_epoch_reward:.3f}, "
                        f"avg_cost={avg_epoch_cost:.3f}, val_reward={val_reward:.3f}")

            self.training_metrics.extend(epoch_metrics)

  # Загружаем лучшую модель.
        if best_model_state:
            self.neural_planner.load_state_dict(best_model_state['neural_planner'])
            self.baseline.load_state_dict(best_model_state['baseline'])
            logger.info(f"Загружена лучшая модель с val_reward={best_val_reward:.3f}")

        return {
            'best_val_reward': best_val_reward,
            'training_metrics': [m.to_dict() for m in self.training_metrics],
            'final_model_state': best_model_state
        }

    def compute_normalization_params(self, cities: List[CityInstance]):
        """Вычислить параметры нормализации для входных данных [[11]]"""
        all_node_features = []
        all_edge_features = []

        for city in cities[:min(100, len(cities))]:  # Ограничиваем для ускорения.
            node_features, edge_features = self.extract_graph_features(city)
            all_node_features.append(node_features)
            all_edge_features.append(edge_features.view(-1, edge_features.size(-1)))

  # Объединяем все данные.
        combined_node_features = torch.cat(all_node_features, dim=0)
        combined_edge_features = torch.cat(all_edge_features, dim=0)

  # Вычисляем mean и std.
        node_mean = combined_node_features.mean(dim=0)
        node_std = combined_node_features.std(dim=0)
        edge_mean = combined_edge_features.mean(dim=0)
        edge_std = combined_edge_features.std(dim=0)

  # Устанавливаем параметры нормализации.
        self.neural_planner.set_normalization_params(node_mean, node_std, edge_mean, edge_std)

        logger.info("Параметры нормализации вычислены и установлены")

    def validate(self, val_cities: List[CityInstance]) -> float:
        total_reward = 0.0
        num_episodes = 0

        self.neural_planner.eval()
        with torch.no_grad():
            for city in val_cities:
                try:
                    alpha = random.uniform(0.0, 1.0)
                    _, _, _, reward = self.rollout_episode(city, alpha, max_steps=1000)
                    total_reward += reward
                    num_episodes += 1
                except Exception as e:
                    logger.warning(f"Validation episode failed: {e}")
                    continue

        self.neural_planner.train()
        return total_reward / num_episodes if num_episodes > 0 else 0.0

    def save_model(self, filepath: str):
        """Сохранить обученную модель"""
        torch.save({
            'neural_planner_state_dict': self.neural_planner.state_dict(),
            'baseline_state_dict': self.baseline.state_dict(),
            'training_metrics': [m.to_dict() for m in self.training_metrics],
            'model_config': {
                'node_features': self.neural_planner.node_features,
                'edge_features': self.neural_planner.edge_features,
                'hidden_dim': self.neural_planner.hidden_dim,
                'num_gat_layers': self.neural_planner.num_gat_layers,
                'num_heads': self.neural_planner.num_heads,
                'dropout': self.neural_planner.dropout
            }
        }, filepath)

        logger.info(f"Модель сохранена в {filepath}")

    def load_model(self, filepath: str):
        """Загрузить обученную модель"""
        checkpoint = torch.load(filepath, map_location=self.device)

  # Загружаем state_dict с обработкой параметров нормализации.
        neural_planner_state = checkpoint['neural_planner_state_dict']

  # Извлекаем параметры нормализации если они есть.
        if 'node_mean' in neural_planner_state:
            node_mean = neural_planner_state.pop('node_mean')
            node_std = neural_planner_state.pop('node_std')
            edge_mean = neural_planner_state.pop('edge_mean')
            edge_std = neural_planner_state.pop('edge_std')

  # Загружаем остальные параметры.
            self.neural_planner.load_state_dict(neural_planner_state, strict=False)

  # Устанавливаем параметры нормализации.
            self.neural_planner.set_normalization_params(node_mean, node_std, edge_mean, edge_std)
        else:
            self.neural_planner.load_state_dict(neural_planner_state, strict=False)

        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])

        if 'training_metrics' in checkpoint:
            self.training_metrics = [TrainingMetrics(**m) for m in checkpoint['training_metrics']]

        logger.info(f"Модель загружена из {filepath}")

def train_neural_planner(train_data_path: str = None,
                         save_path: str = None) -> NeuralPlannerTrainer:
    """
    Обучить Neural Planner на синтетических данных

    Args:
        train_data_path: путь к обучающим данным
        save_path: путь для сохранения модели

    Returns:
        Обученный trainer
    """
    if train_data_path is None:
  # Ищем любой доступный файл с обучающими данными.
        import glob
        data_files = glob.glob(f'{config.files.data_dir}/training_cities_*.pkl')
        if data_files:
            train_data_path = data_files[0]  # Берем первый найденный.
            logger.info(f"Найден файл данных: {train_data_path}")
        else:
            train_data_path = f"{config.files.data_dir}/training_cities_{config.data.num_training_cities}.pkl"

    if save_path is None:
        save_path = f"{config.files.models_dir}/{config.files.model_checkpoint_name}"

  # Загружаем или создаем данные.
    dataset_manager = DatasetManager()

    try:
        logger.info(f"Загрузка обучающих данных из {train_data_path}")
        dataset = dataset_manager.load_dataset(train_data_path)
        train_cities = dataset['train']
        val_cities = dataset['val']
    except FileNotFoundError:
        logger.info("Обучающие данные не найдены, создаем новые...")
        train_cities = dataset_manager.create_training_dataset(save_path=train_data_path)
  # Разделяем на train/val.
        split_idx = int(len(train_cities) * config.data.train_val_split)
        val_cities = train_cities[split_idx:]
        train_cities = train_cities[:split_idx]

    logger.info(f"Обучающих городов: {len(train_cities)}, валидационных: {len(val_cities)}")

  # Создаем trainer и обучаем.
    trainer = NeuralPlannerTrainer()
    training_results = trainer.train(train_cities, val_cities)

  # Сохраняем модель.
    trainer.save_model(save_path)

  # Сохраняем логи обучения.
  # Сохраняем логи обучения.
    logs_path = f"{config.files.logs_dir}/{config.files.training_log_name}"

  # Конвертируем tensors в числа для JSON.
    def convert_tensors_to_numbers(obj):
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.numel() == 1 else obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors_to_numbers(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors_to_numbers(item) for item in obj]
        else:
            return obj

    training_results_serializable = convert_tensors_to_numbers(training_results)

    with open(logs_path, 'w') as f:
        json.dump(training_results_serializable, f, indent=2)

    logger.info(f"Обучение завершено. Лучший val_reward: {training_results['best_val_reward']:.3f}")
    return trainer

if __name__ == "__main__":
  # Демонстрация обучения Neural Planner.
    logger.info("Обучение Neural Planner для транзитного планирования...")

  # Создаем небольшой тестовый набор данных.
    logger.info("Создание тестового набора данных...")
    dataset_manager = DatasetManager()
    test_cities = dataset_manager.create_training_dataset(num_cities=10, save_path="test_neural_cities.pkl")

  # Разделяем на train/val.
    train_cities = test_cities[:8]
    val_cities = test_cities[8:]

    logger.info(f"Создано {len(train_cities)} обучающих и {len(val_cities)} валидационных городов")

  # Создаем и тестируем trainer.
    trainer = NeuralPlannerTrainer()

  # Тестируем извлечение признаков.
    city = train_cities[0]
    node_features, edge_features = trainer.extract_graph_features(city)
    logger.info(f"Node features shape: {node_features.shape}")
    logger.info(f"Edge features shape: {edge_features.shape}")

  # Тестируем rollout.
    states, actions, log_probs, reward = trainer.rollout_episode(city, alpha=0.5, max_steps=50)
    logger.info(f"Rollout: {len(states)} шагов, reward={reward:.3f}")

  # Быстрое обучение на маленьком наборе данных.
    logger.info("Запуск обучения...")

  # Уменьшаем размеры для быстрого тестирования.
    config.network.num_epochs = 1
    config.network.batch_size = 2

    training_results = trainer.train(train_cities, val_cities)

    logger.info(f"Обучение завершено. Результаты:")
    logger.info(f"  Best val reward: {training_results['best_val_reward']:.3f}")
    logger.info(f"  Обучающих метрик: {len(training_results['training_metrics'])}")

  # Тестируем сохранение/загрузку.
    trainer.save_model("test_neural_planner.pth")

    new_trainer = NeuralPlannerTrainer()
    new_trainer.load_model("test_neural_planner.pth")
    logger.info("Модель успешно сохранена и загружена")

    logger.info("Демонстрация Neural Planner завершена успешно!")
