import logging
import config
logger = logging.getLogger(__name__)
import collections
import pickle
import warnings

from ortools.linear_solver import pywraplp
from shapely.geometry import Point, LineString
from tqdm import tqdm
import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from sklearn.cluster import DBSCAN  # Pip install scikit-learn.

def merge_close_stops(stops_gdf: gpd.GeoDataFrame,
                      merge_dist_m: float = 40.0,
                      crs_metric: str = "EPSG:32636") -> gpd.GeoDataFrame:
    """
    Объединяет все остановки, находящиеся ближе merge_dist_m, в одну.
    Геометрия каждой группы заменяется центроидом (средним x-y).
    Дополнительные колонки можно агрегировать по-другому (sum/mean/first).
    """
    if len(stops_gdf) == 0:
        return stops_gdf.copy()

  # 1) переводим в метрическую проекцию.
    stops = stops_gdf.to_crs(crs_metric).copy()

  # 2) координаты для DBSCAN.
    coords = np.c_[stops.geometry.x, stops.geometry.y]

  # 3) кластеризуем (eps = порог, min_samples = 1 => каждая точка попадает в кластер).
    labels = DBSCAN(eps=merge_dist_m, min_samples=1).fit_predict(coords)
    stops["cluster"] = labels

  # 4) агрегируем кластеры.
    agg = {
        "geometry": lambda g: Point(g.x.mean(), g.y.mean())
    }
    merged = stops.groupby("cluster").agg(agg).reset_index(drop=True)
    merged_gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs=crs_metric)

  # 5) возвращаем в оригинальную CRS, чтобы остальной код не менялся.
    merged_gdf = merged_gdf.to_crs(stops_gdf.crs)

    return merged_gdf

  # Отключаем предупреждения.
warnings.filterwarnings('ignore')

  # Загружаем данные из GeoJSON файлов.
logger.info("Загрузка данных из локальных GeoJSON файлов...")

  # Загрузка дорожной сети.
roads = gpd.read_file(config.data("clusters", "roads_cluster.geojson"))
roads = roads.to_crs('EPSG:3857')
logger.info("Дорожная сеть загружена:", len(roads), "объектов")

  # Загрузка всех других данных и объединение в один DataFrame.
data_parts = []

try:
    residential_buildings = gpd.read_file(config.data("clusters", "residential_buildings_cluster.geojson"))
    residential_buildings = residential_buildings.to_crs('EPSG:3857')
    residential_buildings['building'] = 'residential'
    data_parts.append(residential_buildings)
except FileNotFoundError:
    pass

try:
    commercial_buildings = gpd.read_file(config.data("clusters", "commercial_buildings_cluster.geojson"))
    commercial_buildings = commercial_buildings.to_crs('EPSG:3857')
    commercial_buildings['office'] = True
    data_parts.append(commercial_buildings)
except FileNotFoundError:
    pass

try:
    poi = gpd.read_file(config.data("clusters", "poi_cluster.geojson"))
    poi = poi.to_crs('EPSG:3857')
    poi['amenity'] = 'school'
    data_parts.append(poi)
except FileNotFoundError:
    pass

try:
    bus_stops = gpd.read_file(config.data("clusters", "bus_stops_cluster.geojson"))
    bus_stops = bus_stops.to_crs('EPSG:3857')
    bus_stops = merge_close_stops(bus_stops, merge_dist_m=90)
    bus_stop_points = bus_stops[bus_stops.geometry.type == 'Point'].copy()
    logger.info(f"Загружено {len(bus_stops)} автобусных остановок из файла")
except FileNotFoundError:
    bus_stops = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:3857')
    bus_stops = merge_close_stops(bus_stops, merge_dist_m=90)
    bus_stop_points = gpd.GeoDataFrame(columns=['geometry'], crs='EPSG:3857')
    logger.info("Файл bus_stops_cluster.geojson не найден")

if data_parts:
    data = pd.concat(data_parts, ignore_index=True)
    data = gpd.GeoDataFrame(data, crs='EPSG:3857')
    logger.info("Общие данные загружены:", len(data), "объектов")
else:
    logger.info("ОШИБКА: Не удалось загрузить данные!")
    exit(1)

  # === ФИЛЬТРАЦИЯ ПО ГРАНИЦАМ КЛАСТЕРА (ОПЦИОНАЛЬНО) ===.
try:
    with open('results/largest_cluster_boundary.pkl', 'rb') as f:
        cluster_boundary = pickle.load(f)

    cluster_gdf = gpd.GeoDataFrame(geometry=[cluster_boundary], crs='EPSG:3857')

  # Фильтруем данные по границам кластера.
    def filter_by_cluster(gdf, cluster_boundary):
        if len(gdf) == 0:
            return gdf
        if gdf.geometry.type.iloc[0] in ['Polygon', 'MultiPolygon']:
            gdf_copy = gdf.copy()
            gdf_copy['geometry'] = gdf.geometry.centroid
            return gdf[gdf_copy.within(cluster_boundary)]
        return gdf[gdf.within(cluster_boundary)]

    bus_stops_in_cluster = filter_by_cluster(bus_stops, cluster_boundary)
  # Используйте bus_stops_in_clu.

  # Ster только для анализа покрытия, но для визуализации — исходные bus_stops!

    logger.info(f"После фильтрации по кластеру: {len(data)} объектов, {len(roads)} дорог")

except FileNotFoundError:
    logger.info("Файл границ кластера не найден, используем все данные")

  # Преобразование дорожной сети в граф.
logger.info("Создание графа дорожной сети...")

  # Создаем узлы и ребра из линий дорог.
nodes_data = []
edges_data = []
node_id = 0
node_coords = {}  # Словарь для хранения уникальных координат узлов.

  # Функция для округления координат (для обнаружения совпадающих узлов).
def round_coords(x, y, digits=6):
    return (round(x, digits), round(y, digits))

  # Обработка каждой дороги для извлечения узлов и ребер.
for idx, road in roads.iterrows():
    if road.geometry.geom_type == 'LineString':
        coords = list(road.geometry.coords)

  # Проходим по всем точкам линии.
        for i in range(len(coords)):
            x, y = coords[i]
            rounded = round_coords(x, y)

  # Если эта точка еще не была добавлена как узел, добавляем.
            if rounded not in node_coords:
                node_coords[rounded] = node_id
                nodes_data.append({
                    'node_id': node_id,
                    'osmid': f'n{node_id}',
                    'geometry': Point(x, y),
                    'x': x,
                    'y': y
                })
                node_id += 1

  # Добавляем ребро между последовательными точками.
            if i > 0:
                prev_x, prev_y = coords[i - 1]
                prev_rounded = round_coords(prev_x, prev_y)

  # Создаем ребро между предыдущей и текущей точками.
                u = node_coords[prev_rounded]
                v = node_coords[rounded]

                if u != v:  # Проверяем, что это не тот же самый узел.
                    edges_data.append({
                        'u': u,
                        'v': v,
                        'osmid': f'e{len(edges_data)}',
                        'geometry': LineString([coords[i - 1], coords[i]])
                    })

  # Создаем геодатафреймы для узлов и ребер.
nodes = gpd.GeoDataFrame(nodes_data, crs=roads.crs)
edges = gpd.GeoDataFrame(edges_data, crs=roads.crs)

logger.info(f"Граф создан: {len(nodes)} узлов и {len(edges)} ребер")

  # Создаем граф NetworkX для вычисления центральности и других сетевых метрик.
G = nx.Graph()
for _, row in nodes.iterrows():
    G.add_node(row['node_id'], x=row['x'], y=row['y'])

for _, row in edges.iterrows():
    G.add_edge(row['u'], row['v'])

  # Проверяем связность графа.
components = list(nx.connected_components(G))
if len(components) > 0:
    largest_cc = max(components, key=len)
    logger.info(f"Крупнейший связный компонент содержит {len(largest_cc)} узлов из {len(G.nodes())}")

  # Если граф несвязный, работаем только с крупнейшим компонентом.
    if len(largest_cc) < len(G.nodes()):
        logger.info("Внимание: граф несвязный, используем только крупнейший компонент")
        G = G.subgraph(largest_cc).copy()
        nodes = nodes[nodes['node_id'].isin(largest_cc)]
        edges = edges[edges['u'].isin(largest_cc) & edges['v'].isin(largest_cc)]
else:
    logger.info("Внимание: граф пустой!")

  # Создаем кандидатов из узлов.
candidates = nodes.copy()
candidates['centroid'] = candidates['geometry']
candidates.reset_index(drop=True, inplace=True)

  # Разделяем данные по типам на основе структуры данных.
logger.info("Разделение данных на категории...")

  # Анализируем структуру данных, чтобы выявить различные типы объектов.
if 'type' in data.columns:
    logger.info("Типы объектов в данных:", data['type'].value_counts().to_dict())

  # 1. Пытаемся найти границы административных районов.
districts = None
if 'boundary' in data.columns:
    districts = data[data['boundary'] == 'administrative'].copy()
    logger.info(f"Найдено {len(districts)} административных границ по тегу 'boundary'")

if districts is None or len(districts) == 0:
  # Ищем полигоны, которые могут быть районами.
    possible_districts = data[data.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    if len(possible_districts) > 0:
  # Берем самые крупные полигоны.
        area_threshold = possible_districts.geometry.area.quantile(0.75)
        districts = possible_districts[possible_districts.geometry.area > area_threshold].copy()
        logger.info(f"Найдено {len(districts)} возможных административных районов по размеру полигонов")

    if districts is None or len(districts) == 0:
  # Если не нашли районы, создаем простую сетку.
        from shapely.geometry import box

        bounds = data.total_bounds
  # Делим область на 4 района.
        x_mid = (bounds[0] + bounds[2]) / 2
        y_mid = (bounds[1] + bounds[3]) / 2

        district_geometries = [
            box(bounds[0], bounds[1], x_mid, y_mid),  # Юго-западный.
            box(x_mid, bounds[1], bounds[2], y_mid),  # Юго-восточный.
            box(bounds[0], y_mid, x_mid, bounds[3]),  # Северо-западный.
            box(x_mid, y_mid, bounds[2], bounds[3])  # Северо-восточный.
        ]

        districts = gpd.GeoDataFrame({
            'geometry': district_geometries,
            'name': ['Район 1', 'Район 2', 'Район 3', 'Район 4']
        }, crs=data.crs)
        logger.info("Созданы 4 искусственных района на основе границ области")

  # Добавляем имена районам, если их нет.
if 'name' not in districts.columns:
    districts['name'] = [f"District_{i}" for i in range(len(districts))]

  # 2. Ищем жилые здания.
residential_buildings = None
building_tags = ['residential', 'apartments', 'house', 'detached', 'terrace', 'dormitory', 'bungalow', 'cabin', 'farm']

if 'building' in data.columns:
  # Если есть тег building, используем его для фильтрации.
    residential_buildings = data[data['building'].isin(building_tags)].copy()
    logger.info(f"Найдено {len(residential_buildings)} жилых зданий по тегу 'building'")

if residential_buildings is None or len(residential_buildings) == 0:
  # Если не нашли здания по тегу, ищем полигоны.
    polys = data[data.geometry.type.isin(['Polygon', 'MultiPolygon'])].copy()
    if len(polys) > 0:
  # Берем полигоны небольшого размера, которые могут быть зданиями.
        area_threshold = polys.geometry.area.quantile(0.75)  # Верхняя граница для зданий.
        residential_buildings = polys[polys.geometry.area < area_threshold].copy()
        logger.info(f"Взяли {len(residential_buildings)} полигонов как потенциальные жилые здания")

  # 3. Ищем рабочие места.
work_buildings = None
work_tags = ['office', 'industrial', 'commercial', 'retail', 'shop']

for tag in work_tags:
    if tag in data.columns:
        tag_buildings = data[data[tag].notna()].copy()
        if work_buildings is None:
            work_buildings = tag_buildings
        else:
            work_buildings = pd.concat([work_buildings, tag_buildings])

if work_buildings is not None:
    work_buildings = work_buildings.drop_duplicates()
    logger.info(f"Найдено {len(work_buildings)} зданий, связанных с работой")
else:
  # Если не нашли работы по тегам, используем случайное подмножество полигонов.
    if 'residential_buildings' in locals() and residential_buildings is not None:
        non_residential_polys = data[(data.geometry.type.isin(['Polygon', 'MultiPolygon'])) &
                                     (~data.index.isin(residential_buildings.index))].copy()
        if len(non_residential_polys) > 0:
            work_buildings = non_residential_polys.sample(min(len(non_residential_polys), 100))
            logger.info(f"Взяли {len(work_buildings)} случайных полигонов как рабочие места")
        else:
            work_buildings = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=data.crs)
            logger.info("Не удалось найти здания для рабочих мест")
    else:
        work_buildings = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=data.crs)
        logger.info("Не удалось найти здания для рабочих мест")

  # 4. Ищем POI (Points of Interest).
poi = None
poi_tags = ['railway', 'subway', 'amenity']

for tag in poi_tags:
    if tag in data.columns:
        tag_poi = data[data[tag].notna()].copy()
        if poi is None:
            poi = tag_poi
        else:
            poi = pd.concat([poi, tag_poi])

if poi is not None:
    poi = poi.drop_duplicates()
  # Оставляем только точки.
    poi = poi[poi.geometry.type == 'Point'].copy()
    logger.info(f"Найдено {len(poi)} точек интереса (POI)")
else:
  # Если не нашли POI по тегам, используем случайное подмножество точек.
    points = data[data.geometry.type == 'Point'].copy()
    if len(points) > 0:
        poi = points.sample(min(len(points), 30))
        logger.info(f"Взяли {len(poi)} случайных точек как POI")
    else:
        poi = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=data.crs)
        logger.info("Не удалось найти POI")

  # 5. Используем уже загруженные автобусные остановки.
  # 5. Используем уже загруженные автобусные остановки.
logger.info(f"Используем {len(bus_stops)} автобусных остановок, загруженных из файла")

logger.info(f"Данные разделены на категории:")
logger.info(f"- Административные районы: {len(districts)} объектов")
logger.info(f"- Жилые здания: {len(residential_buildings) if residential_buildings is not None else 0} объектов")
logger.info(f"- Рабочие места: {len(work_buildings) if work_buildings is not None else 0} объектов")
logger.info(f"- Точки интереса: {len(poi) if poi is not None else 0} объектов")
logger.info(f"- Автобусные остановки: {len(bus_stops) if bus_stops is not None else 0} объектов")

  # Преобразуем к единой проекции для метрических расчетов.
districts = districts.to_crs(epsg=3857)

  # Пытаемся загрузить данные о населении районов.
try:
    district_population = pd.read_csv('district_population.csv')
    districts = districts.merge(district_population, left_on='name', right_on='district_name', how='left')
    logger.info("Данные о населении районов загружены из district_population.csv")
except Exception as e:
    logger.info(f"Не удалось загрузить данные о населении: {e}")
    logger.info("Создание примерных данных о населении для районов...")
    districts['district_name'] = districts['name']
  # Оценка населения на основе площади.
    districts['population'] = districts.geometry.area / 1000000 * 5000  # 5000 человек на км².

  # Обработка жилых зданий.
logger.info("Обработка жилых зданий...")
residential_buildings = residential_buildings.to_crs(epsg=3857)

  # Определение числа этажей и квартир.
if 'building:levels' in residential_buildings.columns:
    residential_buildings['levels'] = pd.to_numeric(residential_buildings['building:levels'], errors='coerce').fillna(2)
else:
    residential_buildings['levels'] = 2  # По умолчанию 2 этажа.

residential_buildings['area'] = residential_buildings.geometry.area * residential_buildings['levels']
mean_flat_area = 38  # Средняя площадь квартиры в м².
residential_buildings['flats'] = (residential_buildings['area'] / mean_flat_area).round(0)

  # Определение района для каждого здания.
build_pts = residential_buildings.copy()
build_pts['geometry'] = residential_buildings.geometry.centroid
build_pts = gpd.sjoin(build_pts, districts[['geometry', 'name', 'population']], how='left', predicate='within')

  # Обработка случаев, когда здание не находится ни в одном районе.
build_pts['name_right'] = build_pts['name_right'].fillna('Unknown')
build_pts['population'] = build_pts['population'].fillna(50000)  # Значение по умолчанию.

  # Суммарное число квартир по району.
flats_sum = build_pts.groupby('name_right')['flats'].sum().reset_index().rename(columns={'flats': 'flats_sum'})
build_pts = build_pts.merge(flats_sum, on='name_right')

  # Предотвращаем деление на ноль.
build_pts['flats_sum'] = build_pts['flats_sum'].replace(0, 1)

  # Среднее население на квартиру.
build_pts['people_per_flat'] = build_pts['population'] / build_pts['flats_sum']
build_pts['pop_est'] = (build_pts['people_per_flat'] * build_pts['flats']).round(0).astype(int)

  # ИСПРАВЛЕНО: преобразуем данные в единую систему координат UTM для поиска ближайших узлов.
logger.info("Привязка зданий к узлам графа...")
candidates = candidates.to_crs(epsg=32636)  # UTM zone для СПб.
nodes = nodes.to_crs(epsg=32636)  # UTM zone для СПб.
build_pts = build_pts.to_crs(epsg=32636)  # UTM zone для СПб.

  # ИСПРАВЛЕНО: Используем KDTree для эффективного поиска ближайших узлов.
from scipy.spatial import KDTree

  # ИСПРАВЛЕНО: создаем массив координат из геометрии, а не из x, y полей.
coords_nodes = np.array([(node.geometry.x, node.geometry.y) for _, node in nodes.iterrows()])
tree = KDTree(coords_nodes)

def find_nearest_node(point):
    """Находим ближайший узел для точки."""
  # ИСПРАВЛЕНО: правильная передача координат в KDTree.
    dist, idx = tree.query((point.x, point.y), k=1)
    return int(nodes.iloc[idx]['node_id'])

build_pts['nearest_node'] = build_pts.geometry.apply(find_nearest_node)

  # Суммируем жителей по ближайшему узлу.
pop_per_node = build_pts.groupby('nearest_node')['pop_est'].sum()
candidates['population'] = candidates['node_id'].map(pop_per_node).fillna(0).astype(int)

  # Обработка рабочих мест.
logger.info("Обработка данных о рабочих местах...")
work_buildings = work_buildings.to_crs(epsg=3857)

  # Определение числа этажей.
if 'building:levels' in work_buildings.columns:
    work_buildings['levels'] = pd.to_numeric(work_buildings['building:levels'], errors='coerce').fillna(2)
else:
    work_buildings['levels'] = 2  # По умолчанию 2 этажа.

  # Вычисление площади.
work_buildings['flat_area'] = work_buildings.geometry.apply(lambda geom:
                                                            geom.area if hasattr(geom, 'area') and callable(
                                                                geom.area) else np.nan)
work_buildings['area'] = work_buildings['flat_area'] * work_buildings['levels']

  # Оценка количества работников.
def estimate_employees(row):
  # Проверяем наличие различных тегов для определения типа.
    is_office = False
    is_industrial = False
    is_commercial = False

    for tag in work_tags:
        if tag in row and pd.notna(row[tag]):
            if tag == 'office':
                is_office = True
            elif tag == 'industrial':
                is_industrial = True
            else:  # Shop, retail, commercial.
                is_commercial = True

    if pd.notna(row.get('area')) and row.get('area', 0) > 0:
        if is_office:
            return row['area'] / 10
        elif is_industrial:
            return row['area'] / 40
        elif is_commercial:
            return row['area'] / 6

  # Для точечных объектов или объектов без площади.
    if hasattr(row, 'geometry') and row.geometry.type == 'Point':
        if is_office:
            return 30  # Среднее число для офиса.
        elif is_industrial:
            return 50  # Среднее число для промышленного объекта.
        elif is_commercial:
            return 10  # Среднее число для магазина.

    return 3  # Значение по умолчанию.

work_buildings['employees'] = work_buildings.apply(estimate_employees, axis=1).fillna(1).astype(int)

  # Центроиды для площадных объектов.
jobs_pts = work_buildings.copy()
if 'geometry' in jobs_pts.columns and len(jobs_pts) > 0:
    jobs_pts.loc[jobs_pts.geometry.type != 'Point', 'geometry'] = jobs_pts.loc[
        jobs_pts.geometry.type != 'Point', 'geometry'].centroid

  # Привязка к ближайшему узлу.
jobs_pts = jobs_pts.to_crs(epsg=32636)  # ИСПРАВЛЕНО: явно указываем преобразование.
jobs_pts['nearest_node'] = jobs_pts.geometry.apply(find_nearest_node)

  # Суммируем работников по узлам.
jobs_per_node = jobs_pts.groupby('nearest_node')['employees'].sum()
candidates['jobs'] = candidates['node_id'].map(jobs_per_node).fillna(0).astype(int)

  # Обработка точек интереса (POI).
logger.info("Обработка точек интереса (POI)...")
poi = poi.to_crs(epsg=32636)  # ИСПРАВЛЕНО: единая CRS.
poi_points = poi[poi.geometry.type == 'Point'].copy()
  # Равномерное размещение connection stops из POI.
logger.info("Points of interest:", len(poi_points))

  # Проверяем, есть ли POI.
if len(poi_points) > 0:
  # Найдем ближайшие узлы для каждого POI.
  # ИСПРАВЛЕНО: Используем центроиды для всех типов геометрии.
    poi_points['geometry'] = poi_points.geometry.centroid
    poi_points['nearest_node'] = poi_points.geometry.apply(find_nearest_node)

  # Получаем координаты для каждого узла из candidates.
    node_coords = {}
    for idx, row in candidates.iterrows():
        node_coords[row.node_id] = (row.geometry.x, row.geometry.y)

  # Собираем информацию о connection stops.
    connection_data = []
    for node_id in set(poi_points['nearest_node'].values):
  # Подсчитываем, сколько POI связаны с этим узлом (важность узла).
        poi_count = sum(1 for n in poi_points['nearest_node'] if n == node_id)
  # Получаем координаты узла.
        if node_id in node_coords:
            x, y = node_coords[node_id]
            connection_data.append({
                'node_id': node_id,
                'x': x,
                'y': y,
                'importance': poi_count
            })

  # Переводим в DataFrame для удобства обработки.
    connections_df = pd.DataFrame(connection_data)

  # Устанавливаем минимальное расстояние между connection stops.
    min_distance = 250  # В метрах.

  # Если количество connection stops слишком велико, фильтруем их.
    if len(connections_df) > 1:
  # Преобразуем в numpy массив для быстрой обработки.
        coords = np.array(connections_df[['x', 'y']])

  # Используем KDTree для эффективного поиска соседей.
        tree = KDTree(coords)

  # Сортируем connection stops по важности (количеству связанных POI).
        sorted_connections = connections_df.sort_values('importance', ascending=False)

  # Выбираем остановки с учетом минимального расстояния.
        selected_stops = []
        selected_indices = []

        for idx, row in sorted_connections.iterrows():
            x, y = row['x'], row['y']
            coord = np.array([x, y])

  # Если это первая остановка или она достаточно далеко от уже выбранных.
            if not selected_indices or not any(
                    np.linalg.norm(coord - coords[i]) < min_distance for i in selected_indices):
                selected_stops.append(row['node_id'])
  # Находим индекс в исходном массиве coords.
                selected_indices.append(np.where((coords[:, 0] == x) & (coords[:, 1] == y))[0][0])

        connection_stops = set(selected_stops)
    else:
  # Если POI мало, просто берем все связанные узлы.
        connection_stops = set(connections_df['node_id'])
else:
    connection_stops = set()

logger.info("Connection stops:", len(connection_stops))

candidates['is_connection'] = candidates['node_id'].isin(connection_stops)

  # Вычисляем центральность узлов.
deg_cent = nx.degree_centrality(G)
candidates['degree_centrality'] = candidates['node_id'].map(deg_cent)

  # Выбираем только non-connection nodes как кандидаты на key stops.
candidate_idxs = candidates[~candidates.is_connection].index

  # Обработка существующих автобусных остановок.
logger.info("Обработка существующих автобусных остановок...")

  # Для анализа покрытия используем UTM.
bus_stops_for_analysis = bus_stops.to_crs(epsg=32636)
  # Bus_stop_points остаются в EPSG:3857 для визуализации.

  # Убеждаемся, что считаем только уникальные остановки.
if len(bus_stop_points) > 0:
    existing_stops_count = len(bus_stop_points.drop_duplicates(subset='geometry'))
  # ИСПРАВЛЕНО: явно привязываем автобусные остановки к узлам.
    bus_stop_points['nearest_node'] = bus_stop_points.geometry.apply(find_nearest_node)
else:
    existing_stops_count = 50  # По умолчанию, если нет данных.
    logger.info(f"Данные об остановках не найдены. Используем значение по умолчанию: {existing_stops_count} остановок")

max_optimized_stops = existing_stops_count

remaining = max_optimized_stops - len(connection_stops)
N_KEYSTOPS = min(70, max(1, int(remaining * 0.3)))  # 30% key, но не больше 40 (пример).
N_ORDINARY_stops = round(max(0, remaining - N_KEYSTOPS))

logger.info(
    f"Key: {N_KEYSTOPS}, Ordinary: {N_ORDINARY_stops}, Connection: {len(connection_stops)}, Всего: {N_KEYSTOPS + N_ORDINARY_stops + len(connection_stops)} (Предел: {max_optimized_stops})")

  # Считаем pairwise расстояния между всеми узлами и кандидатами.
logger.info("Расчет расстояний между узлами...")
  # ИСПРАВЛЕНО: правильно используем координаты в UTM.
coords_all = np.array(list(zip(candidates.geometry.x, candidates.geometry.y)))
coords_key = coords_all[candidate_idxs]

  # Для больших наборов данных ограничим максимальное расстояние назначения.
max_assignment_dist = 3000  # Метров, можно настроить.

  # Евклидово расстояние в метрах вместо haversine (т.к. уже в UTM).
from scipy.spatial.distance import cdist

D = cdist(coords_all, coords_key, metric='euclidean')  # Расстояния в метрах.

  # Строим P-median через ortools MIP с оптимизацией памяти.
logger.info("Строим P-median через ortools MIP")
solver = pywraplp.Solver.CreateSolver('SCIP')

  # Проверяем, что есть достаточно кандидатов для создания key stops.
if len(candidate_idxs) == 0:
    logger.info("Нет кандидатов для key stops!")
    key_stop_idxs = []
else:
  # Оптимизация 1: Используем другой решатель для более надежного решения.
    solver = pywraplp.Solver.CreateSolver('CBC')

    if not solver:
        logger.info("Не удалось создать решатель CBC, пробуем SCIP")
        solver = pywraplp.Solver.CreateSolver('SCIP')

  # Оптимизация 2: Задаем параметры решателя.
    solver.SetTimeLimit(300000)  # 5 минут в миллисекундах.

    logger.info("Создание переменных оптимизации...")
    x = {}
    y = {}

  # Оптимизация 3: Создаем словари для группировки данных.
    relevant_j_by_i = {}
    i_by_j = collections.defaultdict(list)

  # Оптимизация 4: Обрабатываем данные партиями для снижения нагрузки на память.
    batch_size = 500
    for i_batch_start in tqdm(range(0, len(candidates), batch_size)):
        i_batch_end = min(i_batch_start + batch_size, len(candidates))

        for i in range(i_batch_start, i_batch_end):
  # Находим только k ближайших кандидатов вместо проверки всех.
            k = min(30, len(candidate_idxs))  # Проверяем только 30 ближайших.
            closest_j = np.argpartition(D[i, :], k)[:k]

  # Фильтруем по максимальному расстоянию.
            relevant_j = [j_idx for j_idx in closest_j if D[i, j_idx] < max_assignment_dist]

            if not relevant_j and len(candidate_idxs) > 0:
  # Если нет кандидатов в разумном расстоянии, добавляем ближайшего.
                j_idx = np.argmin(D[i, :])
                relevant_j = [j_idx]

  # Сохраняем релевантных кандидатов для текущего узла.
            relevant_j_by_i[i] = relevant_j

  # Создаем переменные x сразу.
            for j_idx in relevant_j:
                x[i, j_idx] = solver.IntVar(0, 1, f'x_{i}_{j_idx}')
                i_by_j[j_idx].append(i)

  # Оптимизация 5: Освобождаем память от огромной матрицы расстояний.
    distances_dict = {}  # Словарь для хранения только нужных расстояний.
    for i in relevant_j_by_i:
        for j_idx in relevant_j_by_i[i]:
            distances_dict[(i, j_idx)] = D[i, j_idx]

  # Ограничение: для каждого i, сумма x[i, j] = 1.
    logger.info("Ограничение: для каждого i, сумма x[i, j] = 1")
    for i in tqdm(relevant_j_by_i.keys()):
        solver.Add(solver.Sum([x[i, j] for j in relevant_j_by_i[i]]) == 1)

  # Создаем переменные y для кандидатов в key stops.
    logger.info("Создание переменных для ключевых остановок...")
    for j in tqdm(range(len(candidate_idxs))):
        y[j] = solver.IntVar(0, 1, f'y_{j}')

  # Оптимизация 6: Группируем ограничения по j вместо перебора всех i,j.
    logger.info("Связывание переменных x и y...")
    for j in tqdm(i_by_j.keys()):
        for i in i_by_j[j]:
            if (i, j) in x:
                solver.Add(x[i, j] <= y[j])

  # Точно задаем нужное количество key stops.
    solver.Add(solver.Sum([y[j] for j in range(len(candidate_idxs))]) <= N_KEYSTOPS)

  # Оптимизация 7: Более эффективное построение целевой функции.
    logger.info("Формирование целевой функции...")
    objective_terms = []

  # Обрабатываем целевую функцию партиями.
    for i in tqdm(relevant_j_by_i.keys()):
  # Предварительно вычисляем вес для узла i.
        w = candidates.loc[i, 'degree_centrality'] + 0.001
        w *= (candidates.loc[i, 'population'] * 0.7 + candidates.loc[i, 'jobs'] * 0.3 + 1) / 100

        for j_idx in relevant_j_by_i[i]:
            d = distances_dict.get((i, j_idx), 0)
            objective_terms.append(w * d * x[i, j_idx])

  # Минимизируем целевую функцию.
    solver.Minimize(solver.Sum(objective_terms))

  # Оптимизация 8: Добавляем вывод прогресса и логи.
    logger.info(f"Задача создана: {solver.NumVariables()} переменных, {solver.NumConstraints()} ограничений")
    logger.info("Запуск решателя...")

  # Решаем задачу.
    status = solver.Solve()
    logger.info(f"Статус решения: {status}")

    if status != pywraplp.Solver.OPTIMAL and status != pywraplp.Solver.FEASIBLE:
        logger.info('Не найдено оптимальное или допустимое решение!')
        logger.info(f"Время работы решателя: {solver.WallTime()} мс")

  # Оптимизация 9: Если не найдено оптимальное решение, используем жадный алгоритм.
        logger.info("Применяем жадный алгоритм для поиска key stops...")

  # Создаем список кандидатов с их весами значимости.
        candidates_with_weights = []
        for j in range(len(candidate_idxs)):
            idx = candidate_idxs[j]
            weight = candidates.loc[idx, 'degree_centrality'] + 0.001
            weight *= (candidates.loc[idx, 'population'] * 0.7 + candidates.loc[idx, 'jobs'] * 0.3 + 1) / 100
            candidates_with_weights.append((j, idx, weight))

  # Сортируем кандидатов по весу.
        candidates_with_weights.sort(key=lambda x: x[2], reverse=True)

  # Выбираем N_KEYSTOPS лучших кандидатов.
        selected_candidates = candidates_with_weights[:N_KEYSTOPS]
        key_stop_idxs = [idx for _, idx, _ in selected_candidates]

        logger.info(f"Жадный алгоритм выбрал {len(key_stop_idxs)} key stops")
    else:
  # Получаем результаты из оптимизации.
        key_stop_idxs = []
        for j in range(len(candidate_idxs)):
            if j in y and y[j].solution_value() > 0.5:
                key_stop_idxs.append(candidate_idxs[j])

        logger.info(f"Найдено {len(key_stop_idxs)} key stops через оптимизацию")
        logger.info(f"Значение целевой функции: {solver.Objective().Value()}")

with open('key_stop_idxs.pkl', 'wb') as f:
    pickle.dump(key_stop_idxs, f)
"""

with open("key_stop_idxs.pkl", "rb") as f:
    key_stop_idxs = pickle.load(f)
"""
logger.info("Key stops:", len(key_stop_idxs))

  # Для каждого узла, demand = population*production + jobs*attraction.
candidates['transit_demand'] = candidates['population'] * 1.0 + candidates['jobs'] * 0.7

service_radius = 550  # Meters.

  # Получаем ID уже занятых stops.
used_nodes = set(list(connection_stops) + list(candidates.loc[key_stop_idxs, 'node_id'].values))
ordinary_candidates = candidates[~candidates.node_id.isin(used_nodes)].copy()

  # ВСЮДУ ДАЛЕЕ используем coords в UTM (метры!).
coords_all = np.array(list(zip(candidates.geometry.x, candidates.geometry.y)))
coords_ord = np.array(list(zip(ordinary_candidates.geometry.x, ordinary_candidates.geometry.y))) if len(
    ordinary_candidates) > 0 else np.array([])

  # Проверяем, есть ли кандидаты для обычных остановок.
if len(ordinary_candidates) == 0:
    logger.info("Нет кандидатов для обычных остановок!")
    ordinary_stop_geoms = gpd.GeoDataFrame(columns=candidates.columns)
else:
  # Расчет расстояний в метрах.
    D_ord = cdist(coords_all, coords_ord)  # Уже в метрах!
    coverage = (D_ord < service_radius)

  # Задача максимального покрытия: выбрать N_ORDINARY_stops, чтобы покрыть max суммарный спрос demand.
    logger.info("Задача максимального покрытия")
    solver2 = pywraplp.Solver.CreateSolver('SCIP')
    z = {}
    for j in tqdm(range(len(ordinary_candidates))):
        z[j] = solver2.IntVar(0, 1, '')

  # Ограничение — число ordinary stops.
    solver2.Add(solver2.Sum([z[j] for j in range(len(ordinary_candidates))]) <= N_ORDINARY_stops)

    min_distance = 250  # Метров.

  # 1. Создаём массив координат уже выбранных остановок (connection и key stops).
    existing_stops_coords = np.array([
        [candidates.loc[i, 'geometry'].x, candidates.loc[i, 'geometry'].y]
        for i in candidates.index
        if candidates.loc[i, 'node_id'] in used_nodes
    ])

  # 2. Для каждого кандидата на обычную остановку проверяем расстояние до существующих остановок.
    from scipy.spatial import cKDTree

    if len(existing_stops_coords) > 0:  # Проверка, есть ли уже выбранные остановки.
        existing_tree = cKDTree(existing_stops_coords)

        forbidden_candidates = []
        for j in range(len(ordinary_candidates)):
            dist, _ = existing_tree.query([coords_ord[j]], k=1)
            if dist < min_distance:  # ИСПРАВЛЕНО: dist вместо dist[0].
                forbidden_candidates.append(j)

  # Запрещаем выбор кандидатов, которые слишком близко к существующим остановкам.
        for j in forbidden_candidates:
            solver2.Add(z[j] == 0)

  # 3. Строим KDTree для кандидатов на обычные остановки.
    ord_tree = cKDTree(coords_ord)

  # 4. Находим все пары кандидатов, которые слишком близко друг к другу.
    close_pairs = list(ord_tree.query_pairs(min_distance))

  # 5. Добавляем ограничения: из каждой пары "близких" кандидатов можно выбрать максимум одного.
    penalty_for_close_stops = 2000  # Настройте этот параметр.
    penalty_terms = []

    for i, j in close_pairs:
  # Создаем переменную, которая = 1, если обе остановки выбраны.
        both_selected = solver2.IntVar(0, 1, f'both_{i}_{j}')
        solver2.Add(both_selected >= z[i] + z[j] - 1)
        solver2.Add(both_selected <= z[i])
        solver2.Add(both_selected <= z[j])

  # Добавляем штраф в целевую функцию.
        penalty_terms.append(both_selected * penalty_for_close_stops)

  # Модифицируем целевую функцию - более сложная версия.
    logger.info("Целевая — макс суммарный спрос покрытых")

  # 1. Вводим переменные: covered[i].
    covered = {}
    for i in range(len(candidates)):
        covered[i] = solver2.IntVar(0, 1, f'covered_{i}')

    for i in tqdm(range(len(candidates))):
  # Если уже покрыто key/connection stop, то константа 1.
        if candidates.loc[i, 'node_id'] in used_nodes:
            covered[i].SetBounds(1, 1)
            continue
        dj = [j for j in range(len(ordinary_candidates)) if coverage[i, j]]
        if not dj:
            covered[i].SetBounds(0, 0)
            continue
  # Covered[i] <= сумма выбранных остановок для этой точки.
        solver2.Add(covered[i] <= solver2.Sum([z[j] for j in dj]))
  # Если выбран хотя бы один stop — covered[i]=1.
        for j in dj:
            solver2.Add(z[j] <= covered[i])

  # Улучшенная целевая функция: учитываем население, рабочие места и доступность.
    pop_weight = 1.0
    job_weight = 0.8
    accessibility_weight = 0.2  # Для степени центральности.

    objective_terms = []
    for i in range(len(candidates)):
        pop_term = candidates.loc[i, 'population'] * pop_weight * covered[i]
        job_term = candidates.loc[i, 'jobs'] * job_weight * covered[i]
        access_term = candidates.loc[i, 'degree_centrality'] * accessibility_weight * covered[i]
        objective_terms.append(pop_term + job_term + access_term)

  # Максимизировать общее взвешенное покрытие за вычетом штрафов.
    solver2.Maximize(
        solver2.Sum(objective_terms) -
        solver2.Sum(penalty_terms)
    )

    status2 = solver2.Solve()
    if status2 != pywraplp.Solver.OPTIMAL:
        logger.info('No optimal ordinary stops!')
        ordinary_stop_geoms = gpd.GeoDataFrame(columns=candidates.columns)
    else:
        ordinary_stop_geoms = ordinary_candidates.iloc[
            [j for j in range(len(ordinary_candidates)) if z[j].solution_value() > 0.5]].copy()

with open('ordinary_stop_geoms.pkl', 'wb') as f:
    pickle.dump(ordinary_stop_geoms, f)
"""
with open("ordinary_stop_geoms.pkl", "rb") as f:
    ordinary_stop_geoms = pickle.load(f)
"""
logger.info("Ordinary stops:", len(ordinary_stop_geoms))

  # All optimized stops:.
opt_stops_parts = []

if len(connection_stops) > 0:
    conn_stops = candidates[candidates.node_id.isin(connection_stops)].copy()
    conn_stops['type'] = 'connection'
    opt_stops_parts.append(conn_stops)

if len(key_stop_idxs) > 0:
    key_stops = candidates.loc[key_stop_idxs].copy()
    key_stops['type'] = 'key'
    opt_stops_parts.append(key_stops)

if len(ordinary_stop_geoms) > 0:
    ordinary_stop_geoms['type'] = 'ordinary'
    opt_stops_parts.append(ordinary_stop_geoms)

if opt_stops_parts:
    opt_stops = pd.concat(opt_stops_parts)
else:
    opt_stops = gpd.GeoDataFrame(columns=candidates.columns)
    opt_stops['type'] = []

  # Приводим всё к EPSG:3857 для визуализации (как в data_download.py).
edges = edges.to_crs('EPSG:3857')
opt_stops = opt_stops.to_crs('EPSG:3857')
  # Bus_stop_points уже в EPSG:3857.

  # Улучшенная визуализация карты.
fig, ax = plt.subplots(1, 2, figsize=(18, 9))

  # Карта существующих остановок.
edges.plot(ax=ax[0], linewidth=0.1, color='gray', alpha=0.7, zorder=1)
bus_stop_points.plot(ax=ax[0], color='blue', markersize=7, alpha=0.7, label="Существующие остановки", zorder=2)
ax[0].set_title("Существующие автобусные остановки", fontsize=14)
ax[0].legend(loc='upper right')

  # Карта оптимизированных остановок.
edges.plot(ax=ax[1], linewidth=0.1, color='gray', alpha=0.7, zorder=1)

  # Визуализируем разные типы остановок разными цветами.
color_dict = {'connection': 'red', 'key': 'orange', 'ordinary': 'green'}
size_dict = {'connection': 10, 'key': 8, 'ordinary': 6}

if len(opt_stops) > 0:
    opt_stops['color'] = opt_stops['type'].map(color_dict)
    opt_stops['markersize'] = opt_stops['type'].map(size_dict)

  # Рисуем остановки с настроенными параметрами.
    for stop_type in color_dict.keys():
        if stop_type in opt_stops['type'].values:
            stops_of_type = opt_stops[opt_stops['type'] == stop_type]
            stops_of_type.plot(ax=ax[1], color=color_dict[stop_type], markersize=size_dict[stop_type],
                               alpha=0.8, zorder=2, label=f"{stop_type.capitalize()} stops")

ax[1].set_title("Оптимизированные автобусные остановки", fontsize=14)

  # Создаем правильную легенду для второго графика.
from matplotlib.lines import Line2D

custom_lines = [Line2D([0], [0], color=color_dict[t], marker='o', linestyle='None',
                       markersize=size_dict[t], label=f"{t.capitalize()} stops")
                for t in color_dict.keys()]
ax[1].legend(handles=custom_lines, loc='upper right')

plt.tight_layout()
plt.savefig('bus_stops_comparison.png', dpi=300)
plt.show()

def coverage_stats(stops, buf_range_m=[200, 300, 400, 500, 600]):
    if len(stops) == 0:
  # Возвращаем нули если нет остановок.
        return pd.DataFrame([{'radius': r, 'population': 0, 'jobs': 0, 'area': 0} for r in buf_range_m])

    res = []
  # CRS : epsg=32636.
    coords_stops = np.array(list(zip(stops.geometry.x, stops.geometry.y)))
    coords_cands = np.array(list(zip(candidates.geometry.x, candidates.geometry.y)))
    from scipy.spatial.distance import cdist
    D = cdist(coords_cands, coords_stops)  # В метрах.

    for r in buf_range_m:
        within_r = (D < r).any(axis=1)
        pop_cov = candidates['population'][within_r].sum() / candidates['population'].sum() if candidates[
                                                                                                   'population'].sum() > 0 else 0
        job_cov = candidates['jobs'][within_r].sum() / candidates['jobs'].sum() if candidates['jobs'].sum() > 0 else 0
        area_cov = within_r.sum() / len(candidates)
        res.append({'radius': r, 'population': pop_cov, 'jobs': job_cov, 'area': area_cov})

    return pd.DataFrame(res)

stats_exist = coverage_stats(bus_stops_for_analysis)
stats_opt = coverage_stats(opt_stops.to_crs(epsg=32636))

  # Улучшенная визуализация метрик покрытия.
plt.figure(figsize=(10, 6))
plt.plot(stats_exist['radius'], stats_exist['population'] * 100, 'b-', linewidth=2, label="Существующие - Население")
plt.plot(stats_exist['radius'], stats_exist['jobs'] * 100, 'b--', linewidth=2, label="Существующие - Рабочие места")
  # Plt.plot(stats_opt['radius'], stats_opt['population']**16 * 100, 'r-', linewidth=2, label="Оптимизированные - Население").
plt.plot(stats_opt['radius'], stats_opt['population'] * 100, 'r-', linewidth=2, label="Оптимизированные - Население")
plt.plot(stats_opt['radius'], stats_opt['jobs'] * 100, 'r--', linewidth=2, label="Оптимизированные - Рабочие места")
plt.plot(stats_exist['radius'], stats_exist['area'] * 100, 'b:', linewidth=1, label="Существующие - Площадь")
plt.plot(stats_opt['radius'], stats_opt['area'] * 100, 'r:', linewidth=1, label="Оптимизированные - Площадь")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel("Радиус покрытия (м)", fontsize=12)
plt.ylabel("Покрытие (%)", fontsize=12)
plt.legend(loc='lower right')
plt.title("Сравнение статистики покрытия", fontsize=14)
plt.tight_layout()
plt.savefig('coverage_stats.png', dpi=300)
plt.show()

def stop_dist_hist(stops):
    if len(stops) <= 1:
        return np.array([0])  # Если остановка только одна, возвращаем массив с нулем.

    coords = np.array(list(zip(stops.geometry.x, stops.geometry.y)))
    tree = KDTree(coords)
    dists, _ = tree.query(coords, k=2)  # Next nearest neighbour.
  # K=1 is distance 0 (itself), so take k=2.
    return dists[:, 1]

  # Обработка случая с пустыми данными.
if len(bus_stops_for_analysis) > 1:
    ds_existing = stop_dist_hist(bus_stops_for_analysis)
else:
    ds_existing = np.array([0])

if len(opt_stops) > 1:
    ds_opt = stop_dist_hist(opt_stops)
else:
    ds_opt = np.array([0])

  # Улучшенная визуализация гистограммы расстояний.
plt.figure(figsize=(10, 5))
bins = np.arange(0, 2000, 100)  # Расстояния в метрах.
plt.hist(ds_existing, bins=bins, alpha=0.6, label="Существующие", color='blue', edgecolor='black')
plt.hist(ds_opt, bins=bins, alpha=0.6, label="Оптимизированные", color='red', edgecolor='black')
plt.xlabel("Расстояние до ближайшей остановки (м)", fontsize=12)
plt.ylabel("Частота", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=10)
plt.title("Распределение расстояний между остановками", fontsize=14)
plt.tight_layout()
plt.savefig('stop_distances.png', dpi=300)
plt.show()

total_stops = len(connection_stops) + len(key_stop_idxs) + len(ordinary_stop_geoms)
logger.info(f"Всего остановок: {total_stops} (максимум {max_optimized_stops})")

  # Проверка, что не превышаем максимум.
if total_stops > max_optimized_stops:
    logger.info("ВНИМАНИЕ: общее количество остановок превысило максимальное!")
else:
    logger.info("Ограничение на максимальное количество остановок соблюдено.")

  # Дополнительные статистики.
logger.info("\nДополнительная статистика:")
logger.info(f"Среднее расстояние между существующими остановками: {ds_existing.mean():.1f} м")
logger.info(f"Среднее расстояние между оптимизированными остановками: {ds_opt.mean():.1f} м")

  # Проверяем, есть ли данные для радиуса 400м.
if 400 in stats_exist['radius'].values and 400 in stats_opt['radius'].values:
    logger.info(
        f"Покрытие населения в радиусе 400м (существующие): {stats_exist.loc[stats_exist['radius'] == 400, 'population'].values[0] * 100:.1f}%")
    logger.info(
        f"Покрытие населения в радиусе 400м (оптимизированные): {stats_opt.loc[stats_opt['radius'] == 400, 'population'].values[0] * 100:.1f}%")
    logger.info(
        f"Покрытие рабочих мест в радиусе 400м (существующие): {stats_exist.loc[stats_exist['radius'] == 400, 'jobs'].values[0] * 100:.1f}%")
    logger.info(
        f"Покрытие рабочих мест в радиусе 400м (оптимизированные): {stats_opt.loc[stats_opt['radius'] == 400, 'jobs'].values[0] * 100:.1f}%")

  # Создаем геодатафрейм с оптимизированными остановками.
if len(opt_stops_parts) > 0:
    opt_stops = pd.concat(opt_stops_parts)
else:
    opt_stops = gpd.GeoDataFrame(columns=candidates.columns)
    opt_stops['type'] = []

  # Сохраняем в формате pickle для последующего использования.
with open('opt_stops.pkl', 'wb') as f:
    pickle.dump(opt_stops, f)

logger.info("Оптимизированные остановки сохранены в файл opt_stops.pkl для использования в алгоритме трассировки")