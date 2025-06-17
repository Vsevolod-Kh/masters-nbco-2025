import logging
logger = logging.getLogger(__name__)

import osmnx as ox

  # Import seaborn as sns.
  # Import requests.
  # Import json.
  # Import math.
  # From scipy.stats import gaussian_kde.
  # From tqdm import tqdm.
import pickle
import networkx as nx
from shapely.geometry import Point, LineString, Point, Polygon
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from scipy.spatial import cKDTree
import geopandas as gpd

logger.info(1)
  # Получаем administrative polygons для районов.
city = 'Saint Petersburg, Russia'
admin_tags = {'boundary': 'administrative', 'admin_level':'9'}
districts = ox.features_from_place(city, admin_tags)
districts = districts[districts.geometry.type.isin(['Polygon', 'MultiPolygon'])]
districts = districts.to_crs(3857)
districts = districts.reset_index(drop=True)
districts = districts[['geometry','name']]

  # Скачайте население районов вручную (например, по данным администрации города) и сохраните в файле district_population.csv.
  # Формат: columns=['district_name','population'].

  # Пример подгрузки вручную:.
import warnings
warnings.filterwarnings('ignore')
district_population = pd.read_csv('district_population.csv')  # Имя колонки должно совпадать с названием района в districts.
  # Присоединяем население к районам через название (понадобится привести названия к совпадающим).
districts = districts.merge(district_population, left_on='name', right_on='district_name', how='right').drop_duplicates()
  # Скачиваем здания OSM с тегом "building=residential".
tags = {'building': ['residential', 'apartments', 'house', 'detached', 'terrace', 'dormitory', 'bungalow',
                    'cabin', 'farm']}
buildings = ox.features_from_place(city, tags)
buildings = buildings[buildings.geometry.type.isin(['Polygon', 'MultiPolygon'])]
buildings = buildings.to_crs(3857)
buildings = buildings.reset_index(drop=True)
logger.info(2)
  # Для каждого здания попробуем взять число квартир.
  # В OSM встречаются теги 'apartments', 'flats', 'rooms'.

  # Если flats (квартиры) нет — считаем flats = площадь_здания / средняя площадь кв.
buildings['levels'] = pd.to_numeric(buildings['building:levels'], errors='coerce').fillna(2)
buildings['area'] = buildings.geometry.area * buildings['levels']
mean_flat_area = 38  # М2 (можно уточнить).
buildings['flats'] = (buildings['area'] / mean_flat_area).round(0)

  # Определяем район для каждого здания (spatial join).
build_pts = buildings.copy()
build_pts['geometry'] = buildings.geometry.centroid
build_pts = gpd.sjoin(build_pts, districts[['geometry','name','population']], how='left', predicate='within')
logger.info(3)
  # Суммарное число квартир по району.
flats_sum = build_pts.groupby('name_right')['flats'].sum().reset_index().rename(columns={'flats':'flats_sum'})
build_pts = build_pts.merge(flats_sum, on='name_right')

  # Среднее население на квартиру.
build_pts['people_per_flat'] = build_pts['population'] / build_pts['flats_sum']
build_pts['pop_est'] = (build_pts['people_per_flat'] * build_pts['flats']).round(0).astype(int)

  # Пример Overpass для офисных и промышленных зданий ("office", "business", "industrial").
  # Используем osmnx.
tags_work = {'office':True, 'industrial':True, 'commercial':True, 'retail':True,'shop':True}
jobs_buildings = ox.features_from_place(city, tags_work)
jobs_buildings = jobs_buildings[jobs_buildings.geometry.type.isin(['Polygon','MultiPolygon','Point'])]
jobs_buildings = jobs_buildings.to_crs(3857)
logger.info(4)
  # Будет большое количество зданий; население рабочих мест будем считать по усреднённым нормам:.
  # ~10-15 работников на 1 офис, ~1 работник на 15-30 м2, смотря по типу помещения.
def estimate_employees(row):
    if hasattr(row, 'area'):
        if 'office' in str(row):  # Грубая оценка.
            return row.area/10
        elif 'industrial' in str(row):
            return row.area/40
        elif 'retail' in str(row) or 'commercial' in str(row):
            return row.area/6
    return 3  # Fallback.
jobs_buildings['levels'] = pd.to_numeric(jobs_buildings['building:levels'], errors='coerce').fillna(2)
jobs_buildings['area'] = jobs_buildings.area if 'area' in jobs_buildings.columns else jobs_buildings.geometry.area  * buildings['levels']
jobs_buildings['employees'] = jobs_buildings.apply(estimate_employees, axis=1).astype(int)
jobs_pts = jobs_buildings.copy()
jobs_pts['geometry'] = jobs_buildings.centroid
jobs_pts = jobs_pts[jobs_pts['employees'] > 0]
logger.info(5)
poi_tags = {
    'amenity': ['school', 'hospital', 'kindergarten', 'university', 'public', 'train_station', 'retirement_home',
                    'stadium', 'sports_hall', 'college'],
    'shop': 'mall',
    'railway': 'station'
}
  # Скачиваем POI как точки.
poi_df = ox.features_from_place(city, tags=poi_tags)
poi_df = poi_df.to_crs(3857)
poi_df = poi_df[poi_df.geometry.type == 'Point']
logger.info(6)
  # Метро (станции) — обычно railway=subway_entrance или station.
metro_tags = {'railway': 'station', 'station':'subway'}
metros = ox.features_from_place(city, metro_tags)
metros = metros[metros.geometry.type == 'Point'].to_crs(3857)

  # Объединяем все точки (buildings+jobs+POI).
demand_pts = pd.concat([
    build_pts[['geometry', 'pop_est']].rename(columns={'pop_est':'demand'}),
    jobs_pts[['geometry','employees']].rename(columns={'employees':'demand'}),
    poi_df[['geometry']].assign(demand=200),  # Пример для POI (считать коэффициенты вручную).
    metros[['geometry']].assign(demand=200)
]).reset_index(drop=True)
demand_pts = gpd.GeoDataFrame(demand_pts, crs=3857)
logger.info(7)
"""fig, ax = plt.subplots(figsize=(14, 10))
  # В строении KDE используем все точки с весом demand.
sns.kdeplot(
    x=demand_pts.geometry.x, y=demand_pts.geometry.y,
    weights=demand_pts.demand, cmap='magma', fill=True, thresh=0, levels=100, ax=ax, bw_adjust=0.5, alpha = 0.5)
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=demand_pts.crs)
plt.title("Теплокарта спроса на ОТ в Санкт-Петербурге")
plt.show()"""

tags_stops = {'highway':'bus_stop'}
bus_stops = ox.features_from_place(city, tags=tags_stops)
bus_stops = bus_stops[bus_stops.geometry.type=='Point'].to_crs(3857)

  # Для "предложения" — считаем простое покрытие: каждый demand_pt ищет ближайшую остановку.
tree = cKDTree(np.vstack([bus_stops.geometry.x, bus_stops.geometry.y]).T)
distances, idx = tree.query(np.vstack([demand_pts.geometry.x, demand_pts.geometry.y]).T, k=1)
demand_pts['nearest_stop_dist'] = distances

  # Определяем покрытие (например, удовлетворенность = 1 если <400м, экспоненциально убывает дальше).
ALPHA = 0.004
demand_pts['P_access'] = np.exp(-ALPHA * demand_pts['nearest_stop_dist'])

demand_pts['unsatisfied'] = demand_pts['demand'] * (1 - demand_pts['P_access'])

x_min = min(demand_pts.geometry.x.min(), bus_stops.geometry.x.min())
x_max = max(demand_pts.geometry.x.max(), bus_stops.geometry.x.max())
y_min = min(demand_pts.geometry.y.min(), bus_stops.geometry.y.min())
y_max = max(demand_pts.geometry.y.max(), bus_stops.geometry.y.max())

  # ! ОСНОВНОЕ: HEXBIN по СПРОСУ !
fig, ax = plt.subplots(figsize=(14,10))
hb = ax.hexbin(
    demand_pts.geometry.x, demand_pts.geometry.y,
    C=demand_pts.demand, gridsize=80, reduce_C_function=np.sum,
    cmap='Oranges', mincnt=1
)
plt.colorbar(hb, ax=ax, label='Суммарный спрос по hex')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=demand_pts.crs)
plt.title('Hexbin по суммарному спросу')
plt.axis('off')
plt.tight_layout()
plt.show()

  # HEXBIN по предложению (число остановок на hex).
fig, ax = plt.subplots(figsize=(14,10))
hb = ax.hexbin(
    bus_stops.geometry.x, bus_stops.geometry.y,
    gridsize=80, cmap='Blues', mincnt=1  # Важно: просто считаем точки.
)
plt.colorbar(hb, ax=ax, label='Число остановок ОТ по hex')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=bus_stops.crs)
plt.title('Hexbin по предложению (остановки ОТ)')
plt.axis('off')
plt.tight_layout()
plt.show()

  # ====== Карта баланса: разница (остановки - спрос в hex'ах) =======.
logger.info("Создание карты дисбалансов в формате шестиугольников...")

  # 1. Получение центров hex-решеток.
hx = plt.hexbin(
    demand_pts.geometry.x, demand_pts.geometry.y,
    C=demand_pts.demand, gridsize=80, reduce_C_function=np.sum,
    cmap='Oranges', mincnt=1
)
plt.close()  # Чтобы не показывать промежуточный график.
hex_centers = hx.get_offsets()  # Shape (N_hex, 2).
hex_squares = hx.get_array()  # Суммарный спрос в каждом hex.

  # 2. Для каждого hex считаем кол-во остановок (из "предложения") в +/-epsilon от центра hex.
tree_stops = cKDTree(np.vstack([bus_stops.geometry.x, bus_stops.geometry.y]).T)
proposal_per_hex = []
hex_radius = (x_max - x_min) / 80 * 1.2  # Подгоним радиус поиска под размер hex'а.

for center in hex_centers:
    nstops = len(tree_stops.query_ball_point(center, hex_radius))
    proposal_per_hex.append(nstops)

proposal_per_hex = np.array(proposal_per_hex)

  # 3. Баланс: предложение минус спрос (нормированный, чтобы числа были сопоставимы).
balance_per_hex = proposal_per_hex - (hex_squares / (demand_pts.demand.mean() + 1e-5))

  # 4. ФИЛЬТРАЦИЯ сильных дисбалансов - показываем только значения больше порога по модулю.
threshold = 90  # Задаем пороговое значение.
mask_significant = (balance_per_hex > threshold) | (balance_per_hex < -threshold)

  # Создаем отфильтрованные массивы с центрами и значениями дисбаланса.
filtered_centers = hex_centers[mask_significant]
filtered_balance = balance_per_hex[mask_significant]

logger.info(len(filtered_centers))

  # 5. Визуализация баланса в формате hexbin только для значимых значений.
fig, ax = plt.subplots(figsize=(14, 10))

  # Сначала рисуем все шестиугольники серым цветом для фона.
hb_background = ax.hexbin(
    hex_centers[:, 0], hex_centers[:, 1],
    C=np.ones_like(balance_per_hex), gridsize=80,
    cmap='Greys', alpha=0.1, mincnt=1
)

  # Затем рисуем только значимые шестиугольники с цветовой схемой.
if len(filtered_centers) > 0:  # Проверяем, что есть значимые значения.
    hb = ax.hexbin(
        filtered_centers[:, 0], filtered_centers[:, 1],
        C=filtered_balance, gridsize=80, cmap='coolwarm_r',
        vmin=-max(abs(filtered_balance.min()), abs(filtered_balance.max())),
        vmax=max(abs(filtered_balance.min()), abs(filtered_balance.max())),
        mincnt=1
    )
    plt.colorbar(hb, ax=ax, label=f'Дисбаланс (синий: недостаток, красный: избыток)\nПоказаны только значения > |{threshold}|')
else:
    logger.info("Нет значений дисбаланса, превышающих порог по модулю:", threshold)

ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=demand_pts.crs)
plt.title(f'Карта дисбалансов (шестиугольники) - только значения |дисбаланс| > {threshold}', fontsize=15)
plt.axis('off')
plt.tight_layout()
plt.savefig("balance_hexagons.png", dpi=300, bbox_inches='tight')
plt.show()

logger.info("Карта дисбалансов сохранена в файл balance_hexagons.png")

logger.info("Запуск кластеризации по непрерывным шестиугольникам дисбаланса...")

  # Создаем директорию для сохранения результатов.
import os

if not os.path.exists("results"):
    os.makedirs("results")
    logger.info("Создана директория 'results' для сохранения результатов")

  # Используем filtered_centers - уже отфильтрованные шестиугольники с сильным дисбалансом.
  # Фильтруем только шестиугольники с недостатком остановок (синие).
mask_negative = filtered_balance < 0
hex_demand = filtered_centers[mask_negative]
hex_demand_balance = filtered_balance[mask_negative]
logger.info(f"Найдено {len(hex_demand)} шестиугольников с недостатком остановок")

if len(hex_demand) > 0:
    logger.info("Поиск групп соседних шестиугольников...")

  # Определяем точный размер шестиугольника как в hexbin.
    x_range = x_max - x_min
    y_range = y_max - y_min
    hex_spacing_x = x_range / 80
    hex_spacing_y = y_range / (80 * np.sqrt(3) / 2)
    hex_radius_actual = hex_spacing_x / np.sqrt(3)

    logger.info(f"Используем радиус шестиугольника: {hex_radius_actual:.2f}")

  # Используем KDTree для определения соседства.
    from scipy.spatial import KDTree

    tree = KDTree(hex_demand)
    neighbor_distance = hex_spacing_x * 1.1  # Небольшой запас для погрешностей.

  # Создаем граф смежности.
    G = nx.Graph()

  # Добавляем узлы.
    for i in range(len(hex_demand)):
        G.add_node(i)

  # Находим все пары соседних шестиугольников.
    pairs = list(tree.query_pairs(neighbor_distance))
    logger.info(f"Найдено {len(pairs)} пар соседних шестиугольников")

  # Добавляем ребра между соседними шестиугольниками.
    for i, j in pairs:
        G.add_edge(i, j)

  # Находим компоненты связности (кластеры).
    connected_components = list(nx.connected_components(G))
    logger.info(f"Найдено {len(connected_components)} соединенных групп шестиугольников")

    if connected_components:
  # ИСПРАВЛЕНО: Фильтруем кластеры по размеру, плотности И компактности.
        min_cluster_size = 5  # Минимальное количество шестиугольников в кластере.
        significant_clusters = []

        for component in connected_components:
            if len(component) >= min_cluster_size:
  # Проверяем плотность кластера.
                cluster_coords = hex_demand[list(component)]

  # Вычисляем компактность кластера.
                centroid = np.mean(cluster_coords, axis=0)
                distances = np.sqrt(np.sum((cluster_coords - centroid) ** 2, axis=1))
                max_distance = np.max(distances)
                avg_distance = np.mean(distances)

  # НОВОЕ: Проверяем форму кластера (не слишком вытянутый).
  # Вычисляем главные компоненты для определения вытянутости.
                from sklearn.decomposition import PCA

                if len(cluster_coords) >= 3:  # Нужно минимум 3 точки для PCA.
                    pca = PCA(n_components=2)
                    pca.fit(cluster_coords)

  # Отношение главных компонент показывает вытянутость.
                    eigenvalues = pca.explained_variance_
                    aspect_ratio = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else float('inf')

  # Критерий компактности: отношение не должно быть слишком большим.
                    max_aspect_ratio = 3.0  # Максимальное отношение длины к ширине.
                    is_compact = aspect_ratio <= max_aspect_ratio
                else:
  # Для маленьких кластеров считаем компактными.
                    aspect_ratio = 1.0
                    is_compact = True

  # Критерий плотности: средняя дистанция не должна быть слишком большой.
                density_threshold = hex_spacing_x * 3  # Максимальная средняя дистанция от центра.
                is_dense = avg_distance <= density_threshold

  # НОВОЕ: Дополнительная проверка на "круглость" через выпуклую оболочку.
                from scipy.spatial import ConvexHull

                if len(cluster_coords) >= 3:
                    try:
                        hull = ConvexHull(cluster_coords)
                        hull_area = hull.volume  # В 2D это площадь.

  # Площадь окружности с тем же "радиусом" (max_distance).
                        circle_area = np.pi * max_distance ** 2

  # Коэффициент "круглости": отношение площади выпуклой оболочки к площади окружности.
                        roundness = hull_area / circle_area if circle_area > 0 else 0
                        min_roundness = 0.3  # Минимальная "круглость".
                        is_round = roundness >= min_roundness
                    except:
  # Если не удалось построить выпуклую оболочку.
                        roundness = 1.0
                        is_round = True
                else:
                    roundness = 1.0
                    is_round = True

  # Кластер принимается, если он плотный, компактный и достаточно круглый.
                if is_dense and is_compact and is_round:
                    significant_clusters.append(component)
                    logger.info(f" Кластер размером {len(component)}: "
                          f"средняя дистанция {avg_distance:.1f}м, "
                          f"вытянутость {aspect_ratio:.2f}, "
                          f"круглость {roundness:.2f}")
                else:
                    logger.info(f" Отклонен кластер размером {len(component)}: "
                          f"плотный={is_dense}, "
                          f"компактный={is_compact} (вытянутость {aspect_ratio:.2f}), "
                          f"круглый={is_round} (круглость {roundness:.2f})")

        logger.info(f"Найдено {len(significant_clusters)} компактных кластеров")

        if significant_clusters:
  # Сортируем кластеры по размеру.
            significant_clusters = sorted(significant_clusters, key=len, reverse=True)
            largest_cluster = list(significant_clusters[0])
            logger.info(f"Самый большой компактный кластер содержит {len(largest_cluster)} шестиугольников")

  # Получаем координаты центров шестиугольников в кластере.
            largest_cluster_centers = hex_demand[largest_cluster]

  # ИСПРАВЛЕНО: Создаем геометрию кластера из шестиугольников с ОЧЕНЬ МАЛЕНЬКИМ буфером.
            from shapely.geometry import Polygon
            from shapely.ops import unary_union

  # Создаем точные шестиугольники для каждого центра в кластере.
            cluster_hexagons = []
            for center in largest_cluster_centers:
  # Создаем шестиугольник с горизонтальными сторонами.
                angles = np.linspace(0, 2 * np.pi, 7)[:-1]
                rotation = np.pi / 6  # 30 градусов для горизонтальных граней.
                hex_vertices = [(center[0] + hex_radius_actual * 0.95 * np.cos(angle + rotation),
                                 center[1] + hex_radius_actual * 0.95 * np.sin(angle + rotation))
                                for angle in angles]
                cluster_hexagons.append(Polygon(hex_vertices))

  # НОВОЕ: Объединяем шестиугольники и добавляем очень маленький буфер.
            cluster_geom_base = unary_union(cluster_hexagons)

  # Добавляем очень маленький буфер для сглаживания границ.
            small_buffer = hex_radius_actual * 0.1  # 10% от радиуса шестиугольника.
            cluster_geom = cluster_geom_base.buffer(small_buffer)

            logger.info(f"Добавлен маленький буфер {small_buffer:.1f}м для сглаживания границ кластера")

  # Визуализация кластера.
            logger.info("Визуализация самого большого компактного кластера дисбаланса...")
            fig, ax = plt.subplots(figsize=(14, 10))

  # Показываем карту дисбаланса как фон.
            hb_background = ax.hexbin(
                hex_centers[:, 0], hex_centers[:, 1],
                C=np.ones_like(balance_per_hex), gridsize=80,
                cmap='Greys', alpha=0.1, mincnt=1
            )

  # Показываем значимые шестиугольники.
            if len(filtered_centers) > 0:
                hb = ax.hexbin(
                    filtered_centers[:, 0], filtered_centers[:, 1],
                    C=filtered_balance, gridsize=80, cmap='coolwarm_r',
                    vmin=-max(abs(filtered_balance.min()), abs(filtered_balance.max())),
                    vmax=max(abs(filtered_balance.min()), abs(filtered_balance.max())),
                    mincnt=1, alpha=0.6
                )

  # ИСПРАВЛЕНО: Показываем точные границы шестиугольников кластера.
            for hexagon in cluster_hexagons:
                x, y = hexagon.exterior.xy
                ax.fill(x, y, color='blue', alpha=0.4, edgecolor='darkblue', linewidth=1)

  # Рисуем общую границу кластера с буфером.
            import geopandas as gpd

            cluster_gdf = gpd.GeoDataFrame(geometry=[cluster_geom], crs=demand_pts.crs)
            cluster_gdf.boundary.plot(ax=ax, color='blue', linewidth=3, alpha=0.9)

  # Добавляем подложку карты.
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=demand_pts.crs)
            plt.title("Самый большой компактный кластер недостатка остановок (с буфером)", fontsize=15)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig("results/largest_cluster.png", dpi=300)
            plt.show()

  # Сохраняем границы кластера.
            logger.info("Сохранение границ компактного кластера...")
            cluster_gdf.to_file("results/largest_cluster_boundary.geojson", driver="GeoJSON")

            with open("results/largest_cluster_boundary.pkl", "wb") as f:
                pickle.dump(cluster_geom, f)

            logger.info("Границы кластера сохранены в:")
            logger.info("- results/largest_cluster_boundary.geojson")
            logger.info("- results/largest_cluster_boundary.pkl")

        else:
            logger.info("Не найдено компактных кластеров достаточного размера")
    else:
        logger.info("Не найдено групп соединенных шестиугольников")
else:
    logger.info("Недостаточно шестиугольников с дефицитом остановок для поиска кластеров")

  # === ПОДСЧЕТ АВТОБУСНЫХ ОСТАНОВОК В ГРАНИЦАХ КЛАСТЕРА ===.
logger.info("\n=== АНАЛИЗ АВТОБУСНЫХ ОСТАНОВОК В ГРАНИЦАХ КЛАСТЕРА ===")

if 'cluster_geom' in locals() and cluster_geom is not None:
    logger.info("Подсчет автобусных остановок в границах кластера...")

  # Преобразуем геометрию кластера в тот же CRS, что и остановки.
    cluster_gdf_for_analysis = gpd.GeoDataFrame(geometry=[cluster_geom], crs=demand_pts.crs)
    bus_stops_for_analysis = bus_stops.to_crs(demand_pts.crs)

  # ИСПРАВЛЕНО: Фильтруем только автобусные остановки (highway=bus_stop).
  # Убеждаемся, что у нас только автобусные остановки.
    if 'highway' in bus_stops_for_analysis.columns:
        bus_stops_only = bus_stops_for_analysis[bus_stops_for_analysis['highway'] == 'bus_stop'].copy()
    else:
        bus_stops_only = bus_stops_for_analysis.copy()  # Если колонки нет, считаем что все остановки автобусные.

    logger.info(f"Всего автобусных остановок в городе: {len(bus_stops_only)}")

  # Находим автобусные остановки внутри границ кластера.
    stops_in_cluster = gpd.sjoin(bus_stops_only, cluster_gdf_for_analysis,
                                 how='inner', predicate='within')

  # Подсчитываем количество автобусных остановок.
    num_bus_stops_in_cluster = len(stops_in_cluster)
    total_bus_stops = len(bus_stops_only)

  # Вычисляем площадь кластера в км².
    cluster_area_km2 = cluster_geom.area / 1_000_000  # Переводим из м² в км².

  # Вычисляем плотность автобусных остановок.
    if cluster_area_km2 > 0:
        bus_stop_density = num_bus_stops_in_cluster / cluster_area_km2
    else:
        bus_stop_density = 0

  # Выводим результаты.
    logger.info(f" Автобусных остановок в кластере: {num_bus_stops_in_cluster}")
    logger.info(f" Всего автобусных остановок в городе: {total_bus_stops}")
    logger.info(f" Доля автобусных остановок в кластере: {num_bus_stops_in_cluster / total_bus_stops * 100:.1f}%")
    logger.info(f" Площадь кластера: {cluster_area_km2:.2f} км²")
    logger.info(f" Плотность автобусных остановок в кластере: {bus_stop_density:.1f} остановок/км²")

  # Дополнительная статистика по названиям остановок (если есть информация).
    if 'name' in stops_in_cluster.columns:
        named_stops = stops_in_cluster[stops_in_cluster['name'].notna()]
        logger.info(f" Автобусных остановок с названиями: {len(named_stops)}")

        if len(named_stops) > 0:
            logger.info("️  Примеры названий автобусных остановок в кластере:")
            for i, name in enumerate(named_stops['name'].head(5)):
                logger.info(f"   {i + 1}. {name}")

  # Анализ покрытия кластера автобусными остановками.
    if num_bus_stops_in_cluster > 0:
  # Создаем буферы вокруг автобусных остановок (радиус пешеходной доступности).
        walking_radius = 400  # Метров - стандартный радиус доступности для автобусных остановок.
        stop_buffers = stops_in_cluster.geometry.buffer(walking_radius)
        coverage_area = unary_union(stop_buffers)

  # Вычисляем пересечение с кластером.
        covered_area = cluster_geom.intersection(coverage_area)
        coverage_ratio = covered_area.area / cluster_geom.area

        logger.info(f" Покрытие кластера автобусными остановками (радиус {walking_radius}м): {coverage_ratio * 100:.1f}%")

  # Оценка качества покрытия автобусными остановками.
        if coverage_ratio > 0.8:
            coverage_quality = "Отличное"
        elif coverage_ratio > 0.6:
            coverage_quality = "Хорошее"
        elif coverage_ratio > 0.4:
            coverage_quality = "Удовлетворительное"
        else:
            coverage_quality = "Недостаточное"

        logger.info(f" Качество покрытия автобусными остановками: {coverage_quality}")

  # Рекомендации по улучшению.
        uncovered_ratio = 1 - coverage_ratio
        if uncovered_ratio > 0.2:
            estimated_new_stops = int(np.ceil(uncovered_ratio * cluster_area_km2 * 4))  # ~4 остановки на км².
            logger.info(
                f" Рекомендуется добавить примерно {estimated_new_stops} автобусных остановок для улучшения покрытия")
    else:
        logger.info("️  В кластере НЕТ автобусных остановок!")
        estimated_new_stops = int(np.ceil(cluster_area_km2 * 4))  # ~4 остановки на км².
        logger.info(f" Рекомендуется добавить примерно {estimated_new_stops} автобусных остановок")

  # Визуализация автобусных остановок в кластере.
    logger.info("Создание карты автобусных остановок в кластере...")
    fig, ax = plt.subplots(figsize=(12, 10))

  # Показываем карту дисбаланса как фон.
    hb_background = ax.hexbin(
        hex_centers[:, 0], hex_centers[:, 1],
        C=np.ones_like(balance_per_hex), gridsize=80,
        cmap='Greys', alpha=0.1, mincnt=1
    )

  # Показываем значимые шестиугольники.
    if len(filtered_centers) > 0:
        hb = ax.hexbin(
            filtered_centers[:, 0], filtered_centers[:, 1],
            C=filtered_balance, gridsize=80, cmap='coolwarm_r',
            vmin=-max(abs(filtered_balance.min()), abs(filtered_balance.max())),
            vmax=max(abs(filtered_balance.min()), abs(filtered_balance.max())),
            mincnt=1, alpha=0.4
        )

  # Выделяем границу кластера.
    cluster_gdf_for_analysis.boundary.plot(ax=ax, color='blue', linewidth=3, alpha=0.9)

  # Показываем все автобусные остановки в городе.
    bus_stops_only.plot(ax=ax, color='gray', markersize=15, alpha=0.3,
                        label=f'Все автобусные остановки ({total_bus_stops})')

  # Выделяем автобусные остановки в кластере.
    if num_bus_stops_in_cluster > 0:
        stops_in_cluster.plot(ax=ax, color='red', markersize=25, alpha=0.8,
                              label=f'Автобусные остановки в кластере ({num_bus_stops_in_cluster})')

  # Показываем зоны покрытия автобусных остановок в кластере.
        if walking_radius > 0:
            for _, stop in stops_in_cluster.iterrows():
                circle = plt.Circle((stop.geometry.x, stop.geometry.y), walking_radius,
                                    color='red', alpha=0.1, fill=True)
                ax.add_patch(circle)

  # Добавляем подложку карты.
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=demand_pts.crs)

    plt.title(
        f"Автобусные остановки в кластере дисбаланса\n{num_bus_stops_in_cluster} из {total_bus_stops} автобусных остановок",
        fontsize=14)
    plt.legend(loc='upper right')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("results/bus_stops_in_cluster.png", dpi=300)
    plt.show()

  # Сохраняем информацию об автобусных остановках в кластере.
    bus_stops_analysis = {
        'total_bus_stops_in_cluster': num_bus_stops_in_cluster,
        'total_bus_stops_city': total_bus_stops,
        'cluster_area_km2': cluster_area_km2,
        'bus_stop_density': bus_stop_density,
        'coverage_ratio': coverage_ratio if num_bus_stops_in_cluster > 0 else 0,
        'coverage_quality': coverage_quality if num_bus_stops_in_cluster > 0 else "Нет остановок",
        'recommended_new_stops': estimated_new_stops if 'estimated_new_stops' in locals() else 0
    }

  # Сохраняем в JSON для дальнейшего использования.
    import json

    with open("results/bus_stops_analysis.json", "w", encoding='utf-8') as f:
        json.dump(bus_stops_analysis, f, ensure_ascii=False, indent=2)

    logger.info("Анализ автобусных остановок сохранен в results/bus_stops_analysis.json")

else:
    logger.info("Кластер не найден или не создан. Анализ автобусных остановок невозможен.")

logger.info("=== АНАЛИЗ АВТОБУСНЫХ ОСТАНОВОК ЗАВЕРШЕН ===")

