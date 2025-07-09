import os
from sarenv import (
    CLIMATE_TEMPERATE,
    CLIMATE_DRY,
    ENVIRONMENT_TYPE_FLAT,
    ENVIRONMENT_TYPE_MOUNTAINOUS,
    DataGenerator,
    get_logger,
)

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sarenv import (
    DatasetLoader,
    SARDatasetItem,
    get_logger,
)
from sarenv.utils.geo import get_utm_epsg
from sarenv.utils.plot import DEFAULT_COLOR, FEATURE_COLOR_MAP
from shapely.geometry import Point
from sarenv.utils.lost_person_behavior import get_environment_radius
from sarenv import visualize_features, visualize_heatmap

log = get_logger()

# Your points data: (latitude, longitude, climate, environment_type)
# We'll use longitude, latitude order for center_point as in your example (x, y)
points = [
    # Flat points
    #A lot of features in these landscapes
    (1, -2.704825, 51.117314, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Glastonbuy, UK
    (2, 11.558208, 55.360132, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Suserup, DK
    (3, 12.5036994, 51.1341115, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Leipzig, DE
    (4, 11.863719, 53.629115, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Hohen Pritz, DE
    (5, 14.846426, 49.781765, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Litichovice, CZ
    # Dominant Features:
    (6, 4.587955, 49.341530, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Orfeuil, FR, Fields dominated
    (7, 23.758429, 52.668738, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Podcerkwy, PL, Forests dominated
    (8, 4.864062, 52.829158, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Wieringerwaard, NL, Roads dominated
    (9, 1.989096, 47.461731, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Salbris, FR,  Forests dominated
    (10, 18.604416,51.651660, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Jakubice, PL, Fields dominated
    # Water dominated
    (11, 13.563963, 54.538038, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Binz, DE
    (12, 2.824659, 51.176157, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Middelkerke, BE
    (13, 11.787275, 54.916919, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Guldborg, DK
    (14, -2.380309, 50.641932, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Osmington, UK
    (15, -1.381205, 43.851981, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_FLAT), #Messanges, FR

    # Mountainous points
    # A lot of features in these landscapes
    (16, 9.838304, 46.826512, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Davos, CH
    (17, 6.415801, 45.702916, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Le grand village, FR
    (18, -3.374092, 51.852928, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Merthyr Tydfil, UK
    (19, 2.739522, 45.599168, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Murat-le-Quaire, FR
    (20, 12.820894, 47.229553, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Fusch an der Großglocknerstraße, AU
    # Dominant Features:
    (21, 11.629584, 46.533445, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Ortisei, IT, Meadows dominated
    (22, -3.069811, 54.467693, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Lake District, UK, Meadows dominated
    (23, 8.482994, 46.588844, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Realp, CH, Roads, Meadows
    (24, 26.374320, 45.671357, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Gura Teghii, RO, Forests dominated
    (25, 26.516688, 46.109879,  CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Mănăstirea Cașin, Forests dominated
    # Water dominated
    (26, 6.656861, 60.338501, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Utnes, NO
    (27, -5.517658, 57.569832, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Torridon, UK
    (28, -7.19519, 62.153469, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Vesmanna, FO
    (29, -23.1622,  66.103296, CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Hnifsdalur, IS
    (30, -3.79075, 54.887314,  CLIMATE_TEMPERATE, ENVIRONMENT_TYPE_MOUNTAINOUS), #Rockcliffe, UK

    #Flat points in dry climates
    #A lot of features in these landscapes
    (31, -0.128459, 41.522016, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Valfarta, ES
    (32, -0.917384, 37.764843, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Torre Pacheco, ES
    (33, -6.169819, 38.355880, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Usagre, ES
    (34, -5.686065, 37.352518, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Casablanca, ES
    (35, -2.105520, 39.094056, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Albacete, ES
    # Dominant Features:
    (36, 26.738899,40.507894,  CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Bolayır, TR, Fields dominated
    (37, -1.008155,37.732251, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Los Carriones, ES, Fields dominated
    (38, -6.309778,37.208551, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Parque Nacional de Doñana, ES, Forests dominated
    (39, -5.022312,41.018847,  CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Mamblas, ES, Fields dominated
    (40, -4.997156,41.213098,  CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Velascálvaro, ES, Fields dominated
    #Water dominated
    (41, -2.960160, 39.041375,  CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Argamasilla de Alba, ES
    (42, -6.839751, 37.226609, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Huelva, ES
    (43, -0.712404, 38.190567, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #el Derramador, ES
    (44, 15.087337, 36.698295, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Maucini, IT
    (45, -0.895824, 37.732841, CLIMATE_DRY, ENVIRONMENT_TYPE_FLAT), #Los Alcázares, ES
    #Mountainous points in dry climates
    #A lot of features in these landscapes
    (46, -3.453313, 37.140462, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Pinos Genil, ES
    (47, -2.748850, 38.121628, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Santiago Pontones, ES
    (48, -1.898658, 38.378261,  CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Las Cobatillas, ES
    (49, -4.557149, 37.355200,  CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Lucena, ES
    (50, -0.176962, 40.401785, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS),  #Benasal, ES
    # Dominant Features:
    (51, 8.900381, 39.053632, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Pula, IT, Forests dominated
    (52, 2.921109, 39.859034, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Escorca, ES, Rocks Dominated
    (53, -5.050116, 40.387466, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Hoyos de Miguel, ES
    (54, -2.296073, 38.552255, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Molinicos, ES
    (55, -3.038068, 39.544426, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Quintanar de la Sierra, ES
    #Water dominated
    (56, 15.917733, 38.007144, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Condofuri, IT
    (57, 14.421495, 37.030572, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Acate, IT
    (58, 1.8260448, 41.366684, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Subirats, ES
    (59, -3.436059, 36.889949, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Orgiva, ES
    (60,  8.895671, 38.951591, CLIMATE_DRY, ENVIRONMENT_TYPE_MOUNTAINOUS), #Domus de Maria, IT
]

def generate_all():
    data_gen = DataGenerator()
    base_output_dir = "sarenv_dataset"

    size_to_load = "xlarge"
    for id, lon, lat, climate, env_type in points:
        out_dir = os.path.join(base_output_dir, str(id))

        os.makedirs(out_dir, exist_ok=True)
        log.info(f"Generating dataset for point ({lat}, {lon}) at {out_dir}")

        data_gen.export_dataset(
            center_point=(lon, lat),
            output_directory=out_dir,
            environment_climate=climate,
            environment_type=env_type,
            meter_per_bin=30,
        )

    for id, lon, lat, climate, env_type in points:
        out_dir = os.path.join(base_output_dir, str(id))

        os.makedirs(out_dir, exist_ok=True)
        log.info(f"Generating dataset for point ({lat}, {lon}) at {out_dir}")

        try:
            # Initialize the new DynamicDatasetLoader
            loader = DatasetLoader(dataset_directory=out_dir)

            log.info(f"Loading data for size: '{size_to_load}'")
            item = loader.load_environment(size_to_load)

            if item:
                # Call the new all-in-one visualization function
                visualize_features(item,False, False, 0, False)
                plt.savefig(os.path.join(out_dir, f"features_{item.size}.png"))

                visualize_heatmap(item, False, False,False)
                plt.savefig(os.path.join(out_dir, f"heatmap_{item.size}.png"))
            else:
                log.error(f"Could not load the specified size: '{size_to_load}'")

        except FileNotFoundError:
            log.error(
                f"Error: The dataset directory '{out_dir}' or its master files were not found."
            )
            log.error(
                "Please run the `export_dataset()` method from the DataGenerator first."
            )
        except Exception as e:
            log.error(f"An unexpected error occurred: {e}", exc_info=True)



if __name__ == "__main__":
    generate_all()
