
from sarenv import EnvironmentBuilder, Environment

tags_mapping = {
    "structure": {
        "building": True,
        "man_made": True,
        "bridge": True,
        "tunnel": True,
    },
    "road": {"highway": True, "tracktype": True},
    "linear": {
        "railway": True,
        "barrier": True,
        "fence": True,
        "wall": True,
        "pipeline": True,
    },
    "drainage": {"waterway": ["drain", "ditch", "culvert", "canal"]},
    "water": {
        "natural": ["water", "wetland"],
        "water": True,
        "wetland": True,
        "reservoir": True,
    },
    "brush": {"landuse": ["grass"]}, # TODO check if meadow is supposed to be in this feature
    "scrub": {"natural": "scrub"},
    "woodland": {"landuse": ["forest", "wood"], "natural": "wood"},
    "field": {"landuse": ["farmland", "farm", "meadow"]},
    "rock": {"natural": ["rock", "bare_rock", "scree", "cliff"]},
}

builder = EnvironmentBuilder()
for feature, tags in tags_mapping.items():
    builder.set_feature(feature, tags)
env = builder.set_polygon_file("FlatTerrainNature.geojson").build()

env.visualise_environment()

heatmap = env.get_combined_heatmap()

env.plot_heatmap(
    heatmap,
    use_sliders=False,
    show_basemap=True,
    show_features=False,
    show_heatmap=True,
    export=False,
    show_coverage=False,
)


