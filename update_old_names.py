import os


classes = {"cold_ocean": "aquatic", "deep_cold_ocean": "aquatic", "deep_frozen_ocean": "aquatic",
           "deep_lukewarm_ocean": "aquatic", "deep_ocean": "aquatic", "lukewarm_ocean": "aquatic",
           "ocean": "aquatic", "river": "aquatic", "warm_ocean": "aquatic", "ice_spikes": "snowy",
           "frozen_ocean": "snowy", "frozen_peaks": "snowy", "frozen_river": "snowy", "grove": "snowy",
           "jagged_peaks": "snowy", "snowy_beach": "snowy", "snowy_plains": "snowy", "snowy_slopes": "snowy",
           "snowy_taiga": "snowy", "badlands": "arid", "beach": "arid", "desert": "arid",
           "eroded_badlands": "arid", "savanna": "arid", "savanna_plateau": "arid", "windswept_savanna": "arid",
           "wooded_badlands": "arid", "bamboo_jungle": "forest", "birch_forest": "forest",
           "dark_forest": "forest", "flower_forest": "forest", "forest": "forest", "jungle": "forest",
           "mangrove_swamp": "forest", "mushroom_fields": "forest", "old_growth_birch_forest": "forest",
           "old_growth_pine_taiga": "forest", "old_growth_spruce_taiga": "forest", "sparse_jungle": "forest",
           "swamp": "forest", "taiga": "forest", "windswept_forest": "forest", "meadow": "plains",
           "plains": "plains", "stony_peaks": "plains", "stony_shore": "plains", "sunflower_plains": "plains",
           "windswept_gravelly_hills": "plains", "windswept_hills": "plains"}


# Iterate directory
def update_prefix(dir_path):
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            old_file = path
            path = path.split("-")
            if path[0] in classes.keys():
                path[0] = classes.get(path[0])
                new_file = "-".join(path)
                os.rename(dir_path + old_file, dir_path + new_file)


update_prefix(r'data/test/')
update_prefix(r'data/train/')
