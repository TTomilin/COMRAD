from _batch_utils import generate_batch
from examples.benchmark.armory_siege import ArmorySiegeScenario


SCENARIO_CLS = ArmorySiegeScenario
BASE_NAME = "armory_siege"
OUTPUT_SUBDIR = "batch_armory_siege"
EXPECTED_COUNT = 1215
AXIS_OPTIONS = {
    "room_distance": [
        {"_label": "close", "distance": 600},
        {"_label": "mid", "distance": 700},
        {"_label": "far", "distance": 900},
    ],
    "corridor_width": [
        {"_label": "tight", "corridor_width": 2},
        {"_label": "standard", "corridor_width": 3},
        {"_label": "wide", "corridor_width": 4},
    ],
    "core_health_budget": [
        {"_label": "fragile", "core_health": 600},
        {"_label": "baseline", "core_health": 1000},
        {"_label": "durable", "core_health": 1400},
    ],
    "enemy_pressure": [
        {"_label": "light", "enemy_difficulty": 0.5},
        {"_label": "standard", "enemy_difficulty": 1.0},
        {"_label": "heavy", "enemy_difficulty": 1.5},
    ],
}


def main() -> None:
    generate_batch(
        SCENARIO_CLS,
        axis_options=AXIS_OPTIONS,
        base_name=BASE_NAME,
        output_subdir=OUTPUT_SUBDIR,
    )


if __name__ == "__main__":
    main()
