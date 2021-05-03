import sys

from predictionutils.pipeline import Pipeline


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_modifier = sys.argv[1]
    else:
        path_modifier = ""
    config_path = f'config/config-optimize{path_modifier}.yml'
    pipeline = Pipeline(config_path=config_path)
    pipeline.optimize()
