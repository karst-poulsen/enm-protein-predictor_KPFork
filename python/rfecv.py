import sys
from predictionutils.pipeline import Pipeline


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_modifier = sys.argv[1]
    else:
        path_modifier = ""
    config_path = f'/Users/mct19/repos/ENM-Protein-Predictor/config/config-rfecv{path_modifier}.yml'
    pipeline = Pipeline(config_path=config_path)
    run = pipeline.rfecv()
