import sys
from datetime import datetime as dt
from predictionutils.pipeline import Pipeline


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_modifier = sys.argv[1]
    else:
        path_modifier = ""
    default_config_path = f'/Users/mct19/repos/ENM-Protein-Predictor/config/config-rfecv{path_modifier}.yml'
    pipeline = Pipeline(config_path=default_config_path)
    run_date = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')
    run = pipeline.rfecv(f'/Users/mct19/repos/ENM-Protein-Predictor/Input_Files/mask-{run_date}.csv')
