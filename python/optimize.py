from datetime import datetime as dt
import sys

from predictionutils.pipeline import Pipeline


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_modifier = sys.argv[1]
    else:
        path_modifier = ""
    config_path = f'/Users/mct19/repos/ENM-Protein-Predictor/config/config-optimize{path_modifier}.yml'
    pipeline = Pipeline(config_path=config_path)
    run_date = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')
    pipeline.optimize(f'/Users/mct19/repos/ENM-Protein-Predictor/Output_Files/best-params{path_modifier}-{run_date}.yml')
