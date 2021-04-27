import json
import sys
from datetime import datetime as dt
from predictionutils.pipeline import Pipeline


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_modifier = sys.argv[1]
    else:
        path_modifier = ""
    default_config_path = f'/Users/mct19/repos/ENM-Protein-Predictor/config/config{path_modifier}.yml'
    pipeline = Pipeline()
    run = pipeline.train()
    run_metrics = dict([(k, str(v)) for k, v in run[0].items()])
    print(run_metrics)
    run_date = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')
    filename = f"/Users/mct19/repos/ENM-Protein-Predictor/Output_Files/pipeline-out{path_modifier}-{run_date}.json"
    json = json.dumps(run_metrics)
    with open(filename, 'w+') as f:
        f.write(json)
