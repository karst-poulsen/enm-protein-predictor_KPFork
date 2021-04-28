import json
import sys
from datetime import datetime as dt
from predictionutils.pipeline import Pipeline


if __name__ == '__main__':
    if len(sys.argv) == 2:
        path_modifier = sys.argv[1]
    else:
        path_modifier = ""
    config_path = f'/Users/mct19/repos/ENM-Protein-Predictor/config/config-train{path_modifier}.yml'
    pipeline = Pipeline(config_path=config_path)
    run = pipeline.train()
    # run_metrics = dict([(k, str(v)) for k, v in run[0].items()])
    print(run[0])
    run_date = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')
    filename = f"/Users/mct19/repos/ENM-Protein-Predictor/Output_Files/pipeline-out{path_modifier}-{run_date}.json"
    json = json.dumps(run[0])
    with open(filename, 'w+') as f:
        f.write(json)
