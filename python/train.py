import json
from datetime import datetime as dt
from predictionutils.pipeline import Pipeline



if __name__ == '__main__':
    pipeline = Pipeline()
    run = pipeline.run()
    run_metrics = dict([(k, str(v)) for k, v in run[0].items()])
    print(run_metrics)
    run_date = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')
    filename = f"/Users/mct19/repos/ENM-Protein-Predictor/Output_Files/pipeline-out-{run_date}.json"
    json = json.dumps(run_metrics)
    with open(filename, 'w+') as f:
        f.write(json)
