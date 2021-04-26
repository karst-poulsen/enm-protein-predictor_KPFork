import json
from dataclasses import dataclass
from datetime import datetime as dt
from typing import List
from yamldataclassconfig.config import YamlDataClassConfig

from predictionutils.data_utils import Database
from predictionutils.pipeline import Pipeline


default_config_path = "/Users/mct19/repos/ENM-Protein-Predictor/config/config.yml"


@dataclass
class Config(YamlDataClassConfig):
    BOOTSTRAP: bool = False
    CATEGORICAL_FIELDS: List[str] = None
    DATA_PATH: str = None
    DROP_FIELDS: List[str] = None
    ENRICHMENT_SPLIT_VALUE: int = None
    ESTIMATOR_COUNT: int = None
    ITERATIONS: int = None
    MASK_PATH: str = None
    MIN_SAMPLE_SPLIT: int = None
    NUM_JOBS: int = None
    TRAIN_PERCENTAGE: float = None

    def list_variables(self):
        return [self.BOOTSTRAP, self.CATEGORICAL_FIELDS, self.DATA_PATH, self.DROP_FIELDS,
                self.ENRICHMENT_SPLIT_VALUE, self.ESTIMATOR_COUNT, self.ITERATIONS,
                self.MASK_PATH, self.MIN_SAMPLE_SPLIT, self.NUM_JOBS, self.TRAIN_PERCENTAGE]


if __name__ == '__main__':
    CONFIGS = Config()
    CONFIGS.load(default_config_path)
    BOOTSTRAP, CATEGORICAL_FIELDS, DATA_PATH, DROP_FIELDS, ENRICHMENT_SPLIT_VALUE, ESTIMATOR_COUNT, ITERATIONS, MASK_PATH, MIN_SAMPLE_SPLIT, NUM_JOBS, TRAIN_PERCENTAGE = CONFIGS.list_variables()

    db = Database(DATA_PATH, ENRICHMENT_SPLIT_VALUE, CATEGORICAL_FIELDS, DROP_FIELDS)
    pipeline = Pipeline(BOOTSTRAP, db, ESTIMATOR_COUNT, MASK_PATH, MIN_SAMPLE_SPLIT, NUM_JOBS, TRAIN_PERCENTAGE)
    run = pipeline.run()
    run_metrics = dict([(k, str(v)) for k, v in run[0].items()])
    print(run_metrics)
    run_date = dt.strftime(dt.now(), format='%Y_%m_%dT%H:%M:%s')
    filename = f"/Users/mct19/repos/ENM-Protein-Predictor/Output_Files/pipeline-out-{run_date}.json"
    json = json.dumps(run_metrics)
    with open(filename, 'w+') as f:
        f.write(json)
