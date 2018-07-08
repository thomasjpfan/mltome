import os
import csv

import numpy as np
from sacred.observers.base import RunObserver

from mltome.logging import get_log_file_handler


class CSVObserver(RunObserver):

    COLS = ['model_id', 'start_time', 'delta_time', 'train', 'valid']

    def __init__(self, results_fn):
        super().__init__()
        self.results_fn = results_fn

        if os.path.exists(self.results_fn):
            return
        with open(self.results_fn, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLS)
            writer.writeheader()

    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        if command not in ['train', 'train_hp']:
            self.record_local = False
            return
        try:
            self.model_id = config['model_id'] + '_' + command
        except KeyError:
            self.model_id = f'{_id}_{command}'

        self.start_time = start_time

        try:
            self.record_local = config['record_local']
        except KeyError:
            self.record_local = False

    def completed_event(self, stop_time, result):
        if not result or len(result) != 2 or not self.record_local:
            return
        d_time = (stop_time - self.start_time).total_seconds()
        result = {
            'model_id': self.model_id,
            'start_time': self.start_time.isoformat(),
            'delta_time': f'{d_time:.2f}',
            'train': result[1],
            'valid': result[0]
        }

        target_row = None
        rows = []
        with open(self.results_fn, 'r') as f:
            reader = csv.DictReader(f, fieldnames=self.COLS)
            next(reader)
            for i, row in enumerate(reader):
                rows.append(row)
                if row['model_id'] == self.model_id:
                    target_row = i

        if target_row is None:
            with open(self.results_fn, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=self.COLS)
                writer.writerow(result)
            return

        del rows[target_row]
        rows.append(result)
        with open(self.results_fn, 'w') as f:
            writer = csv.DictWriter(f, fieldnames=self.COLS)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


class ArtifactObserver(RunObserver):
    def __init__(self, logger):
        self.logger = logger

    def started_event(self, ex_info, command, host_info, start_time, config,
                      meta_info, _id):
        try:
            run_dir = config['run_dir']
        except KeyError:
            raise EnvironmentError(f'No run_dir defined')

        if command == 'predict' and not os.path.exists(run_dir):
            raise EnvironmentError(f'run_id must exist to predict')

        os.makedirs(run_dir, exist_ok=True)
        log_fn = os.path.join(run_dir, f'log_{command}.txt')
        file_hander = get_log_file_handler(log_fn)

        self.logger.addHandler(file_hander)
        self.val_test_score_fn = os.path.join(run_dir, "val_train_score.txt")

    def completed_event(self, stop_time, result):
        if not result or len(result) != 2:
            return
        val_train_score = np.array(result)
        np.savetxt(self.val_test_score_fn, val_train_score)
