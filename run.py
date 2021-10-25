#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
# sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/features')
sys.path.insert(0, 'src/model')

from build_features import load_data
# from analysis import compute_aggregates
from train import train_test


def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

    if 'data' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh)

        # make the data target
        A, X, y, idx_train, idx_val, idx_test = load_data(**data_cfg)

    # if 'analysis' in targets:
    #     with open('config/analysis-params.json') as fh:
    #         analysis_cfg = json.load(fh)

    #     # make the data target
    #     compute_aggregates(data, **analysis_cfg)

    if 'model' in targets:
        with open('config/model-params.json') as fh:
            model_cfg = json.load(fh)

        # make the data target
        train_test(A, X, y, idx_train, idx_val, idx_test, **model_cfg)

    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
