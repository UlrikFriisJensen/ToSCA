#!/bin/bash

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/Seperated_cell_parameters/setup_json.json