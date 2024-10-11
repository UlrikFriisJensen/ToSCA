#!/bin/bash

python train.py --setup_json test_setup.json

python test.py --test_data validation --setup_json ./models/Test_run2/setup_json.json