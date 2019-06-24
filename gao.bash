#!/usr/bin/env bash

allennlp predict --output-file naive_bert.json --cuda-device 0 --use-dataset-reader --predictor bert_multitask --include-package models --include-package utils --include-package metric --include-package predictor model $1

python3 gao.py --out_json $2