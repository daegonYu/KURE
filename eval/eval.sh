#!/bin/bash

set -e

cd /workspace/gits/KURE/eval
nohup python evaluate.py > eval.log 2>&1