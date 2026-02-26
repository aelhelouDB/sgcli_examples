# Overview
This example shows how to launch CLI from a Git folder of a library without pre-installing.

Meta Lingua is an open source library in https://github.com/facebookresearch/lingua

# How to run

\# need to run from df1 workspace because the training data in
\# /Volumes/main/rohitkg/data/raw_data/lingua_train_data
export DATABRICKS_CONFIG_PROFILE=df1
sgcli run -f workload.yaml
