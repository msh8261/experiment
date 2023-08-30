#!/bin/sh
mlflow run experiment -e auto_logger --experiment-name auto_logging -P max_iter=1000


mlflow run -e auto_logger --experiment-name auto_logging -P max_iter=1000