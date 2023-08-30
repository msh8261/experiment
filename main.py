# Importing MLflow
import mlflow

# Running a project
mlflow.projects.run(uri="experiment",
                    entry_point="auto_logger",
                    parameters={"max_iter": 10000},
                    experiment_name="auto_logging")