# Importing dependencies
import warnings
warnings.filterwarnings("ignore")
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import datasets

# Creating an experiment
experiment_id = mlflow.create_experiment(name="test")

# Selecting an existing experiment
mlflow.set_experiment(experiment_name="test")

# Deleting an experiment
mlflow.delete_experiment(experiment_id=experiment_id)

# Setting an experiment for manual logging
mlflow.set_experiment(experiment_name="manual_logging")


# Checking if the script is executed directly
if __name__ == "__main__":
    # Loading data
    data = datasets.load_breast_cancer()
    
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                        data.target,
                                                        stratify=data.target)
    
    # Selecting a parameter range to try out
    C = list(range(1, 5))
    
    # Starting a tracking run
    with mlflow.start_run(run_name="PARENT_RUN"):
        # For each value of C, running a child run
        for param_value in C:
            with mlflow.start_run(run_name="CHILD_RUN", nested=True):
                # Instantiating and fitting the model
                model = LogisticRegression(C=param_value, solver='saga', max_iter=5000)            
                model.fit(X=X_train, y=y_train)
                
                # Logging the current value of C
                mlflow.log_param(key="C", value=param_value)
                
                # Logging the test performance of the current model                
                mlflow.log_metric(key="Score", value=model.score(X_test, y_test)) 
                
                # Saving the model as an artifact
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")








































