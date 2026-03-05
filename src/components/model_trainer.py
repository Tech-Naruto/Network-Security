import os, sys
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import mlflow
import dagshub
from dotenv import load_dotenv
load_dotenv()

DAGSHUB_REPO_OWNER=os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME=os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_REPO_EXP_NAME=os.getenv("DAGSHUB_REPO_EXP_NAME")

dagshub.init(repo_owner=DAGSHUB_REPO_OWNER, repo_name=DAGSHUB_REPO_NAME, mlflow=True)

mlflow.set_experiment(DAGSHUB_REPO_EXP_NAME)

from src.exception.exception import NetworkSecurityException
from src.logging.logger import logging
from src.entity.config_entity import ModelTrainerConfig
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.utils.main_utils import save_object, load_object, load_numpy_array_data, evaluate_models
from src.utils.ml_utils.metric.classification_metric import get_classification_score
from src.utils.ml_utils.model.estimator import NetworkModel


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, best_model, classification_metric, run_name):
        with mlflow.start_run(run_name=run_name):
            f1_score = classification_metric.f1_score
            precision_score = classification_metric.precision_score
            recall_score = classification_metric.recall_score

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision_score", precision_score)
            mlflow.log_metric("recall_score", recall_score)

            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, x_train, y_train, x_test, y_test):
        models = {
            "Logistic Regression": LogisticRegression(),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }

        params = {
            "Logistic Regression": {'max_iter': [200, 300, 400]},
            "KNN": {
                "n_neighbors": [5, 7, 9, 11, 13],
                "weights": ["uniform", "distance"],
                # "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
            },
            "Decision Tree": {
                "criterion": ["gini", "entropy"],
                # "splitter": ["best", "random"],
                # "max_features": ["sqrt", "log2", None],
            },
            "Random Forest": {
                "n_estimators": [50, 100, 150],
                # "criterion": ["gini", "entropy"],
                # "max_features": ["sqrt", "log2", None],
            },
            "Gradient Boosting": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.2],
                # "max_depth": [3, 5, 7],
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 150],
                "learning_rate": [0.01, 0.1, 0.2],
            },
        }

        logging.info("Evaluating Models")
        model_report: dict = evaluate_models(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test, models = models, params = params)

        logging.info("Model report: {}".format(model_report))

        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        
        logging.info(f"Best Model Name: {best_model_name} with score: {best_model_score}")
        best_model = models[best_model_name]

        logging.info("Predicting on train and test data")
        y_train_pred = best_model.predict(x_train)
        train_classification_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

        # >>> Track the mlflow <<<
        self.track_mlflow(best_model, train_classification_metric, "train-run")

        y_test_pred = best_model.predict(x_test)
        test_classification_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

        # >>> Track the mlflow <<<
        self.track_mlflow(best_model, test_classification_metric, "test-run")

        logging.info("Loading preprocessor object")
        preprocessor = load_object(file_path = self.data_transformation_artifact.transformed_object_file_path)

        network_model = NetworkModel(preprocessor = preprocessor, model = best_model)

        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        
        logging.info("Saving Network Model")
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=network_model)

        # >>> Saving final models <<< 
        logging.info("Saving final models")
        os.makedirs("./final_models", exist_ok=True)
        save_object(file_path="./final_models/model.pkl", obj=best_model)
        save_object(file_path="./final_models/preprocessor.pkl", obj=preprocessor)

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_classification_metric,
            test_metric_artifact=test_classification_metric
        )        

        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Initiating model trainer")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Loading train and test array")
            train_arr = load_numpy_array_data(file_path=train_file_path)
            test_arr = load_numpy_array_data(file_path=test_file_path)

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            logging.info("Initiating model training")
            model_trainer_artifact = self.train_model(x_train = x_train, y_train = y_train, x_test = x_test, y_test = y_test)

            logging.info("Model trainer completed")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

