import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# versions=[1, 2, 3]
# for version in versions:
#     client.delete_model_version(name="sk-learn-random-forest-reg-model", version=version)

# Delete a registered model along with all its versions
client.delete_registered_model(name="resnet18")

