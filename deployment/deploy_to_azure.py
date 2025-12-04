"""
Despliegue de artefacto (modelo + scaler + features) a Azure Container Instances.
Requiere un archivo config.json del workspace de Azure ML en el directorio actual.
"""

from azureml.core import Environment, Model, Workspace
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


def deploy_model_to_azure(
    model_path: str = "../models/fire_model.pkl",
    model_name: str = "fire-model",
    service_name: str = "fire-prediction-api",
    workspace_config: str = "config.json",
):
    print("Conectando a Azure ML Workspace...")
    ws = Workspace.from_config(workspace_config)
    print(f"Workspace: {ws.name} | Subscription: {ws.subscription_id}")

    print(f"Registrando artefacto '{model_name}'...")
    model = Model.register(
        workspace=ws,
        model_path=model_path,
        model_name=model_name,
        description="Artefacto RandomForest (modelo + scaler + features)",
        tags={"project": "fire-prediction", "artifact": "model-scaler-features"},
    )
    print(f"Modelo registrado: {model.name} v{model.version}")

    print("Creando environment...")
    env = Environment(name="fire-prediction-env")
    deps = CondaDependencies()
    deps.add_conda_package("numpy==1.24.3")
    deps.add_conda_package("pandas==1.5.3")
    deps.add_pip_package("scikit-learn==1.2.2")
    deps.add_pip_package("joblib==1.3.1")
    deps.add_pip_package("azureml-defaults==1.51.0")
    env.python.conda_dependencies = deps

    print("Configurando inferencia (score.py)...")
    inference_config = InferenceConfig(entry_script="score.py", environment=env)

    print("Configurando deployment ACI...")
    aci_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        auth_enabled=True,
        enable_app_insights=True,
        description="API REST para riesgo de incendios",
        tags={"env": "prod", "project": "fire-prediction"},
    )

    print(f"Desplegando servicio '{service_name}' (puede tardar varios minutos)...")
    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=aci_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)

    print("Deployment listo.")
    print(f"Estado: {service.state}")
    print(f"URL: {service.scoring_uri}")
    print(f"Primary Key: {service.get_keys()[0][:20]}...")
    return service


if __name__ == "__main__":
    deploy_model_to_azure()
