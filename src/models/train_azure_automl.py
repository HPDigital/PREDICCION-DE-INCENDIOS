"""
train_azure_automl.py
=====================
Configuración y ejecución de Azure AutoML para clasificación
"""

from azureml.core import Workspace, Experiment, Dataset
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AzureAutoMLTrainer:
    """Wrapper para facilitar entrenamiento con Azure AutoML"""
    
    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        workspace_name: str
    ):
        """Inicializar conexión con Azure ML Workspace"""
        logger.info("Conectando a Azure ML Workspace...")
        
        self.ws = Workspace(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )
        
        logger.info(f"✓ Conectado a workspace: {self.ws.name}")
        logger.info(f"✓ Location: {self.ws.location}")
        logger.info(f"✓ Resource Group: {self.ws.resource_group}")
    
    def create_or_get_compute(
        self,
        compute_name: str = 'cpu-cluster',
        vm_size: str = 'STANDARD_D3_V2',
        min_nodes: int = 0,
        max_nodes: int = 4
    ) -> ComputeTarget:
        """
        Crear o recuperar cluster de cómputo.
        """
        logger.info(f"Configurando compute target: {compute_name}")
        
        try:
            compute_target = ComputeTarget(
                workspace=self.ws,
                name=compute_name
            )
            logger.info(f"✓ Compute target existente encontrado: {compute_name}")
            
        except ComputeTargetException:
            logger.info(f"Creando nuevo compute target...")
            
            compute_config = AmlCompute.provisioning_configuration(
                vm_size=vm_size,
                min_nodes=min_nodes,
                max_nodes=max_nodes,
                idle_seconds_before_scaledown=300
            )
            
            compute_target = ComputeTarget.create(
                self.ws,
                compute_name,
                compute_config
            )
            
            compute_target.wait_for_completion(show_output=True)
            logger.info(f"✓ Compute target creado: {compute_name}")
        
        return compute_target
    
    def configure_automl(
        self,
        dataset: Dataset,
        label_column: str,
        compute_target: ComputeTarget,
        experiment_timeout_hours: float = 2.0,
        max_concurrent_iterations: int = 4,
        primary_metric: str = 'AUC_weighted'
    ) -> AutoMLConfig:
        """
        Configurar experimento AutoML.
        """
        logger.info("Configurando AutoML...")
        
        automl_config = AutoMLConfig(
            task='classification',
            primary_metric=primary_metric,
            training_data=dataset,
            label_column_name=label_column,
            compute_target=compute_target,
            
            # Configuración de tiempo y recursos
            experiment_timeout_hours=experiment_timeout_hours,
            max_concurrent_iterations=max_concurrent_iterations,
            iterations=100,  # Número máximo de modelos a probar
            iteration_timeout_minutes=15,
            
            # Validación
            n_cross_validations=5,
            validation_size=0.2,
            
            # Early stopping
            enable_early_stopping=True,
            early_stopping_n_iters=10,
            
            # Features
            featurization='auto',  # Feature engineering automático
            enable_stack_ensemble=True,
            enable_voting_ensemble=True,
            
            # Logging
            verbosity=logging.INFO,
            
            # Class balancing
            enable_onnx_compatible_models=False
        )
        
        logger.info("✓ AutoML configurado")
        logger.info(f"  • Task: classification")
        logger.info(f"  • Primary metric: {primary_metric}")
        logger.info(f"  • Max iterations: 100")
        logger.info(f"  • Timeout: {experiment_timeout_hours}h")
        logger.info(f"  • Cross-validation: 5 folds")
        
        return automl_config
    
    def run_experiment(
        self,
        experiment_name: str,
        automl_config: AutoMLConfig
    ):
        """
        Ejecutar experimento AutoML.
        """
        logger.info(f"Iniciando experimento: {experiment_name}")
        
        experiment = Experiment(self.ws, experiment_name)
        
        run = experiment.submit(
            automl_config,
            show_output=True
        )
        
        logger.info("Experimento enviado. Esperando resultados...")
        logger.info("Puede monitorear en Azure ML Studio:")
        logger.info(run.get_portal_url())
        
        # Esperar completar
        run.wait_for_completion(show_output=True)
        
        logger.info("✓ Experimento completado")
        
        return run
    
    def get_best_model(self, run):
        """
        Obtener mejor modelo del experimento.
        """
        logger.info("Recuperando mejor modelo...")
        
        best_run, fitted_model = run.get_output()
        
        logger.info(f"✓ Mejor modelo: {best_run.properties['model_name']}")
        logger.info(f"✓ Run ID: {best_run.id}")
        logger.info(f"✓ Primary metric: {best_run.properties['score']}")
        
        # Obtener métricas
        metrics = best_run.get_metrics()
        
        print("\n" + "="*60)
        print("MÉTRICAS DEL MEJOR MODELO")
        print("="*60)
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  • {metric}: {value:.4f}")
            else:
                print(f"  • {metric}: {value}")
        
        return best_run, fitted_model


def main():
    """Pipeline completo de Azure AutoML"""
    
    # Credenciales (usar variables de entorno en producción)
    SUBSCRIPTION_ID = 'your-subscription-id'
    RESOURCE_GROUP = 'rg-fire-prediction'
    WORKSPACE_NAME = 'ml-fire-cochabamba'
    
    # Inicializar trainer
    trainer = AzureAutoMLTrainer(
        subscription_id=SUBSCRIPTION_ID,
        resource_group=RESOURCE_GROUP,
        workspace_name=WORKSPACE_NAME
    )
    
    # Crear compute
    compute_target = trainer.create_or_get_compute(
        compute_name='cpu-cluster',
        vm_size='STANDARD_D3_V2',
        max_nodes=4
    )
    
    # Ejecutar resto del pipeline...
    print("\n✓ Configuración inicial completada")
    print("✓ Continuar con registro de dataset y entrenamiento")


if __name__ == "__main__":
    main()
