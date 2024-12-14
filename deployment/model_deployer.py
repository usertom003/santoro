import docker
import kubernetes
import time
from typing import Dict
import yaml
import os
from ..monitoring.performance_monitor import PerformanceMonitor

class ModelDeployer:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.k8s_client = kubernetes.client.CoreV1Api()
        self.monitor = PerformanceMonitor()
        
    def deploy_model(self, model_path: str, config: Dict):
        """
        Deploy del modello in produzione
        """
        # Build Docker image
        image_tag = self._build_docker_image(model_path)
        
        # Deploy su Kubernetes
        self._deploy_to_kubernetes(image_tag, config)
        
        # Avvia monitoraggio
        self._start_monitoring(config)
        
    def _build_docker_image(self, model_path: str) -> str:
        """
        Build dell'immagine Docker
        """
        dockerfile = self._generate_dockerfile(model_path)
        image_tag = f"emotion-recognition:{int(time.time())}"
        
        self.docker_client.images.build(
            path=".",
            dockerfile=dockerfile,
            tag=image_tag,
            buildargs={
                "MODEL_PATH": model_path
            }
        )
        
        return image_tag
        
    def _deploy_to_kubernetes(self, image_tag: str, config: Dict):
        """
        Deploy su Kubernetes
        """
        deployment = self._generate_k8s_deployment(image_tag, config)
        service = self._generate_k8s_service(config)
        
        # Applica configurazioni
        kubernetes.utils.create_from_dict(self.k8s_client, deployment)
        kubernetes.utils.create_from_dict(self.k8s_client, service)
        
    def _start_monitoring(self, config: Dict):
        """
        Avvia il monitoraggio del deployment
        """
        self.monitor.start_monitoring(
            model=self._load_deployed_model(),
            production_data=self._get_production_data_stream()
        ) 