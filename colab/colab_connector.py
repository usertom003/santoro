import os
import sys
from google.colab import drive
from google.colab import files
import tensorflow as tf
import numpy as np
from typing import Dict, Optional
import subprocess

class ColabConnector:
    def __init__(self):
        self.drive_mount_path = '/content/drive'
        self.project_path = '/content/emotion_recognition'
        self.model_path = f'{self.project_path}/models'
        
    def setup_environment(self):
        """
        Configura l'ambiente Colab
        """
        # Monta Google Drive
        drive.mount(self.drive_mount_path)
        
        # Installa dipendenze
        self._install_dependencies()
        
        # Configura GPU
        self._setup_gpu()
        
        # Clona repository se necessario
        if not os.path.exists(self.project_path):
            self._clone_repository()
            
    def _install_dependencies(self):
        """
        Installa le dipendenze necessarie
        """
        subprocess.run([
            "pip", "install", 
            "mediapipe", "opencv-python", 
            "tensorflow-model-optimization",
            "docker", "kubernetes", "mlflow", "optuna"
        ])
        
    def _setup_gpu(self):
        """
        Configura l'ambiente GPU
        """
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
                
    def _clone_repository(self):
        """
        Clona il repository del progetto
        """
        subprocess.run([
            "git", "clone",
            "https://github.com/tuouser/emotion_recognition.git",
            self.project_path
        ])
        
    def load_model(self, model_name: str) -> tf.keras.Model:
        """
        Carica un modello da Google Drive
        """
        model_path = f'{self.model_path}/{model_name}'
        return tf.keras.models.load_model(model_path)
        
    def save_model(self, model: tf.keras.Model, model_name: str):
        """
        Salva un modello su Google Drive
        """
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
            
        model_path = f'{self.model_path}/{model_name}'
        model.save(model_path)
        
    def upload_dataset(self, dataset_path: str):
        """
        Carica un dataset su Colab
        """
        uploaded = files.upload()
        for filename in uploaded.keys():
            dest_path = os.path.join(self.project_path, 'datasets', dataset_path)
            subprocess.run(['mv', filename, dest_path])