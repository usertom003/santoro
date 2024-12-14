import mlflow
import optuna
from typing import Dict, List
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from ..evaluation.performance_evaluator import PerformanceEvaluator
import time

class PerformanceMonitor:
    def __init__(self):
        self.evaluator = PerformanceEvaluator()
        self.metrics_history = []
        self.alert_thresholds = {
            'accuracy_drop': 0.1,
            'latency_increase': 50,  # ms
            'memory_usage': 0.85  # 85% utilizzo
        }
        self.inference_times: List[float] = []
        self.fps_history: List[float] = []
        self.memory_usage: List[float] = []
        
    def start_monitoring(self, model, production_data: Dict):
        """
        Avvia il monitoraggio delle performance in produzione
        """
        mlflow.start_run()
        
        try:
            while True:
                metrics = self._collect_metrics(model, production_data)
                self._analyze_metrics(metrics)
                self._log_metrics(metrics)
                self._check_alerts(metrics)
                
                # Aggiorna dashboard in tempo reale
                self._update_dashboard(metrics)
                
                # Salva snapshot periodico
                if self._should_save_snapshot(metrics):
                    self._save_snapshot(model, metrics)
                    
        except Exception as e:
            self._handle_monitoring_error(e)
        finally:
            mlflow.end_run()
            
    def _collect_metrics(self, model, data: Dict) -> Dict:
        """
        Raccoglie metriche di performance
        """
        metrics = {
            'timestamp': datetime.now(),
            'performance': self.evaluator.evaluate_model(model, data),
            'system_metrics': self._collect_system_metrics(),
            'data_quality': self._assess_data_quality(data)
        }
        
        self.metrics_history.append(metrics)
        return metrics 
        
    def start_inference(self) -> float:
        return time.time()
        
    def end_inference(self, start_time: float):
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
    def calculate_metrics(self) -> Dict:
        avg_inference = np.mean(self.inference_times[-100:]) if self.inference_times else 0
        current_fps = 1 / avg_inference if avg_inference > 0 else 0
        
        return {
            'average_inference_time': avg_inference,
            'current_fps': current_fps,
            'memory_usage': self.memory_usage[-1] if self.memory_usage else 0
        }