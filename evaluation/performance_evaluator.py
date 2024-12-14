import numpy as np
from typing import Dict, List
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from ..processors.action_unit_analyzer import ActionUnitAnalyzer

class PerformanceEvaluator:
    def __init__(self):
        self.au_analyzer = ActionUnitAnalyzer()
        self.metrics_history = []
        self.confusion_matrices = {}
        
    def evaluate_model(self, model, test_data: Dict, ground_truth: Dict) -> Dict:
        """
        Valutazione completa del modello
        """
        predictions = model.predict([
            test_data['landmarks'],
            test_data['sequences'],
            test_data['blendshapes']
        ])
        
        evaluation = {
            'emotion_metrics': self._evaluate_emotion_recognition(
                predictions[0], ground_truth['emotions']
            ),
            'micro_expression_metrics': self._evaluate_micro_expressions(
                predictions[1], ground_truth['micro_expressions']
            ),
            'au_metrics': self._evaluate_action_units(
                predictions[2], ground_truth['action_units']
            ),
            'temporal_metrics': self._evaluate_temporal_consistency(predictions),
            'cross_cultural_metrics': self._evaluate_cross_cultural_performance(
                predictions, ground_truth
            )
        }
        
        self.metrics_history.append(evaluation)
        return evaluation
        
    def generate_report(self, save_path: str = 'reports/evaluation_report.pdf'):
        """
        Genera un report dettagliato delle performance
        """
        report = {
            'summary': self._generate_summary(),
            'detailed_metrics': self._generate_detailed_metrics(),
            'visualizations': self._generate_visualizations(),
            'recommendations': self._generate_recommendations()
        }
        
        self._save_report(report, save_path)
        return report

    def _evaluate_emotion_recognition(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict:
        """
        Valuta le performance del riconoscimento delle emozioni
        """
        metrics = {}
        
        # Metriche base
        metrics['accuracy'] = np.mean(np.argmax(predictions, axis=1) == np.argmax(ground_truth, axis=1))
        metrics['confusion_matrix'] = confusion_matrix(
            np.argmax(ground_truth, axis=1),
            np.argmax(predictions, axis=1)
        )
        
        # Metriche per classe
        metrics['per_emotion'] = classification_report(
            np.argmax(ground_truth, axis=1),
            np.argmax(predictions, axis=1),
            output_dict=True
        )
        
        # Analisi degli errori
        metrics['error_analysis'] = self._analyze_errors(predictions, ground_truth)
        
        return metrics 