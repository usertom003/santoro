import tensorflow as tf
from typing import Dict, List, Tuple
import numpy as np
from ..data_handlers.dataset_manager import DatasetManager

class TransferLearner:
    def __init__(self, base_model_path: str):
        self.base_model = tf.keras.models.load_model(base_model_path)
        self.dataset_manager = DatasetManager()
        self.fine_tuning_rate = 1e-5
        self.batch_size = 16
        
    def prepare_transfer_model(self, target_dataset: str) -> tf.keras.Model:
        """
        Prepara il modello per il transfer learning
        """
        # Congela i layer base
        for layer in self.base_model.layers[:-3]:
            layer.trainable = False
            
        # Aggiungi layer di adattamento
        x = self.base_model.layers[-3].output
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        
        # Layer specifici per il target domain
        outputs = []
        for task in ['emotions', 'micro_expressions', 'action_units']:
            task_output = tf.keras.layers.Dense(
                self._get_output_dim(task, target_dataset),
                activation='softmax',
                name=f'{task}_output'
            )(x)
            outputs.append(task_output)
            
        transfer_model = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=outputs
        )
        
        return transfer_model
        
    def fine_tune(self, model: tf.keras.Model, target_data: Dict) -> Tuple[tf.keras.Model, Dict]:
        """
        Esegue il fine-tuning sul dataset target
        """
        # Prepara strategie di fine-tuning
        strategies = {
            'progressive_unfreezing': self._progressive_unfreezing,
            'discriminative_fine_tuning': self._discriminative_fine_tuning,
            'layer_wise_fine_tuning': self._layer_wise_fine_tuning
        }
        
        best_model = None
        best_performance = float('inf')
        
        # Prova diverse strategie
        for name, strategy in strategies.items():
            tuned_model = strategy(model, target_data)
            performance = self._evaluate_fine_tuning(tuned_model, target_data)
            
            if performance < best_performance:
                best_model = tuned_model
                best_performance = performance
                
        return best_model, {
            'final_loss': best_performance,
            'adaptation_metrics': self._calculate_adaptation_metrics(best_model, target_data)
        } 