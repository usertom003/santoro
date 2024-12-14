import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from ..data_handlers.dataset_manager import DatasetManager
import mediapipe as mp
import datetime

class EmotionTrainingPipeline:
    def __init__(self):
        self.dataset_manager = DatasetManager()
        self.batch_size = 32
        self.validation_split = 0.2
        self.learning_rate = 0.001
        
    def prepare_training_data(self) -> Tuple[Dict, Dict]:
        """
        Prepara i dati per il training
        """
        # Carica tutti i dataset
        datasets = self.dataset_manager.load_and_preprocess_datasets()
        
        # Prepara i dati per il training
        train_data = {
            'landmarks': [],
            'sequences': [],
            'blendshapes': [],
            'emotions': [],
            'micro_expressions': []
        }
        
        # Processa ogni immagine/sequenza
        for dataset in datasets['emotions']:
            features = self.dataset_manager.extract_mediapipe_features(dataset['image'])
            train_data['landmarks'].append(features['landmarks'])
            train_data['blendshapes'].append(features['blendshapes'])
            train_data['emotions'].append(dataset['label'])
            
        for sequence in datasets['micro_expressions']:
            sequence_features = self._process_sequence(sequence)
            train_data['sequences'].append(sequence_features)
            train_data['micro_expressions'].append(sequence['label'])
            
        return self._split_train_validation(train_data)
        
    def _process_sequence(self, sequence: Dict) -> np.ndarray:
        """
        Processa una sequenza di frames per microespressioni
        """
        sequence_features = []
        for frame in sequence['frames']:
            features = self.dataset_manager.extract_mediapipe_features(frame)
            sequence_features.append(features)
        return np.array(sequence_features) 

    def train_model(self, train_data: Dict, validation_data: Dict):
        """
        Esegue il training del modello con supporto per training distribuito
        """
        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            model = self.build_model()
            
            # Configurazione training
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=tf.keras.optimizers.schedules.CosineDecayRestarts(
                    initial_learning_rate=self.learning_rate,
                    first_decay_steps=1000
                )
            )
            
            losses = {
                'emotion_output': tf.keras.losses.CategoricalCrossentropy(),
                'micro_expression_output': tf.keras.losses.BinaryCrossentropy(),
                'au_output': tf.keras.losses.MeanSquaredError()
            }
            
            metrics = {
                'emotion_output': [
                    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                    tf.keras.metrics.AUC(name='auc')
                ],
                'micro_expression_output': [
                    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')
                ],
                'au_output': [
                    tf.keras.metrics.MeanSquaredError(name='mse')
                ]
            }
            
            model.compile(
                optimizer=optimizer,
                loss=losses,
                metrics=metrics
            )
        
        # Training con validazione
        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=self.epochs,
            callbacks=self._get_callbacks(),
            workers=4,
            use_multiprocessing=True
        )
        
        return model, history

    def _get_callbacks(self) -> List:
        """
        Configura i callbacks per il training
        """
        return [
            tf.keras.callbacks.ModelCheckpoint(
                filepath='models/checkpoints/model_{epoch:02d}_{val_loss:.2f}.h5',
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6
            ),
            tf.keras.callbacks.TensorBoard(
                log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                histogram_freq=1,
                update_freq='epoch'
            )
        ]