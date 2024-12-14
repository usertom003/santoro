import tensorflow as tf
import numpy as np
from typing import Dict, List
import mediapipe as mp
from sklearn.model_selection import train_test_split

class EmotionModelTrainer:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 100
        self.sequence_length = 30  # Per microespressioni
        
    def build_model(self):
        """
        Costruisce il modello multi-stream per emozioni e microespressioni
        """
        # Input streams
        landmark_input = tf.keras.Input(shape=(478, 3))  # MediaPipe face landmarks
        sequence_input = tf.keras.Input(shape=(self.sequence_length, 478, 3))
        blendshape_input = tf.keras.Input(shape=(52,))  # MediaPipe blendshapes
        
        # Landmark stream
        x1 = tf.keras.layers.Dense(256, activation='relu')(landmark_input)
        x1 = tf.keras.layers.Dropout(0.3)(x1)
        
        # Sequence stream per microespressioni
        x2 = tf.keras.layers.LSTM(128, return_sequences=True)(sequence_input)
        x2 = tf.keras.layers.LSTM(64)(x2)
        
        # Blendshape stream
        x3 = tf.keras.layers.Dense(64, activation='relu')(blendshape_input)
        
        # Fusion
        combined = tf.keras.layers.Concatenate()([x1, x2, x3])
        
        # Shared layers
        x = tf.keras.layers.Dense(512, activation='relu')(combined)
        x = tf.keras.layers.Dropout(0.5)(x)
        
        # Output branches
        emotion_output = tf.keras.layers.Dense(8, activation='softmax', name='emotion')(x)
        micro_exp_output = tf.keras.layers.Dense(5, activation='sigmoid', name='micro_expression')(x)
        
        return tf.keras.Model(
            inputs=[landmark_input, sequence_input, blendshape_input],
            outputs=[emotion_output, micro_exp_output]
        ) 

    def train_model(self, train_data: Dict, validation_data: Dict):
        """
        Training del modello con supporto multi-task
        """
        model = self.build_model()
        
        # Configurazione ottimizzatore e loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        losses = {
            'emotion': 'categorical_crossentropy',
            'micro_expression': 'binary_crossentropy'
        }
        loss_weights = {
            'emotion': 1.0,
            'micro_expression': 0.5
        }
        
        # Metriche per ogni task
        metrics = {
            'emotion': ['accuracy'],
            'micro_expression': ['accuracy', tf.keras.metrics.AUC()]
        }
        
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics=metrics
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'models/emotion_recognition_model.h5',
                monitor='val_loss',
                save_best_only=True
            )
        ]
        
        # Training
        history = model.fit(
            [train_data['landmarks'], train_data['sequences'], train_data['blendshapes']],
            [train_data['emotions'], train_data['micro_expressions']],
            validation_data=(
                [validation_data['landmarks'], validation_data['sequences'], validation_data['blendshapes']],
                [validation_data['emotions'], validation_data['micro_expressions']]
            ),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )
        
        return model, history 