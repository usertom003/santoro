import numpy as np
from typing import Dict, List
import mediapipe as mp

class ActionUnitAnalyzer:
    def __init__(self):
        self.action_units = {
            'AU1': {'name': 'Inner Brow Raiser', 'landmarks': [27, 65, 66]},
            'AU2': {'name': 'Outer Brow Raiser', 'landmarks': [46, 105, 107]},
            'AU4': {'name': 'Brow Lowerer', 'landmarks': [9, 336, 337]},
            'AU5': {'name': 'Upper Lid Raiser', 'landmarks': [159, 145, 33]},
            'AU6': {'name': 'Cheek Raiser', 'landmarks': [117, 118, 101]},
            'AU7': {'name': 'Lid Tightener', 'landmarks': [384, 385, 386]},
            'AU9': {'name': 'Nose Wrinkler', 'landmarks': [5, 6, 197]},
            'AU10': {'name': 'Upper Lip Raiser', 'landmarks': [167, 164, 165]},
            'AU12': {'name': 'Lip Corner Puller', 'landmarks': [57, 287, 288]},
            'AU15': {'name': 'Lip Corner Depressor', 'landmarks': [287, 273, 335]},
            'AU17': {'name': 'Chin Raiser', 'landmarks': [18, 200, 199]},
            'AU20': {'name': 'Lip Stretcher', 'landmarks': [61, 91, 84]},
            'AU23': {'name': 'Lip Tightener', 'landmarks': [78, 95, 88]},
            'AU25': {'name': 'Lips Part', 'landmarks': [13, 14, 312]},
            'AU26': {'name': 'Jaw Drop', 'landmarks': [78, 95, 88]},
            'AU28': {'name': 'Lip Suck', 'landmarks': [324, 308, 415]}
        }
        
        # Mappatura emozioni-AU
        self.emotion_au_mapping = {
            'happiness': ['AU6', 'AU12'],
            'sadness': ['AU1', 'AU4', 'AU15'],
            'anger': ['AU4', 'AU5', 'AU7', 'AU23'],
            'fear': ['AU1', 'AU2', 'AU4', 'AU5', 'AU20', 'AU26'],
            'surprise': ['AU1', 'AU2', 'AU5', 'AU26'],
            'disgust': ['AU9', 'AU10', 'AU15', 'AU17'],
            'contempt': ['AU12', 'AU14']
        }
        
    def analyze_action_units(self, landmarks: List[Dict]) -> Dict[str, float]:
        """
        Analizza l'attivazione delle Action Units
        """
        au_activations = {}
        
        for au_code, au_info in self.action_units.items():
            # Calcola l'attivazione per ogni AU
            activation = self._calculate_au_activation(
                landmarks, 
                au_info['landmarks']
            )
            au_activations[au_code] = {
                'name': au_info['name'],
                'activation': activation,
                'intensity': self._calculate_intensity(activation)
            }
            
        return au_activations
        
    def _calculate_au_activation(self, landmarks: List[Dict], au_landmarks: List[int]) -> float:
        """
        Calcola l'attivazione di una specifica Action Unit
        """
        # Estrai i punti di riferimento per l'AU
        points = [landmarks[idx] for idx in au_landmarks]
        
        # Calcola la distanza euclidea tra i punti
        distances = []
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                dist = np.sqrt(
                    (points[i].x - points[j].x)**2 + 
                    (points[i].y - points[j].y)**2
                )
                distances.append(dist)
                
        # Normalizza e calcola l'attivazione
        return np.mean(distances) 