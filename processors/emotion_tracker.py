import numpy as np
from collections import deque
from typing import Dict, List
import pandas as pd

class EmotionTracker:
    def __init__(self, window_size: int = 60):  # 60 frame = 2 secondi a 30fps
        self.window_size = window_size
        self.emotion_history = deque(maxlen=window_size)
        self.micro_exp_history = deque(maxlen=window_size)
        self.blend_weights = {
            'happiness_surprise': 0.7,
            'anger_disgust': 0.8,
            'fear_surprise': 0.6
        }
        
    def update(self, emotions: Dict, micro_expressions: Dict) -> Dict:
        """
        Aggiorna lo stato emotivo e analizza le tendenze
        """
        self.emotion_history.append(emotions)
        self.micro_exp_history.append(micro_expressions)
        
        analysis = {
            'current_state': self._analyze_current_state(),
            'emotional_trend': self._analyze_emotional_trend(),
            'mixed_expressions': self._detect_mixed_expressions(),
            'confidence_scores': self._calculate_confidence()
        }
        
        return analysis
        
    def _analyze_current_state(self) -> Dict:
        """
        Analizza lo stato emotivo corrente
        """
        if not self.emotion_history:
            return {}
            
        current = self.emotion_history[-1]
        dominant_emotion = max(current.items(), key=lambda x: x[1])
        
        return {
            'dominant_emotion': dominant_emotion[0],
            'intensity': dominant_emotion[1],
            'secondary_emotions': self._get_secondary_emotions(current)
        }
        
    def _analyze_emotional_trend(self) -> Dict:
        """
        Analizza il trend emotivo nel tempo
        """
        if len(self.emotion_history) < 3:
            return {}
            
        df = pd.DataFrame(list(self.emotion_history))
        
        trends = {}
        for emotion in df.columns:
            # Calcola la derivata delle emozioni
            derivative = np.gradient(df[emotion].values)
            trends[emotion] = {
                'direction': 'increasing' if derivative[-1] > 0 else 'decreasing',
                'rate': abs(derivative[-1]),
                'stability': 1 - np.std(derivative)
            }
            
        return trends 