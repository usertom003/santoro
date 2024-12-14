import mediapipe as mp
import numpy as np
from typing import List, Dict
import cv2
from emotion_recognition.processors.action_unit_analyzer import ActionUnitAnalyzer

class FaceAnalyzer:
    def __init__(self):
        self.expression_buffer = []
        self.buffer_size = 30  # Frame buffer per microespressioni
        self.blendshape_names = [
            'neutral', 'happiness', 'sadness', 'surprise',
            'fear', 'anger', 'disgust', 'contempt'
        ]
        self.au_analyzer = ActionUnitAnalyzer()
        
    def analyze_facial_features(self, face_landmarks: Dict) -> Dict:
        """
        Analizza le caratteristiche facciali per l'emozione
        """
        features = {}
        
        # Analisi delle Action Units
        au_activations = self.au_analyzer.analyze_action_units(face_landmarks)
        features['action_units'] = au_activations
        
        # Analisi delle distanze relative
        features['symmetry'] = self._analyze_facial_symmetry(face_landmarks)
        features['muscle_activation'] = self._analyze_muscle_activation(face_landmarks)
        features['emotional_intensity'] = self._calculate_emotional_intensity(face_landmarks)
        
        # Integrazione con Holistic
        features['pose'] = self._analyze_pose(face_landmarks)
        features['hand_gestures'] = self._analyze_hand_gestures(face_landmarks)
        
        return features
    
    def analyze_micro_expressions(self, face_landmarks_sequence: List[Dict]) -> Dict:
        """
        Analizza le microespressioni nel tempo
        """
        self.expression_buffer.append(face_landmarks_sequence)
        if len(self.expression_buffer) > self.buffer_size:
            self.expression_buffer.pop(0)
            
        micro_expressions = {
            'onset': self._detect_expression_onset(),
            'apex': self._detect_expression_apex(),
            'offset': self._detect_expression_offset()
        }
        
        return micro_expressions
        
    def _analyze_facial_symmetry(self, landmarks: Dict) -> float:
        """
        Calcola la simmetria facciale
        """
        left_features = landmarks['left_eye'] + landmarks['left_eyebrow']
        right_features = landmarks['right_eye'] + landmarks['right_eyebrow']
        
        symmetry_score = np.mean([
            abs(l.x + r.x) for l, r in zip(left_features, right_features)
        ])
        
        return symmetry_score
    
    def _detect_expression_onset(self) -> Dict:
        """
        Rileva l'inizio di una microespressione
        """
        if len(self.expression_buffer) < 3:
            return {}
        
        onset_features = {}
        for i in range(len(self.expression_buffer) - 2):
            curr_frame = self.expression_buffer[i]
            next_frame = self.expression_buffer[i + 1]
            
            # Calcola il movimento muscolare
            muscle_movement = self._calculate_muscle_movement(curr_frame, next_frame)
            
            # Rileva cambiamenti rapidi
            if self._is_rapid_change(muscle_movement):
                onset_features['frame_idx'] = i
                onset_features['movement'] = muscle_movement
                onset_features['intensity'] = self._calculate_movement_intensity(muscle_movement)
                break
            
        return onset_features
    
    def _calculate_muscle_movement(self, frame1, frame2) -> Dict:
        """
        Calcola il movimento muscolare tra due frame
        """
        movements = {}
        
        # Action Units (FACS)
        aus = {
            'inner_brow': [0, 1],    # AU1
            'outer_brow': [2, 3],    # AU2
            'nose': [27, 28, 29],    # AU9
            'upper_lip': [48, 49, 50],  # AU10
            'corner_lip': [48, 54]    # AU12
        }
        
        for au_name, landmarks in aus.items():
            movement = np.mean([
                np.linalg.norm(
                    np.array([frame2[i].x - frame1[i].x, frame2[i].y - frame1[i].y])
                ) for i in landmarks
            ])
            movements[au_name] = movement
            
        return movements
    
    def _detect_expression_apex(self) -> Dict:
        """
        Rileva il punto di massima intensità della microespressione
        """
        if len(self.expression_buffer) < 3:
            return {}
        
        intensities = [self._calculate_expression_intensity(frame) 
                      for frame in self.expression_buffer]
        
        apex_idx = np.argmax(intensities)
        return {
            'frame_idx': apex_idx,
            'intensity': intensities[apex_idx],
            'au_activation': self._get_action_units_at_frame(apex_idx)
        }
    
    def _detect_expression_offset(self) -> Dict:
        """
        Rileva la fine della microespressione
        """
        if len(self.expression_buffer) < 3:
            return {}
        
        offset_features = {}
        for i in range(len(self.expression_buffer) - 2, 0, -1):
            curr_frame = self.expression_buffer[i]
            prev_frame = self.expression_buffer[i - 1]
            
            muscle_movement = self._calculate_muscle_movement(curr_frame, prev_frame)
            
            if self._is_movement_stabilizing(muscle_movement):
                offset_features['frame_idx'] = i
                offset_features['movement'] = muscle_movement
                break
            
        return offset_features
    
    def _calculate_expression_intensity(self, frame) -> float:
        """
        Calcola l'intensità dell'espressione
        """
        if not frame:
            return 0.0
        
        # Calcola l'intensità basata sui movimenti muscolari
        movements = self._calculate_muscle_movement(frame, self.expression_buffer[0])
        return np.mean(list(movements.values()))
    
    def _is_rapid_change(self, movement: Dict, threshold: float = 0.1) -> bool:
        """
        Determina se il movimento è abbastanza rapido per essere una microespressione
        """
        return any(v > threshold for v in movement.values())
    
    def _is_movement_stabilizing(self, movement: Dict, threshold: float = 0.05) -> bool:
        """
        Determina se il movimento sta tornando alla neutralità
        """
        return all(v < threshold for v in movement.values())
    
    def analyze_mixed_expressions(self, face_landmarks: Dict) -> Dict:
        """
        Analizza la presenza di espressioni miste
        """
        features = self.analyze_facial_features(face_landmarks)
        
        # Analisi delle regioni facciali separate
        upper_face = self._analyze_upper_face(face_landmarks)
        lower_face = self._analyze_lower_face(face_landmarks)
        
        # Rileva incongruenze tra le regioni
        mixed_expressions = self._detect_region_incongruences(upper_face, lower_face)
        
        return {
            'upper_face': upper_face,
            'lower_face': lower_face,
            'mixed_expressions': mixed_expressions,
            'confidence': self._calculate_mixed_expression_confidence(features)
        }
    
    def _analyze_upper_face(self, landmarks: Dict) -> Dict:
        """
        Analizza l'espressione della parte superiore del viso
        """
        upper_features = {}
        
        # Analisi sopracciglia
        brow_movement = self._calculate_brow_movement(landmarks)
        upper_features['brow_action'] = self._classify_brow_action(brow_movement)
        
        # Analisi occhi
        eye_features = self._analyze_eye_state(landmarks)
        upper_features.update(eye_features)
        
        return upper_features
    
    def _analyze_lower_face(self, landmarks: Dict) -> Dict:
        """
        Analizza l'espressione della parte inferiore del viso
        """
        lower_features = {}
        
        # Analisi bocca
        mouth_shape = self._analyze_mouth_shape(landmarks)
        lower_features['mouth_action'] = self._classify_mouth_action(mouth_shape)
        
        # Analisi guance e naso
        nose_wrinkle = self._detect_nose_wrinkle(landmarks)
        cheek_movement = self._analyze_cheek_movement(landmarks)
        
        lower_features.update({
            'nose_wrinkle': nose_wrinkle,
            'cheek_movement': cheek_movement
        })
        
        return lower_features
    
    def _analyze_pose(self, landmarks: Dict) -> Dict:
        """
        Analizza la postura e l'orientamento della testa
        """
        pose_features = {}
        
        # Estrai punti di riferimento per la testa
        nose = landmarks[1]
        left_eye = landmarks[33]
        right_eye = landmarks[263]
        
        # Calcola rotazione della testa
        pose_features['head_rotation'] = self._calculate_head_rotation(
            nose, left_eye, right_eye
        )
        
        # Calcola inclinazione della testa
        pose_features['head_tilt'] = self._calculate_head_tilt(
            left_eye, right_eye
        )
        
        return pose_features