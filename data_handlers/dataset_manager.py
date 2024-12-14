import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import random
from collections import defaultdict

class DatasetManager:
    def __init__(self):
        self.emotion_mapping = {
            'anger': 0,
            'disgust': 1,
            'fear': 2,
            'happiness': 3,
            'sadness': 4,
            'surprise': 5,
            'neutral': 6,
            'contempt': 7
        }
        
        self.landmark_features = {
            'eyebrows': list(range(46, 54)),  # Landmarks per sopracciglia
            'eyes': list(range(33, 46)),      # Landmarks per occhi
            'nose': list(range(27, 35)),      # Landmarks per naso
            'mouth': list(range(48, 68)),     # Landmarks per bocca
            'jaw': list(range(0, 17)),        # Landmarks per mascella
        }
        
    def load_and_preprocess_datasets(self):
        """
        Carica e preprocessa tutti i dataset
        """
        datasets = {
            'emotions': self._load_emotion_datasets(),
            'micro_expressions': self._load_micro_expression_datasets()
        }
        
        return self._combine_datasets(datasets)
    
    def _load_emotion_datasets(self):
        """
        Carica dataset delle emozioni da multiple fonti
        """
        datasets = []
        
        # FER2013
        fer_path = "datasets/fer2013/fer2013.csv"
        if os.path.exists(fer_path):
            fer_data = pd.read_csv(fer_path)
            datasets.append(self._process_fer2013(fer_data))
        
        # AffectNet
        affect_path = "datasets/affectnet/"
        if os.path.exists(affect_path):
            affect_data = self._load_affectnet(affect_path)
            datasets.append(affect_data)
            
        return self._merge_datasets(datasets)
    
    def _load_micro_expression_datasets(self):
        """
        Carica dataset delle microespressioni
        """
        datasets = []
        
        # CASME II
        casme_path = "datasets/casme2/"
        if os.path.exists(casme_path):
            casme_data = self._load_casme2(casme_path)
            datasets.append(casme_data)
        
        # SAMM
        samm_path = "datasets/samm/"
        if os.path.exists(samm_path):
            samm_data = self._load_samm(samm_path)
            datasets.append(samm_data)
            
        return self._merge_datasets(datasets)
    
    def extract_mediapipe_features(self, image) -> Dict:
        """
        Estrae feature usando MediaPipe
        """
        base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )
        
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            results = landmarker.detect(image)
            return self._process_landmarks(results)
            
    def _process_landmarks(self, results) -> Dict:
        """
        Processa i landmark di MediaPipe e li converte in features
        """
        features = {}
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # Estrai features geometriche
            features['distances'] = self._calculate_landmark_distances(landmarks)
            features['angles'] = self._calculate_landmark_angles(landmarks)
            features['blendshapes'] = results.face_blendshapes[0] if results.face_blendshapes else None
            
        return features
        
    def _calculate_landmark_distances(self, landmarks) -> np.ndarray:
        """
        Calcola le distanze euclidee tra landmark chiave
        """
        distances = []
        for region, indices in self.landmark_features.items():
            for i in indices:
                for j in indices:
                    if i < j:
                        dist = np.sqrt(
                            (landmarks[i].x - landmarks[j].x)**2 +
                            (landmarks[i].y - landmarks[j].y)**2
                        )
                        distances.append(dist)
        return np.array(distances) 

    def _load_affectnet(self, path: str) -> Dict:
        """
        Carica e preprocessa il dataset AffectNet
        """
        data = {
            'images': [],
            'labels': [],
            'landmarks': [],
            'expressions': []
        }
        
        for emotion_dir in os.listdir(path):
            emotion_path = os.path.join(path, emotion_dir)
            if not os.path.isdir(emotion_path):
                continue
            
            emotion_label = self.emotion_mapping.get(emotion_dir)
            if emotion_label is None:
                continue
            
            for img_file in os.listdir(emotion_path):
                img_path = os.path.join(emotion_path, img_file)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Estrai features con MediaPipe
                features = self.extract_mediapipe_features(image)
                if features is None:
                    continue
                
                data['images'].append(image)
                data['labels'].append(emotion_label)
                data['landmarks'].append(features['landmarks'])
                data['expressions'].append(features['blendshapes'])
                
        return data

    def _load_casme2(self, path: str) -> Dict:
        """
        Carica e preprocessa il dataset CASME II per microespressioni
        """
        data = {
            'sequences': [],
            'labels': [],
            'onsets': [],
            'apexes': [],
            'offsets': []
        }
        
        # Carica file di annotazione
        annotation_file = os.path.join(path, 'CASME2-coding-20140508.xlsx')
        annotations = pd.read_excel(annotation_file)
        
        for _, row in annotations.iterrows():
            subject = f"sub{row['Subject']}"
            sequence = str(row['Filename']).zfill(3)
            emotion = row['Estimated Emotion']
            
            # Carica sequenza di frames
            sequence_path = os.path.join(path, subject, sequence)
            if not os.path.exists(sequence_path):
                continue
            
            frames = []
            landmarks = []
            
            for frame_file in sorted(os.listdir(sequence_path)):
                frame_path = os.path.join(sequence_path, frame_file)
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                
                features = self.extract_mediapipe_features(frame)
                if features is None:
                    continue
                
                frames.append(frame)
                landmarks.append(features['landmarks'])
                
            if len(frames) < 3:  # Sequenza troppo corta
                continue
            
            data['sequences'].append({
                'frames': frames,
                'landmarks': landmarks,
                'emotion': emotion,
                'onset': row['OnsetFrame'],
                'apex': row['ApexFrame'],
                'offset': row['OffsetFrame']
            })
            
        return data

    def _merge_datasets(self, datasets: List[Dict]) -> Dict:
        """
        Unisce più dataset in un unico formato standardizzato
        
        Args:
            datasets: Lista di dataset da unire
            
        Returns:
            Dict: Dataset unificato con chiavi 'emotions' e 'micro_expressions'
        """
        merged = {
            'emotions': [],
            'micro_expressions': []
        }
        
        for dataset in datasets:
            if 'emotions' in dataset:
                merged['emotions'].extend(dataset['emotions'])
            if 'micro_expressions' in dataset:
                merged['micro_expressions'].extend(dataset['micro_expressions'])
                
        # Shuffle dei dati
        random.shuffle(merged['emotions'])
        random.shuffle(merged['micro_expressions'])
        
        # Bilanciamento delle classi
        merged['emotions'] = self._balance_classes(merged['emotions'])
        merged['micro_expressions'] = self._balance_classes(merged['micro_expressions'])
        
        return merged
        
    def _balance_classes(self, data: List[Dict]) -> List[Dict]:
        """
        Bilancia le classi nel dataset usando under-sampling
        
        Args:
            data: Lista di dizionari contenenti 'image' e 'label'
            
        Returns:
            List[Dict]: Dataset bilanciato
        """
        # Raggruppa per label
        grouped = defaultdict(list)
        for item in data:
            grouped[item['label']].append(item)
            
        # Trova la dimensione della classe più piccola
        min_size = min(len(samples) for samples in grouped.values())
        
        # Under-sampling
        balanced = []
        for label, samples in grouped.items():
            balanced.extend(random.sample(samples, min_size))
            
        return balanced
