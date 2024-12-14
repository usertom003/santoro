import cv2
import numpy as np
import mediapipe as mp
from typing import Dict
import matplotlib.pyplot as plt

class RealtimeVisualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.fig, self.ax = plt.subplots(2, 2, figsize=(15, 10))
        plt.ion()
        
    def visualize_frame(self, frame: np.ndarray, results: Dict):
        """
        Visualizza i risultati in tempo reale
        """
        # Frame originale con landmark
        self.ax[0,0].clear()
        self.ax[0,0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self._draw_landmarks(frame, results['face_landmarks'])
        self.ax[0,0].set_title('Face Landmarks')
        
        # Grafico emozioni
        self.ax[0,1].clear()
        self._plot_emotions(results['emotions'])
        
        # Grafico microespressioni
        self.ax[1,0].clear()
        self._plot_micro_expressions(results['micro_expressions'])
        
        # Mappa di attivazione muscolare
        self.ax[1,1].clear()
        self._plot_muscle_activation(results['muscle_activation'])
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
        
    def _plot_emotions(self, emotions: Dict):
        """
        Visualizza le probabilità delle emozioni
        """
        emotions_names = list(emotions.keys())
        probabilities = list(emotions.values())
        
        bars = self.ax[0,1].bar(emotions_names, probabilities)
        self.ax[0,1].set_ylim(0, 1)
        self.ax[0,1].set_title('Emotion Probabilities')
        
        # Aggiungi valori sopra le barre
        for bar in bars:
            height = bar.get_height()
            self.ax[0,1].text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{height:.2f}',
                ha='center', va='bottom'
            ) 

    def _draw_landmarks(self, frame, landmarks):
        """
        Disegna i landmark facciali con annotazioni
        """
        if not landmarks:
            return
        
        # Configurazione drawing
        drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(0,255,0)
        )
        
        # Disegna mesh facciale
        self.mp_drawing.draw_landmarks(
            image=frame,
            landmark_list=landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

    def _plot_micro_expressions(self, micro_expressions: Dict):
        """
        Visualizza l'analisi delle microespressioni
        """
        # Timeline delle microespressioni
        times = list(range(len(micro_expressions['sequence'])))
        intensities = micro_expressions['intensities']
        
        self.ax[1,0].plot(times, intensities, '-o')
        self.ax[1,0].set_title('Micro-Expression Timeline')
        self.ax[1,0].set_xlabel('Frame')
        self.ax[1,0].set_ylabel('Intensity')
        
        # Evidenzia onset, apex e offset
        if 'onset' in micro_expressions:
            self.ax[1,0].axvline(x=micro_expressions['onset']['frame_idx'], 
                                color='g', linestyle='--', label='Onset')
        if 'apex' in micro_expressions:
            self.ax[1,0].axvline(x=micro_expressions['apex']['frame_idx'], 
                                color='r', linestyle='--', label='Apex')
        if 'offset' in micro_expressions:
            self.ax[1,0].axvline(x=micro_expressions['offset']['frame_idx'], 
                                color='b', linestyle='--', label='Offset')
        
        self.ax[1,0].legend()

    def _plot_muscle_activation(self, activation: Dict):
        """
        Visualizza la mappa di attivazione muscolare
        """
        # Crea heatmap delle attivazioni muscolari
        muscle_names = list(activation.keys())
        activation_values = np.array(list(activation.values())).reshape(1, -1)
        
        im = self.ax[1,1].imshow(activation_values, aspect='auto', cmap='hot')
        self.ax[1,1].set_title('Muscle Activation Map')
        self.ax[1,1].set_yticks([])
        self.ax[1,1].set_xticks(range(len(muscle_names)))
        self.ax[1,1].set_xticklabels(muscle_names, rotation=45)
        
        plt.colorbar(im, ax=self.ax[1,1])

    def _plot_mixed_expressions(self, mixed_data: Dict):
        """
        Visualizza l'analisi delle espressioni miste
        """
        # Split del plot in regioni facciali
        upper_data = mixed_data['upper_face']
        lower_data = mixed_data['lower_face']
        
        # Crea subplot per ogni regione
        gs = self.fig.add_gridspec(2, 2)
        upper_ax = self.fig.add_subplot(gs[0, :])
        lower_ax = self.fig.add_subplot(gs[1, :])
        
        # Visualizza attivazioni per regione
        self._plot_region_activation(upper_ax, upper_data, 'Upper Face')
        self._plot_region_activation(lower_ax, lower_data, 'Lower Face')
        
        # Evidenzia incongruenze
        if mixed_data['mixed_expressions']:
            self._highlight_incongruences(mixed_data['mixed_expressions'])