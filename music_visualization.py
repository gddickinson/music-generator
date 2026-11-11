"""Music visualization: form graphs, tension plots, and voice leading diagrams."""

from typing import List

import matplotlib.pyplot as plt
import networkx as nx

from fractal_melody_generator import EmotionalStyle
from musical_types import (
    TextureType, Form, Section
)


class Visualization:
    """Advanced music visualization system."""

    def __init__(self, debug_mode: bool = True):
        self.debug_mode = debug_mode

        self.texture_colors = {
            TextureType.MONOPHONIC: '#1f77b4',
            TextureType.HOMOPHONIC: '#2ca02c',
            TextureType.POLYPHONIC: '#d62728',
            TextureType.HETEROPHONIC: '#9467bd',
            TextureType.CONTRAPUNTAL: '#ff7f0e'
        }

        self.emotion_colors = {
            EmotionalStyle.PEACEFUL: '#ADD8E6',
            EmotionalStyle.ENERGETIC: '#FF6B6B',
            EmotionalStyle.MELANCHOLIC: '#4B0082',
            EmotionalStyle.JOYFUL: '#FFD700',
            EmotionalStyle.DARK: '#2F4F4F',
            EmotionalStyle.ETHEREAL: '#E6E6FA'
        }

        self.output_directory = '.'
        if self.debug_mode:
            print("Visualization system initialized")
            print(f"Using output directory: {self.output_directory}")

    def create_form_graph(self, form: Form, filename: str = 'form_structure.png') -> None:
        """Create visualization of musical form structure."""
        if self.debug_mode:
            print(f"\nCreating form visualization with {len(form.sections)} sections")

        G = nx.DiGraph()

        for i, section in enumerate(form.sections):
            G.add_node(f"Section {i}",
                      emotion=section.emotion.value,
                      texture=section.texture.value,
                      key=section.key)
            if self.debug_mode:
                print(f"Added section {i}: {section.emotion.value} in {section.key}")

        for i in range(len(form.sections) - 1):
            G.add_edge(f"Section {i}", f"Section {i+1}")

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G)

        nx.draw_networkx_nodes(G, pos,
                             node_color=[self.emotion_colors[section.emotion]
                                       for section in form.sections],
                             node_size=2000)

        edges = [(f"Section {i}", f"Section {i+1}")
                for i in range(len(form.sections)-1)]
        edge_colors = [self.texture_colors[form.sections[i].texture]
                      for i in range(len(form.sections)-1)]

        nx.draw_networkx_edges(G, pos,
                             edge_color=edge_colors,
                             width=2,
                             alpha=0.7)

        labels = {f"Section {i}": (f"Section {i}\n{section.emotion.value}\n"
                                  f"{section.key}\n{section.texture.value}")
                 for i, section in enumerate(form.sections)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)

        plt.title(f"Musical Form: {form.name}")
        plt.axis('off')

        emotion_patches = [plt.Rectangle((0,0),1,1, fc=color)
                         for color in self.emotion_colors.values()]
        texture_patches = [plt.Rectangle((0,0),1,1, fc=color)
                         for color in self.texture_colors.values()]

        plt.legend(emotion_patches + texture_patches,
                  [e.value for e in EmotionalStyle] +
                  [t.name for t in TextureType],
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(f"{self.output_directory}/{filename}",
                   bbox_inches='tight',
                   dpi=300)
        plt.close()

        if self.debug_mode:
            print(f"Saved form visualization to {filename}")

    def plot_tension_graph(self, sections: List[Section],
                         durations: List[float],
                         tension_values: List[float],
                         filename: str = 'tension_graph.png') -> None:
        """Create visualization of musical tension over time."""
        if self.debug_mode:
            print("\nCreating tension graph visualization")

        plt.figure(figsize=(15, 5))

        current_time = 0
        times = []
        tensions = []
        colors = []

        for section, duration, tension in zip(sections, durations, tension_values):
            times.extend([current_time, current_time + duration])
            tensions.extend([tension, tension])
            colors.extend([self.emotion_colors[section.emotion]]*2)
            current_time += duration

        plt.plot(times, tensions, 'b-', label='Musical Tension', zorder=2)

        for i in range(0, len(times)-1, 2):
            plt.axvspan(times[i], times[i+1],
                       color=colors[i],
                       alpha=0.3,
                       zorder=1)

        plt.xlabel('Time (beats)')
        plt.ylabel('Tension')
        plt.title('Musical Tension Over Time')
        plt.grid(True, alpha=0.3, zorder=0)

        emotion_patches = [plt.Rectangle((0,0),1,1, fc=color, alpha=0.3)
                         for color in self.emotion_colors.values()]
        plt.legend(emotion_patches + [plt.Line2D([0], [0], color='b')],
                  [e.value for e in EmotionalStyle] + ['Tension'],
                  loc='center left',
                  bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(f"{self.output_directory}/{filename}",
                   bbox_inches='tight',
                   dpi=300)
        plt.close()

        if self.debug_mode:
            print(f"Saved tension graph to {filename}")

    def plot_voice_leading(self, voices: List[List[int]],
                         filename: str = 'voice_leading.png') -> None:
        """Visualize voice leading between parts."""
        if self.debug_mode:
            print("\nCreating voice leading visualization")

        plt.figure(figsize=(15, 8))

        for i, voice in enumerate(voices):
            plt.plot(voice,
                    label=f'Voice {i+1}',
                    marker='o',
                    linestyle='-',
                    alpha=0.7)

        plt.xlabel('Time Step')
        plt.ylabel('MIDI Note')
        plt.title('Voice Leading Visualization')
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.output_directory}/{filename}",
                   bbox_inches='tight',
                   dpi=300)
        plt.close()

        if self.debug_mode:
            print(f"Saved voice leading visualization to {filename}")
