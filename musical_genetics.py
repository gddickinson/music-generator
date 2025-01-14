import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import random
from midiutil import MIDIFile
import networkx as nx
from scipy.stats import entropy
from collections import defaultdict

class MusicGene(Enum):
    MELODY = "melody"
    HARMONY = "harmony"
    RHYTHM = "rhythm"
    FORM = "form"
    TEXTURE = "texture"

@dataclass
class GeneticParams:
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_size: int = 2
    tournament_size: int = 3
    target_complexity: float = 0.5  # 0.0 to 1.0
    target_consonance: float = 0.6  # 0.0 to 1.0
    min_phrase_length: int = 8
    max_phrase_length: int = 32

@dataclass
class MusicalPhrase:
    melody: List[int]           # MIDI note numbers
    harmony: List[List[int]]    # Chord progression
    rhythm: List[float]         # Note durations
    dynamics: List[int]         # MIDI velocities
    fitness: float = 0.0

class MusicalGenetics:
    def __init__(self, params: GeneticParams):
        self.params = params
        self.population: List[MusicalPhrase] = []
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.setup_music_theory()

    def setup_music_theory(self):
        """Initialize music theory rules for fitness evaluation"""
        # Consonance ratings for intervals (0 to 1)
        self.consonance_ratings = {
            0: 1.0,   # Unison
            3: 0.6,   # Minor third
            4: 0.8,   # Major third
            5: 0.7,   # Perfect fourth
            7: 0.9,   # Perfect fifth
            8: 0.6,   # Minor sixth
            9: 0.7,   # Major sixth
            12: 0.9,  # Octave
        }

        # Scale degrees for different modes
        self.scale_degrees = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'dorian': [0, 2, 3, 5, 7, 9, 10],
            'mixolydian': [0, 2, 4, 5, 7, 9, 10]
        }

    def initialize_population(self) -> None:
        """Create initial population of musical phrases"""
        self.population = []
        for _ in range(self.params.population_size):
            phrase = self.generate_random_phrase()
            self.evaluate_fitness(phrase)
            self.population.append(phrase)

    def generate_random_phrase(self) -> MusicalPhrase:
        """Generate a random musical phrase"""
        length = random.randint(self.params.min_phrase_length,
                              self.params.max_phrase_length)

        # Generate melody
        melody = self.generate_random_melody(length)

        # Generate harmony
        harmony = self.generate_random_harmony(length)

        # Generate rhythm
        rhythm = self.generate_random_rhythm(length)

        # Generate dynamics
        dynamics = self.generate_random_dynamics(length)

        return MusicalPhrase(melody, harmony, rhythm, dynamics)

    def generate_random_melody(self, length: int) -> List[int]:
        """Generate random melody within musical constraints"""
        scale = self.scale_degrees['major']  # Could be parameterized
        base_note = 60  # Middle C
        melody = []
        current_note = random.choice(scale) + base_note

        for _ in range(length):
            # Choose next note with preference for small intervals
            intervals = [-12, -7, -5, -4, -2, 0, 2, 4, 5, 7, 12]
            weights = [0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
            interval = random.choices(intervals, weights=weights)[0]

            new_note = current_note + interval
            # Ensure note is in reasonable range
            while new_note < 48 or new_note > 84:
                if new_note < 48:
                    new_note += 12
                else:
                    new_note -= 12

            melody.append(new_note)
            current_note = new_note

        return melody

    def generate_random_harmony(self, length: int) -> List[List[int]]:
        """Generate random chord progression"""
        harmony = []
        chord_roots = [60, 62, 64, 65, 67, 69, 71]  # C major scale
        chord_types = [[0, 4, 7], [0, 3, 7], [0, 3, 6]]  # Major, minor, diminished

        for _ in range(length):
            root = random.choice(chord_roots)
            chord_type = random.choice(chord_types)
            chord = [root + interval for interval in chord_type]
            harmony.append(chord)

        return harmony

    def generate_random_rhythm(self, length: int) -> List[float]:
        """Generate random rhythm pattern"""
        durations = [0.25, 0.5, 1.0, 1.5, 2.0]
        weights = [0.2, 0.3, 0.3, 0.1, 0.1]
        return random.choices(durations, weights=weights, k=length)

    def generate_random_dynamics(self, length: int) -> List[int]:
        """Generate random dynamics (MIDI velocities)"""
        base_velocity = random.randint(70, 90)
        variation = 10
        return [max(0, min(127, base_velocity + random.randint(-variation, variation)))
                for _ in range(length)]

    def evaluate_fitness(self, phrase: MusicalPhrase) -> float:
        """Evaluate fitness of a musical phrase"""
        # Initialize fitness components
        melodic_fitness = self.evaluate_melodic_fitness(phrase.melody)
        harmonic_fitness = self.evaluate_harmonic_fitness(phrase.harmony)
        rhythmic_fitness = self.evaluate_rhythmic_fitness(phrase.rhythm)
        structural_fitness = self.evaluate_structural_fitness(phrase)

        # Weight the components
        weights = {
            'melodic': 0.3,
            'harmonic': 0.3,
            'rhythmic': 0.2,
            'structural': 0.2
        }

        # Calculate overall fitness
        fitness = (weights['melodic'] * melodic_fitness +
                  weights['harmonic'] * harmonic_fitness +
                  weights['rhythmic'] * rhythmic_fitness +
                  weights['structural'] * structural_fitness)

        phrase.fitness = fitness
        return fitness

    def evaluate_melodic_fitness(self, melody: List[int]) -> float:
        """Evaluate melodic qualities"""
        score = 0.0

        # Evaluate intervals
        for i in range(len(melody) - 1):
            interval = abs(melody[i+1] - melody[i])
            mod_interval = interval % 12

            # Reward consonant intervals
            if mod_interval in self.consonance_ratings:
                score += self.consonance_ratings[mod_interval]

            # Penalize large jumps
            if interval > 12:
                score -= 0.1 * (interval - 12)

        # Evaluate contour
        contour_score = self.evaluate_melodic_contour(melody)

        # Normalize and combine scores
        return 0.7 * (score / len(melody)) + 0.3 * contour_score

    def evaluate_harmonic_fitness(self, harmony: List[List[int]]) -> float:
        """Evaluate harmonic qualities"""
        score = 0.0

        # Evaluate chord progressions
        for i in range(len(harmony) - 1):
            # Calculate harmonic tension between consecutive chords
            tension = self.calculate_harmonic_tension(harmony[i], harmony[i+1])
            score += 1.0 - tension  # Lower tension = higher score

            # Evaluate voice leading
            voice_leading_score = self.evaluate_voice_leading(harmony[i], harmony[i+1])
            score += voice_leading_score

        return score / (2 * (len(harmony) - 1))  # Normalize

    def evaluate_rhythmic_fitness(self, rhythm: List[float]) -> float:
        """Evaluate rhythmic qualities"""
        # Calculate rhythmic complexity
        unique_durations = len(set(rhythm))
        duration_ratios = [rhythm[i+1]/rhythm[i] for i in range(len(rhythm)-1)]
        complexity = len(set(duration_ratios)) / len(duration_ratios)

        # Calculate rhythmic balance
        total_duration = sum(rhythm)
        duration_weights = [d/total_duration for d in rhythm]
        balance = 1.0 - abs(1.0 - entropy(duration_weights))

        return (complexity * self.params.target_complexity +
                balance * (1 - self.params.target_complexity))

    def evaluate_structural_fitness(self, phrase: MusicalPhrase) -> float:
        """Evaluate overall structural qualities"""
        # Check phrase length
        length_score = 1.0
        if len(phrase.melody) < self.params.min_phrase_length:
            length_score *= 0.5
        elif len(phrase.melody) > self.params.max_phrase_length:
            length_score *= 0.7

        # Evaluate melodic range
        range_score = self.evaluate_melodic_range(phrase.melody)

        # Evaluate structural symmetry
        symmetry_score = self.evaluate_symmetry(phrase)

        return (length_score + range_score + symmetry_score) / 3.0

    def evaluate_melodic_contour(self, melody: List[int]) -> float:
        """Evaluate the melodic contour"""
        # Calculate direction changes
        directions = [np.sign(melody[i+1] - melody[i])
                     for i in range(len(melody)-1)]
        direction_changes = sum(1 for i in range(len(directions)-1)
                              if directions[i] != directions[i+1])

        # Score based on variety of contour
        return min(1.0, direction_changes / (len(melody) * 0.5))

    def evaluate_melodic_range(self, melody: List[int]) -> float:
        """Evaluate if melodic range is appropriate"""
        range_size = max(melody) - min(melody)
        ideal_range = 24  # Two octaves
        return 1.0 - min(1.0, abs(range_size - ideal_range) / ideal_range)

    def evaluate_symmetry(self, phrase: MusicalPhrase) -> float:
        """Evaluate structural symmetry of the phrase"""
        length = len(phrase.melody)
        half_length = length // 2

        # Compare first and second half
        first_half = phrase.melody[:half_length]
        second_half = phrase.melody[half_length:2*half_length]

        # Calculate similarity while allowing transposition
        best_similarity = 0.0
        for transpose in range(-12, 13):
            transposed_second = [n + transpose for n in second_half]
            similarity = sum(1 for a, b in zip(first_half, transposed_second)
                           if abs(a - b) <= 2)
            best_similarity = max(best_similarity, similarity / half_length)

        return best_similarity

    def calculate_harmonic_tension(self, chord1: List[int], chord2: List[int]) -> float:
        """Calculate harmonic tension between two chords"""
        # Convert to pitch classes
        pc1 = set(n % 12 for n in chord1)
        pc2 = set(n % 12 for n in chord2)

        # Calculate shared tones
        common_tones = len(pc1.intersection(pc2))

        # Calculate dissonance
        dissonance = 0
        for n1 in pc1:
            for n2 in pc2:
                interval = abs(n1 - n2) % 12
                if interval not in self.consonance_ratings:
                    dissonance += 1

        return (dissonance / (len(pc1) * len(pc2)) +
                (1 - common_tones / max(len(pc1), len(pc2)))) / 2

    def evaluate_voice_leading(self, chord1: List[int], chord2: List[int]) -> float:
        """Evaluate voice leading between two chords"""
        score = 1.0

        # Sort chords from low to high
        chord1 = sorted(chord1)
        chord2 = sorted(chord2)

        # Check voice crossing
        if len(chord1) == len(chord2):
            for i in range(len(chord1)-1):
                if chord2[i] > chord2[i+1] or chord2[i] < chord1[i]:
                    score -= 0.2

        # Check parallel fifths and octaves
        for i in range(len(chord1)-1):
            for j in range(i+1, len(chord1)):
                interval1 = chord1[j] - chord1[i]
                if len(chord2) > j:
                    interval2 = chord2[j] - chord2[i]
                    if interval1 % 12 == interval2 % 12 and interval1 % 12 in [0, 7]:
                        score -= 0.3

        return max(0.0, score)

    def select_parent(self) -> MusicalPhrase:
        """Select parent using tournament selection"""
        tournament = random.sample(self.population, self.params.tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def crossover(self, parent1: MusicalPhrase, parent2: MusicalPhrase) -> MusicalPhrase:
        """Perform crossover between two parents"""
        if random.random() > self.params.crossover_rate:
            return parent1

        # Choose crossover points
        length = min(len(parent1.melody), len(parent2.melody))
        point1 = random.randint(0, length-2)
        point2 = random.randint(point1+1, length-1)

        # Create child with elements from both parents
        child_melody = (parent1.melody[:point1] +
                       parent2.melody[point1:point2] +
                       parent1.melody[point2:])

        child_harmony = (parent1.harmony[:point1] +
                        parent2.harmony[point1:point2] +
                        parent1.harmony[point2:])

        child_rhythm = (parent1.rhythm[:point1] +
                       parent2.rhythm[point1:point2] +
                       parent1.rhythm[point2:])

        child_dynamics = (parent1.dynamics[:point1] +
                         parent2.dynamics[point1:point2] +
                         parent1.dynamics[point2:])

        return MusicalPhrase(child_melody, child_harmony, child_rhythm, child_dynamics)

    def mutate(self, phrase: MusicalPhrase) -> None:
        """Apply mutations to a musical phrase"""
        if random.random() < self.params.mutation_rate:
            # Choose which aspect to mutate
            mutation_type = random.choice(['melody', 'harmony', 'rhythm', 'dynamics'])

            if mutation_type == 'melody':
                self.mutate_melody(phrase)
            elif mutation_type == 'harmony':
                self.mutate_harmony(phrase)
            elif mutation_type == 'rhythm':
                self.mutate_rhythm(phrase)
            else:
                self.mutate_dynamics(phrase)

    def mutate_melody(self, phrase: MusicalPhrase) -> None:
        """Apply melodic mutations"""
        mutation = random.choice([
            'transpose', 'invert', 'reverse', 'ornament', 'smooth'
        ])

        if mutation == 'transpose':
            # Transpose section
            start = random.randint(0, len(phrase.melody)-2)
            end = random.randint(start+1, len(phrase.melody))
            shift = random.randint(-12, 12)
            phrase.melody[start:end] = [n + shift for n in phrase.melody[start:end]]

        elif mutation == 'invert':
            # Invert a section
            start = random.randint(0, len(phrase.melody)-2)
            end = random.randint(start+1, len(phrase.melody))
            center = phrase.melody[start]
            phrase.melody[start:end] = [center - (n - center) for n in phrase.melody[start:end]]

        elif mutation == 'reverse':
            # Reverse a section
            start = random.randint(0, len(phrase.melody)-2)
            end = random.randint(start+1, len(phrase.melody))
            phrase.melody[start:end] = phrase.melody[start:end][::-1]

        elif mutation == 'ornament':
            # Add ornamental notes
            pos = random.randint(0, len(phrase.melody)-1)
            note = phrase.melody[pos]
            ornament = random.choice([
                [note, note+2, note],  # Turn
                [note, note+1, note],  # Mordent
                [note-1, note, note+1, note]  # Trill
            ])
            phrase.melody[pos:pos+1] = ornament

        elif mutation == 'smooth':
            # Smooth out large intervals
            for i in range(1, len(phrase.melody)-1):
                if abs(phrase.melody[i] - phrase.melody[i-1]) > 7:
                    phrase.melody[i] = (phrase.melody[i-1] + phrase.melody[i+1]) // 2

    def mutate_harmony(self, phrase: MusicalPhrase) -> None:
        """Apply harmonic mutations"""
        mutation = random.choice([
            'substitute', 'extend', 'simplify', 'reharmonize'
        ])

        if mutation == 'substitute':
            # Substitute a chord
            pos = random.randint(0, len(phrase.harmony)-1)
            root = phrase.harmony[pos][0]
            new_type = random.choice([[0,4,7], [0,3,7], [0,3,6], [0,4,7,10]])
            phrase.harmony[pos] = [root + interval for interval in new_type]

        elif mutation == 'extend':
            # Extend chord with additional notes
            pos = random.randint(0, len(phrase.harmony)-1)
            chord = phrase.harmony[pos]
            extension = random.choice([7, 9, 11, 13])
            phrase.harmony[pos].append(chord[0] + extension)

        elif mutation == 'simplify':
            # Simplify chord to triad
            pos = random.randint(0, len(phrase.harmony)-1)
            root = phrase.harmony[pos][0]
            phrase.harmony[pos] = [root, root+4, root+7]

        elif mutation == 'reharmonize':
            # Reharmonize a section
            start = random.randint(0, len(phrase.harmony)-2)
            end = random.randint(start+1, len(phrase.harmony))
            for i in range(start, end):
                root = phrase.harmony[i][0]
                new_type = random.choice([[0,4,7], [0,3,7], [0,3,6]])
                phrase.harmony[i] = [root + interval for interval in new_type]

    def mutate_rhythm(self, phrase: MusicalPhrase) -> None:
        """Apply rhythmic mutations"""
        mutation = random.choice([
            'augment', 'diminish', 'swing', 'regularize'
        ])

        if mutation == 'augment':
            # Double duration of section
            start = random.randint(0, len(phrase.rhythm)-2)
            end = random.randint(start+1, len(phrase.rhythm))
            phrase.rhythm[start:end] = [d * 2 for d in phrase.rhythm[start:end]]

        elif mutation == 'diminish':
            # Halve duration of section
            start = random.randint(0, len(phrase.rhythm)-2)
            end = random.randint(start+1, len(phrase.rhythm))
            phrase.rhythm[start:end] = [d * 0.5 for d in phrase.rhythm[start:end]]

        elif mutation == 'swing':
            # Add swing feel
            for i in range(0, len(phrase.rhythm)-1, 2):
                if phrase.rhythm[i] == phrase.rhythm[i+1]:
                    phrase.rhythm[i] *= 1.5
                    phrase.rhythm[i+1] *= 0.5

        elif mutation == 'regularize':
            # Make rhythms more regular
            common = max(set(phrase.rhythm), key=phrase.rhythm.count)
            for i in range(len(phrase.rhythm)):
                if random.random() < 0.3:
                    phrase.rhythm[i] = common

    def mutate_dynamics(self, phrase: MusicalPhrase) -> None:
        """Apply dynamic mutations"""
        mutation = random.choice([
            'crescendo', 'diminuendo', 'accent', 'normalize'
        ])

        if mutation == 'crescendo':
            # Add gradual increase
            start = random.randint(0, len(phrase.dynamics)-2)
            end = random.randint(start+1, len(phrase.dynamics))
            step = (phrase.dynamics[end-1] - phrase.dynamics[start]) / (end - start)
            for i in range(start, end):
                phrase.dynamics[i] = int(phrase.dynamics[start] + step * (i - start))

        elif mutation == 'diminuendo':
            # Add gradual decrease
            start = random.randint(0, len(phrase.dynamics)-2)
            end = random.randint(start+1, len(phrase.dynamics))
            step = (phrase.dynamics[end-1] - phrase.dynamics[start]) / (end - start)
            for i in range(start, end):
                phrase.dynamics[i] = int(phrase.dynamics[start] + step * (i - start))

        elif mutation == 'accent':
            # Add random accents
            for i in range(len(phrase.dynamics)):
                if random.random() < 0.2:
                    phrase.dynamics[i] = min(127, phrase.dynamics[i] + 20)

        elif mutation == 'normalize':
            # Normalize dynamics
            avg = sum(phrase.dynamics) / len(phrase.dynamics)
            phrase.dynamics = [int(avg) for _ in phrase.dynamics]

    def evolve(self) -> None:
        """Perform one generation of evolution"""
        new_population = []

        # Keep elite individuals
        elite = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population.extend(elite[:self.params.elite_size])

        # Generate remaining population
        while len(new_population) < self.params.population_size:
            parent1 = self.select_parent()
            parent2 = self.select_parent()
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            self.evaluate_fitness(child)
            new_population.append(child)

        self.population = new_population
        self.generation += 1

        # Record statistics
        fitnesses = [p.fitness for p in self.population]
        self.best_fitness_history.append(max(fitnesses))
        self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
        self.diversity_history.append(self.calculate_diversity())

    def calculate_diversity(self) -> float:
        """Calculate population diversity"""
        # Use average pairwise distance between melodies
        distances = []
        samples = min(10, len(self.population))  # Sample for efficiency
        for i in range(samples):
            for j in range(i+1, samples):
                dist = self.melodic_distance(
                    self.population[i].melody,
                    self.population[j].melody
                )
                distances.append(dist)
        return sum(distances) / len(distances) if distances else 0

    def melodic_distance(self, melody1: List[int], melody2: List[int]) -> float:
        """Calculate distance between two melodies"""
        # Use longest common subsequence as basis
        length = min(len(melody1), len(melody2))
        normalized_1 = [n % 12 for n in melody1[:length]]
        normalized_2 = [n % 12 for n in melody2[:length]]

        differences = sum(1 for a, b in zip(normalized_1, normalized_2) if a != b)
        return differences / length

    def visualize_evolution(self, filename: str = "evolution.png") -> None:
        """Create visualization of the evolutionary process"""
        plt.figure(figsize=(15, 10))

        # Plot fitness history
        plt.subplot(2, 1, 1)
        plt.plot(self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(self.avg_fitness_history, 'g-', label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Fitness Evolution')
        plt.legend()
        plt.grid(True)

        # Plot diversity history
        plt.subplot(2, 1, 2)
        plt.plot(self.diversity_history, 'r-', label='Population Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.title('Population Diversity')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def create_midi(self, phrase: MusicalPhrase,
                   filename: str = "evolved_music.mid") -> None:
        """Create MIDI file from evolved phrase"""
        midi = MIDIFile(2)  # 2 tracks - melody and harmony

        # Add melody track
        track = 0
        time = 0
        channel = 0
        midi.addTrackName(track, time, "Melody")
        midi.addTempo(track, time, 120)

        # Add melody notes
        for note, duration, velocity in zip(
            phrase.melody, phrase.rhythm, phrase.dynamics
        ):
            midi.addNote(track, channel, note, time, duration, velocity)
            time += duration

        # Add harmony track
        track = 1
        time = 0
        channel = 1
        midi.addTrackName(track, time, "Harmony")

        # Add harmony notes
        for chord, duration in zip(phrase.harmony, phrase.rhythm):
            for note in chord:
                midi.addNote(track, channel, note, time, duration, 80)
            time += duration

        # Save MIDI file
        with open(filename, "wb") as f:
            midi.writeFile(f)

if __name__ == "__main__":
    # Initialize with default parameters
    params = GeneticParams()
    genetics = MusicalGenetics(params)

    print("Initializing population...")
    genetics.initialize_population()

    # Run evolution
    print("\nEvolving music...")
    for generation in range(params.generations):
        genetics.evolve()
        if generation % 10 == 0:
            print(f"Generation {generation}: " +
                  f"Best Fitness = {genetics.best_fitness_history[-1]:.4f}")

    # Get best phrase
    best_phrase = max(genetics.population, key=lambda x: x.fitness)

    # Create output files
    print("\nCreating output files...")
    genetics.create_midi(best_phrase)
    genetics.visualize_evolution()

    print("\nGenerated files:")
    print("- evolved_music.mid (MIDI file of best evolved phrase)")
    print("- evolution.png (Visualization of evolutionary process)")
    print("\nEvolution complete!")
