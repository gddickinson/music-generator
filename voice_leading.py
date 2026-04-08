"""Voice leading optimization for chord progressions."""

from typing import List


class VoiceLeading:
    """Advanced voice leading handler."""

    def __init__(self):
        self.consonant_intervals = {0, 3, 4, 7, 8, 9}
        self.perfect_consonances = {0, 7}

    def optimize_progression(self, current_chord: List[int],
                           next_chord: List[int]) -> List[int]:
        """Optimize voice leading between chords."""
        best_voicing = next_chord
        min_movement = float('inf')

        for voicing in self.generate_voicings(next_chord):
            total_movement = sum(abs(c - n) for c, n in zip(current_chord, voicing))

            if self.check_voice_crossing(voicing):
                total_movement += 10

            if self.check_parallel_motion(current_chord, voicing):
                total_movement += 5

            if total_movement < min_movement:
                min_movement = total_movement
                best_voicing = voicing

        return best_voicing

    def generate_voicings(self, chord: List[int]) -> List[List[int]]:
        """Generate possible voicings for a chord."""
        voicings = []
        base_voicing = sorted(chord)

        voicings.append(base_voicing)

        for i in range(len(chord)):
            up = base_voicing.copy()
            up[i] += 12
            down = base_voicing.copy()
            down[i] -= 12
            voicings.extend([up, down])

        return voicings

    def check_voice_crossing(self, voicing: List[int]) -> bool:
        """Check for voice crossing."""
        return any(v1 > v2 for v1, v2 in zip(voicing[:-1], voicing[1:]))

    def check_parallel_motion(self, chord1: List[int],
                            chord2: List[int]) -> bool:
        """Check for parallel fifths and octaves."""
        for i in range(len(chord1) - 1):
            for j in range(i + 1, len(chord1)):
                interval1 = abs(chord1[i] - chord1[j]) % 12
                interval2 = abs(chord2[i] - chord2[j]) % 12
                if interval1 == interval2 and interval1 in [0, 7]:
                    return True
        return False
