#!/usr/bin/env python3
"""
Generate test audio files for WebRTC testing.

Creates simple test audio with spoken text using TTS or sine wave tones.
"""

import argparse
import numpy as np
import wave


def generate_sine_wave_audio(output_file: str, duration: float = 5.0, frequency: float = 440.0, sample_rate: int = 16000):
    """Generate a simple sine wave audio file."""
    print(f"Generating {duration}s sine wave at {frequency}Hz...")

    # Generate sine wave
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = np.sin(2 * np.pi * frequency * t)

    # Add envelope (fade in/out)
    fade_samples = int(0.1 * sample_rate)  # 100ms fade
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    audio[:fade_samples] *= fade_in
    audio[-fade_samples:] *= fade_out

    # Convert to int16
    audio_int16 = (audio * 32767 * 0.5).astype(np.int16)  # 50% volume

    # Write WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"✓ Saved to {output_file}")


def generate_speech_pattern_audio(output_file: str, duration: float = 5.0, sample_rate: int = 16000):
    """Generate audio with speech-like patterns (varying frequency/amplitude)."""
    print(f"Generating {duration}s speech-pattern audio...")

    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create speech-like patterns with varying frequency and amplitude
    # Simulate phonemes with different frequencies
    audio = np.zeros_like(t)

    # Divide into segments (simulating syllables)
    segment_duration = 0.2  # 200ms per syllable
    num_segments = int(duration / segment_duration)

    for i in range(num_segments):
        start_idx = int(i * segment_duration * sample_rate)
        end_idx = int((i + 1) * segment_duration * sample_rate)

        if end_idx > len(t):
            break

        # Vary frequency (simulating different phonemes)
        base_freq = 150 + (i % 5) * 50  # 150-350 Hz

        # Add harmonics for more natural sound
        segment_t = t[start_idx:end_idx]
        segment_audio = (
            0.6 * np.sin(2 * np.pi * base_freq * segment_t) +
            0.3 * np.sin(2 * np.pi * base_freq * 2 * segment_t) +
            0.1 * np.sin(2 * np.pi * base_freq * 3 * segment_t)
        )

        # Add amplitude envelope (syllable stress)
        envelope = np.sin(np.pi * np.linspace(0, 1, len(segment_t)))
        segment_audio *= envelope

        audio[start_idx:end_idx] = segment_audio

    # Add pauses every 1 second (simulating word boundaries)
    pause_duration = 0.1  # 100ms pause
    for i in range(int(duration)):
        pause_start = int((i + 0.9) * sample_rate)
        pause_end = int((i + 1.0) * sample_rate)
        if pause_end < len(audio):
            audio[pause_start:pause_end] = 0

    # Normalize and convert to int16
    audio = audio / np.abs(audio).max()
    audio_int16 = (audio * 32767 * 0.5).astype(np.int16)

    # Write WAV file
    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"✓ Saved to {output_file}")


def generate_counting_audio(output_file: str, count_to: int = 10, sample_rate: int = 16000):
    """Generate audio simulating counting with distinct syllables."""
    print(f"Generating counting audio (1 to {count_to})...")

    syllable_duration = 0.3  # 300ms per number
    pause_duration = 0.5     # 500ms between numbers

    total_duration = count_to * (syllable_duration + pause_duration)
    audio = np.zeros(int(total_duration * sample_rate))

    for i in range(count_to):
        start_time = i * (syllable_duration + pause_duration)
        start_idx = int(start_time * sample_rate)
        syllable_samples = int(syllable_duration * sample_rate)

        # Generate tone for this number
        t = np.linspace(0, syllable_duration, syllable_samples, endpoint=False)
        frequency = 200 + (i % 5) * 40  # Varying pitch

        syllable = 0.5 * np.sin(2 * np.pi * frequency * t)

        # Add envelope
        envelope = np.sin(np.pi * np.linspace(0, 1, syllable_samples))
        syllable *= envelope

        audio[start_idx:start_idx + syllable_samples] = syllable

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    with wave.open(output_file, 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    print(f"✓ Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate test audio for WebRTC testing')
    parser.add_argument('--type', choices=['sine', 'speech', 'counting'], default='speech',
                        help='Type of audio to generate')
    parser.add_argument('--output', default='test_audio.wav', help='Output file path')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration in seconds')
    parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate')

    args = parser.parse_args()

    if args.type == 'sine':
        generate_sine_wave_audio(args.output, args.duration, sample_rate=args.sample_rate)
    elif args.type == 'speech':
        generate_speech_pattern_audio(args.output, args.duration, sample_rate=args.sample_rate)
    elif args.type == 'counting':
        generate_counting_audio(args.output, count_to=10, sample_rate=args.sample_rate)

    print(f"\nTest audio file created: {args.output}")
    print(f"  Sample rate: {args.sample_rate} Hz")
    print(f"  Duration: {args.duration:.1f} seconds")
    print(f"\nYou can now test with:")
    print(f"  python webrtc/test_client.py --audio {args.output} --image path/to/avatar.jpg")


if __name__ == '__main__':
    main()
