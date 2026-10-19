"""Tests for audio_io."""

import os
import tempfile
import unittest

import numpy as np
from perch_hoplite import audio_io
import soundfile


class AudioIoTest(unittest.TestCase):

  def test_load_audio_file_stereo_resampled_to_mono(self):
    # Create temporary stereo audio data (1000 samples, 2 channels)
    stereo_data = np.zeros((1000, 2), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
      temp_path = f.name
      soundfile.write(temp_path, stereo_data, samplerate=16000)

    try:
      # Resampling stereo audio should return a 1D mono array
      audio = audio_io.load_audio_file(temp_path, target_sample_rate=22050)
      self.assertEqual(len(audio.shape), 1)
      self.assertGreater(audio.shape[0], 0)
    finally:
      os.unlink(temp_path)

  def test_load_audio_file_mono_unchanged(self):
    # Verify standard mono audio still works correctly
    mono_data = np.zeros((1000,), dtype=np.float32)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
      temp_path = f.name
      soundfile.write(temp_path, mono_data, samplerate=16000)

    try:
      audio = audio_io.load_audio_file(temp_path, target_sample_rate=22050)
      self.assertEqual(len(audio.shape), 1)
      self.assertGreater(audio.shape[0], 0)
    finally:
      os.unlink(temp_path)


if __name__ == '__main__':
  unittest.main()
