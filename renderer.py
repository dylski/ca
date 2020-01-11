"""Renderers take in, store and render generation data."""

from PIL import Image
import numpy as np
import pygame
from pygame import surfarray
import unittest

class Renderer():
  def __init__(self):
    pass

  def next_gen(self, data):
    # Add state for next generation
    assert(False)

  def display(self, data):
    # Display stored data
    assert(False)
    pass


class Renderer1D(Renderer):
  def __init__(self, num_cells, history_size, display_size, state_depth=1,
      testing=False):
    pygame.init()
    self._num_cells = num_cells
    self._history_size = history_size
    self._display_size = display_size
    self._testing = testing
    self._state_depth = state_depth

    self._bitmap_size = tuple(list(display_size) + [3])
    self._state_size = (self._num_cells, self._history_size, state_depth)

    if not self._testing:
      self._display = pygame.display.set_mode(display_size)

    # note: x = axis 0.
    self._bitmap = np.zeros(self._bitmap_size, dtype=np.uint8)
    self._history = np.zeros(self._state_size)

  def next_gen(self, data):
    self._history = np.roll(self._history, 1, axis=1)
    # Indexing with list preserves dimensioality instead of squeezing result
    # e.g. result will be (4,1,3) not squeezed to (4,3)
    # Also insert time dimension into data
    self._history[:, [0]] = np.expand_dims(data, axis=1)
    # self._history[:, [0]] = data

  def display(self):
    self._bitmap = np.zeros((self._num_cells, self._history_size, 3),
        dtype=np.uint8)
    if self._state_depth == 1:
      # Broadcast over RGB
      self._bitmap[:, :] = self._history * 255
    elif self._state_depth == 3:
      self._bitmap = np.array(self._history * 255).astype(np.uint8)
    else:
      raise ValueError('Unsupported state depth {}'.format(self._state_depth))
    self._bitmap = np.array(Image.fromarray(self._bitmap).resize(
      (self._display_size[1], self._display_size[0])))
    if not self._testing:
      surfarray.blit_array(self._display, self._bitmap)
      pygame.display.flip()
    else:
      return self._bitmap

class TestRenderer(unittest.TestCase):
  def test_1D_sizes(self):
    renderer = Renderer1D(num_cells=2, history_size=2, display_size=(320, 240),
        testing=True)
    self.assertTrue(renderer._bitmap_size == (320, 240, 3))
    self.assertTrue(renderer._bitmap.shape == (320, 240, 3))

  def test_1D_history(self):
    renderer = Renderer1D(num_cells=4, history_size=2, display_size=(2,1),
        testing=True)
    self.assertTrue((renderer._history == 0).all())
    new_gen = np.array([1,2,3,4]).reshape((4,1))
    renderer.next_gen(new_gen)
    self.assertTrue(
        (renderer._history[:, [0]] == new_gen.reshape((4,1,1))).all())
    new_gen_1 = np.array([2,3,4,5]).reshape((4,1))
    renderer.next_gen(new_gen_1)
    self.assertTrue(
        (renderer._history[:, [0]] == new_gen_1.reshape((4,1,1))).all())
    self.assertTrue(
        (renderer._history[:, [1]] == new_gen.reshape((4,1,1))).all())
    # renderer.display()

  def test_1D_bitmap(self):
    renderer = Renderer1D(num_cells=4, history_size=2, display_size=(4,2),
        testing=True)
    self.assertTrue((renderer._history == 0).all())
    new_gen = np.array([0,0,1,0]).reshape((4,1))
    renderer.next_gen(new_gen)
    bitmap = renderer.display()
    self.assertTrue(bitmap.shape == (4,2,3))
    self.assertTrue((bitmap[:, 0, :] == np.array([
      [0, 0, 0],
      [0, 0, 0],
      [255, 255, 255],
      [0, 0, 0]])).all())
    self.assertTrue((bitmap[:, 1, :] == np.array([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]])).all())

  def test_1D_rgb_bitmap(self):
    renderer = Renderer1D(num_cells=4, history_size=2, display_size=(4,2),
        state_depth=3, testing=True)
    self.assertTrue((renderer._history == 0).all())
    new_gen = np.zeros((4,3))
    new_gen[0,:] = [0.9, 0.1, 0.1]
    new_gen[1,:] = [0.1, 0.9, 0.1]
    new_gen[2,:] = [0.1, 0.1, 0.9]
    new_gen[3,:] = [0.5, 0.5, 0.5]
    renderer.next_gen(new_gen)
    bitmap = renderer.display()
    self.assertTrue(bitmap.shape == (4,2,3))
    self.assertTrue((bitmap[:, 0, :] == np.array([
      [229, 25, 25],
      [25, 229, 25],
      [25, 25, 229],
      [127, 127, 127]])).all())
    self.assertTrue((bitmap[:, 1, :] == np.array([
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0],
      [0, 0, 0]])).all())


if __name__ == '__main__':
  unittest.main()
