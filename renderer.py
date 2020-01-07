from PIL import Image
import numpy as np
import pygame
from pygame import surfarray
import unittest

class Renderer():
  def __init__(self):
    pass

  def render(self, data):
    pass


class Renderer1D(Renderer):
  def __init__(self, num_cells, history_size, display_size, state_depth=1,
      testing=False):
    pygame.init()
    self._num_cells = num_cells
    self._history_size = history_size
    self._display_size = display_size
    self._testing = testing

    self._bitmap_size = tuple(list(display_size) + [3])
    self._state_size = (self._num_cells, self._history_size, state_depth)

    if not self._testing:
      self._display = pygame.display.set_mode(display_size)

    # note: x = axis 0.
    self._bitmap = np.zeros(self._bitmap_size, dtype=np.uint8)
    self._history = np.zeros(self._state_size)

  def next_gen(self, data):
    self._history = np.roll(self._history, 1, axis=1)
    self._history[:, 0] = np.expand_dims(data, axis=1)

  def display(self):
    self._bitmap = np.zeros((self._num_cells, self._history_size, 3), dtype=np.uint8)
    self._bitmap[:, :] = self._history * 255
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
    new_gen = np.array([1,2,3,4])
    renderer.next_gen(new_gen)
    self.assertTrue(
        (renderer._history[:, 0] == np.expand_dims(new_gen, axis=1)).all())
    new_gen_1 = np.array([2,3,4,5])
    renderer.next_gen(new_gen_1)
    self.assertTrue(
        (renderer._history[:, 0] == np.expand_dims(new_gen_1, axis=1)).all())
    self.assertTrue(
        (renderer._history[:, 1] == np.expand_dims(new_gen, axis=1)).all())

  def test_1D_bitmap(self):
    renderer = Renderer1D(num_cells=4, history_size=2, display_size=(4,2),
        testing=True)
    self.assertTrue((renderer._history == 0).all())
    new_gen = np.array([0,0,1,0])
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


if __name__ == '__main__':
  unittest.main()
