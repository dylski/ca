import numpy as np
import unittest

class World():
  '''
  Cell state grids for 1D, 2D and 3D CAs.
  Grid padding of 1 with dead cells.
  '''
  def __init__(self, dim):
    self._dim = dim
    self._world_1 = np.zeros(np.array(self._dim) + 2)
    self._world_2 = np.zeros(np.array(self._dim) + 2)
    self._world = self._world_1
    self._world_next = self._world_2

  def set_world(self, world):
    pass

  def neighbourhood(self, cell_coords):
    pass

  def set_rules(self, rules):
    self._rules = rules

  def step(self):
    self._step()
    self._flip_worlds()
    pass

  @property
  def world(self):
    pass
    return self._world

  def _step(self):
    # Override with specific step
    pass

  def _flip_worlds(self):
    _world_old = self._world
    self._world = self._world_next
    self._world_next = _world_old

  def set_state(self, cell_coords):
    pass


class World1D(World):
  '''
  Cell state grids for 1D, 2D and 3D CAs.
  Grid padding of 1 with dead cells.
  '''

  def set_world(self, world):
    shape = world.shape
    if shape != self._dim:
      raise ValueError('Dim mismatch {} !+ {}'.format(len(shape), self._dim))
    self._world[1:world.shape[0] + 1] = world

  @property
  def world(self):
    return self._world[1:self._dim[0] + 1]

  def neighbourhood(self, coord):
    return self._world[coord - 1: coord + 2]

  def set_state(self, coord):
    pass


class World2D(World):
  '''
  Cell state grids for 1D, 2D and 3D CAs.
  Grid padding of 1 with dead cells.
  '''

  def set_world(self, world):
    shape = world.shape
    if shape != self._dim:
      raise ValueError('Dim mismatch {} !+ {}'.format(len(shape), self._dim))
    self._world[
        1:world.shape[0] + 1,
        1:world.shape[1] + 1,
        ] = world

  @property
  def world(self):
    return self._world[
        1:self._dim[0] + 1,
        1:self._dim[1] + 1]

  def neighbourhood(self, coords):
    if len(coords) != len(self._world.shape):
      raise ValueError('len(coords) != len(self._world.shape)')

    return self._world[
        coords[0] - 1: coords[0] + 2,
        coords[1] - 1: coords[1] + 2,
        ]

  def _step(self):
    for i in range(1, self._dim[0] + 1):
      for j in range(1, self._dim[1] + 1):
        self._world_next[i, j] = self._rules(self.neighbourhood((i,j)))

  def set_state(self, cell_coords):
    coords = np.array(cell_coords) + 1
    pass


class World3D(World):
  '''
  Cell state grids for 1D, 2D and 3D CAs.
  Grid padding of 1 with dead cells.
  '''

  def set_world(self, world):
    shape = world.shape
    if shape != self._dim:
      raise ValueError('Dim mismatch {} !+ {}'.format(len(shape), self._dim))
    self._world[
        1:world.shape[0] + 1,
        1:world.shape[1] + 1,
        1:world.shape[2] + 1,
        ] = world

  @property
  def world(self):
    return self._world[
        1:self._dim[0] + 1,
        1:self._dim[1] + 1,
        1:self._dim[2] + 1]

  def neighbourhood(self, coords):
    if len(coords) != len(self._world.shape):
      raise ValueError('len(coords) != len(self._world.shape)')

    return self._world[
        coords[0] - 1: coords[0] + 2,
        coords[1] - 1: coords[1] + 2,
        coords[2] - 1: coords[2] + 2,
        ]

  def set_state(self, cell_coords):
    coords = np.array(cell_coords) + 1
    pass


class TestWorld(unittest.TestCase):
  def test_neighbours_1D(self):
    _world = World1D(dim=(4,))
    _world.set_world(np.array([0, 1, 2, 3]))
    neighbours = _world.neighbourhood(2)
    self.assertTrue((neighbours == np.array([0, 1, 2])).all())
    neighbours = _world.neighbourhood(1)
    self.assertTrue((neighbours == np.array([0, 0, 1])).all())
    neighbours = _world.neighbourhood(4)
    self.assertTrue((neighbours == np.array([2, 3, 0])).all())

  def test_neighbours_2D(self):
    _world = World2D(dim=(4,4))
    world = np.array(range(16)).reshape((4,4))
    _world.set_world(world)
    neighbours = _world.neighbourhood((1, 1))
    self.assertTrue((neighbours == np.array([
      [0, 0, 0],
      [0, 0, 1],
      [0, 4, 5],
      ])).all())
    neighbours = _world.neighbourhood([2, 2])
    self.assertTrue((neighbours == np.array([
      [0, 1, 2],
      [4, 5, 6],
      [8, 9, 10]
      ])).all())
    neighbours = _world.neighbourhood([4, 4])
    self.assertTrue((neighbours == np.array(
      [[10, 11, 0],
       [14, 15, 0],
       [0, 0, 0]]
      )).all())

  def test_neighbours_3D(self):
    _world = World3D(dim=(4,4,4))
    world = np.array(range(4*4*4)).reshape((4,4,4))
    _world.set_world(world)
    neighbours = _world.neighbourhood((1, 1, 1))
    self.assertTrue((neighbours == np.array([
      [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      [
        [0, 0, 0],
        [0, 0, 1],
        [0, 4, 5],
      ],
      [
        [0, 0, 0],
        [0, 16, 17],
        [0, 20, 21],
      ],
      ])).all())
    neighbours = _world.neighbourhood([2, 2, 2])
    self.assertTrue((neighbours == np.array([
      [
        [0, 1, 2],
        [4, 5, 6],
        [8, 9, 10],
      ],
      [
        [16, 17, 18],
        [20, 21, 22],
        [24, 25, 26],
      ],
      [
        [32, 33, 34],
        [36, 37, 38],
        [40, 41, 42],
      ],
      ])).all())
    neighbours = _world.neighbourhood([4, 4, 4])
    self.assertTrue((neighbours == np.array([
      [
        [42, 43, 0],
        [46, 47, 0],
        [0, 0, 0],
      ],
      [
        [58, 59, 0],
        [62, 63, 0],
        [0, 0, 0],
      ],
      [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ],
      ])).all())


if __name__ == '__main__':
  unittest.main()
