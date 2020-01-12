import numpy as np
from rules import GOL2D
import rules
import unittest
from world import World1D, World2D

class Test1D(unittest.TestCase):
  def test_binary_1d(self):
    world = World1D(5)
    # Rule 158 = 10011110
    # 111 => 1
    # 110 => 0
    # 101 => 0
    # 100 => 1
    # 011 => 1
    # 010 => 1
    # 001 => 1
    # 000 => 0
    world.set_rules(rules.CreateBinaryLife1DRule(158))
    init = np.array([0,0,1,0,0])
    world.set_states(init)
    state = world.cells
    self.assertTrue((np.expand_dims(init, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([0,1,1,1,0])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([1,1,1,0,1])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([1,1,0,0,1])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([1,0,1,1,1])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([1,0,1,1,0])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([1,0,1,0,1])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())
    world.step()
    state = world.cells
    target = np.array([1,0,1,0,1])
    self.assertTrue((np.expand_dims(target, axis=1) == state).all())

class TestGOL2D(unittest.TestCase):
  def test_glider_2d(self):
    world = World2D((4,4))
    world.set_rules(rules.GOL2D)
    init = np.array([
      [0,0,1,0],
      [1,0,1,0],
      [0,1,1,0],
      [0,0,0,0]])
    world.set_states(init)
    state = world.cells
    self.assertTrue((init.reshape((4,4,1)) == state).all())
    world.step()
    state = world.cells
    target = np.array([
      [0,1,0,0],
      [0,0,1,1],
      [0,1,1,0],
      [0,0,0,0]])
    self.assertTrue((target.reshape((4,4,1)) == state).all())



if __name__ == '__main__':
  unittest.main()
