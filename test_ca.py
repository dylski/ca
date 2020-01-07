import numpy as np
from rules import GOL2D
import rules
import unittest
from world import World1D, World2D

class Test1D(unittest.TestCase):
  def test_binary_1d(self):
    world = World1D(5)
    world.set_rules(rules.CreateBinaryLife1DRule(158))
    init = np.array([0,0,1,0,0])
    world.set_world(init)
    state = world.world
    self.assertTrue((init == state).all())
    world.step()
    state = world.world
    target = np.array([0,1,1,1,0])
    self.assertTrue((target == state).all())
    world.step()
    state = world.world
    target = np.array([1,1,1,0,1])
    self.assertTrue((target == state).all())

class TestGOL2D(unittest.TestCase):
  def test_glider_2d(self):
    world = World2D((4,4))
    world.set_rules(rules.GOL2D)
    init = np.array([
      [0,0,1,0],
      [1,0,1,0],
      [0,1,1,0],
      [0,0,0,0]])
    world.set_world(init)
    state = world.world
    self.assertTrue((init == state).all())
    world.step()
    state = world.world
    target = np.array([
      [0,1,0,0],
      [0,0,1,1],
      [0,1,1,0],
      [0,0,0,0]])
    self.assertTrue((target == state).all())



if __name__ == '__main__':
  unittest.main()
