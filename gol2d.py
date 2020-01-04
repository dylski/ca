import numpy as np
from rules import GOL2D
import unittest
from world import World2D

class Test_GOL2D(unittest.TestCase):
  def test_glider(self):
    world = World2D((4,4))
    world.set_rules(GOL2D)
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
