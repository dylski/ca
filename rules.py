import numpy as np
import unittest

def GOL2D(neighbourhood):
  #print(neighbourhood)
  #import pdb; pdb.set_trace()
  living = neighbourhood.sum()
  if neighbourhood[
      int(np.floor(neighbourhood.shape[0] / 2.)),
      int(np.floor(neighbourhood.shape[1] / 2.))] == 0.0:
    # dead
    if living == 3:
      return 1
    else:
      return 0
  else:
    # alive
    living -= 1
    if living == 2 or living == 3:
      return 1
    else:
      return 0


class TestWorld(unittest.TestCase):
  def test_GOL2D(self):
    self.assertTrue(GOL2D(np.array([
      [0,0,0],
      [0,1,0],
      [0,0,0]])) == 0)
    self.assertTrue(GOL2D(np.array([
      [0,0,0],
      [0,1,1],
      [0,0,1]])) == 1)
    self.assertTrue(GOL2D(np.array([
      [0,1,0],
      [1,1,0],
      [1,0,0]])) == 1)
    self.assertTrue(GOL2D(np.array([
      [0,0,1],
      [0,1,1],
      [0,1,1]])) == 0)
    self.assertTrue(GOL2D(np.array([
      [0,0,0],
      [0,0,0],
      [0,0,0]])) == 0)
    self.assertTrue(GOL2D(np.array([
      [0,0,0],
      [0,0,1],
      [0,0,0]])) == 0)
    self.assertTrue(GOL2D(np.array([
      [0,1,0],
      [1,0,0],
      [0,0,0]])) == 0)
    self.assertTrue(GOL2D(np.array([
      [0,0,0],
      [1,0,1],
      [0,0,1]])) == 1)
    self.assertTrue(GOL2D(np.array([
      [1,0,0],
      [1,0,1],
      [0,0,1]])) == 0)

if __name__ == '__main__':
  unittest.main()
