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

def CreateBinaryLife1DRule(rule):
  def rule_1d(neighbourhood):
    # http://mathworld.wolfram.com/ElementaryCellularAutomaton.html
    if len(neighbourhood) != 3:
      raise ValueError('neighbourhood must be 3 size')
    # Select rule index 0-7
    rule_index = np.packbits(np.append([0,0,0,0,0],
      neighbourhood.astype(int)))[0]
    if rule & (1 << rule_index):
      return 1
    else:
      return 0
  return rule_1d


class TestWorld(unittest.TestCase):
  def test_1D(self):
    rule30 = CreateBinaryLife1DRule(30)
    self.assertTrue(rule30(np.array([0,0,0])) == 0)
    self.assertTrue(rule30(np.array([0,0,1])) == 1)
    self.assertTrue(rule30(np.array([0,1,0])) == 1)
    self.assertTrue(rule30(np.array([0,1,1])) == 1)
    self.assertTrue(rule30(np.array([1,0,0])) == 1)
    self.assertTrue(rule30(np.array([1,0,1])) == 0)
    self.assertTrue(rule30(np.array([1,1,0])) == 0)
    self.assertTrue(rule30(np.array([1,1,1])) == 0)

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
