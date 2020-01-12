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
  # For now uses _first_ state in state array for each cell.
  # Excess states have no infleunce but are mutated just for fun.
  def rule_1d(neighbourhood):
    # http://mathworld.wolfram.com/ElementaryCellularAutomaton.html
    if len(neighbourhood) != 3:
      raise ValueError('neighbourhood must be 3 size')
    # Select rule index 0-7
    neighbourhood_state = neighbourhood[:, 0] > 0.5
    rule_index = np.packbits(np.append([0,0,0,0,0],
      neighbourhood_state.astype(int)))[0]
    mutation = 0.025
    state = neighbourhood[1] + (mutation *
        np.random.normal(size=neighbourhood[1].shape))
    if rule & (1 << rule_index):
      state[0] = 1.
    else:
      state[0] = 0.
    # state = state.clip(0, 1)
    # Mutate and clip ignored state to give red shift if rendered as RGB
    state[0] = state[0].clip(0, 1)
    if state.shape[0] > 1:
      state[1:] = state[1:].clip(0, 0.25)
    return state
  return rule_1d


class TestWorld(unittest.TestCase):
  def test_1D(self):
    rule30 = CreateBinaryLife1DRule(30)
    self.assertTrue(rule30(np.array([0,0,0]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([0,0,1]).reshape(3,1)) == 1)
    self.assertTrue(rule30(np.array([0,1,0]).reshape(3,1)) == 1)
    self.assertTrue(rule30(np.array([0,1,1]).reshape(3,1)) == 1)
    self.assertTrue(rule30(np.array([1,0,0]).reshape(3,1)) == 1)
    self.assertTrue(rule30(np.array([1,0,1]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([1,1,0]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([1,1,1]).reshape(3,1)) == 0)

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
