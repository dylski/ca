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

def CreateBinaryLife1DRule(rule, state_index=0):
  def rule_1d(neighbourhood):
    # http://mathworld.wolfram.com/ElementaryCellularAutomaton.html
    if len(neighbourhood) != 3:
      raise ValueError('neighbourhood must be 3 size')
    # Select rule index 0-7
    neighbourhood_state = neighbourhood[:, state_index] > 0.5
    rule_index = np.packbits(np.append([0,0,0,0,0],
      neighbourhood_state.astype(int)))[0]
    cell_state_change = np.zeros_like(neighbourhood[1])
    if rule & (1 << rule_index):
      cell_state_change[state_index] = 1. - neighbourhood[1][state_index]
    else:
      cell_state_change[state_index] = -neighbourhood[1][state_index]
    return cell_state_change
  return rule_1d

def diffuser_1d(neighbourhood, mutation=True):
  # Expects state values: [Energy, R, G, B]
  # Cell inherits mutated RGB and decayed energy from neighbourhood winner
  # Unless it is the winner in which case it mutates itself.
  # Inherited RGBs are mutated less than if the cell is itself the winner.
  # This is to enable waves of colour to radiate out from winning cells.
  if len(neighbourhood) != 3:
    raise ValueError('neighbourhood must be 3 size')
  if len(neighbourhood[1]) != 4:
    raise ValueError('cell state length expected to be 4')
  winner = 2
  if neighbourhood[2, 0] < neighbourhood[0, 0]:
    winner = 0
  if neighbourhood[1, 0] > neighbourhood[winner, 0]:
    winner = 1
  orig_state_and_colour = neighbourhood[1].copy()
  if winner == 1:
    # Mutate colour
    colour_mutation = 0.01 if mutation else 0.0
    state_mutation = 0.001 if mutation else 0.0
    preserved_state = neighbourhood[1, 0]
    state_and_colour = neighbourhood[1] + (colour_mutation *
        np.random.normal(size=neighbourhood[1].shape))
    state_and_colour[0] = (
        preserved_state + state_mutation * np.random.standard_cauchy())
  else:
    # Inherit colour and decayed state
    state_and_colour = neighbourhood[winner].copy()
    colour_mutation = 0.00001 if mutation else 0.0
    state_mutation = 0.00002 if mutation else 0.0
    preserved_state = neighbourhood[winner, 0]
    state_and_colour = neighbourhood[winner] + (colour_mutation *
        np.random.normal(size=neighbourhood[winner].shape))
    state_and_colour[0] = (
        0.999 * preserved_state + state_mutation * np.random.standard_cauchy())
  return state_and_colour - orig_state_and_colour

def diffuser_2d(neighbourhood, mutation=True):
  # Expects state values: [Energy, R, G, B]
  if neighbourhood.shape != (3,3,4):
    raise ValueError('neighbourhood must be 9 size')
  states = neighbourhood[:, :, 0]
  winner = np.array(np.where(states == states.max()))[:, 0]
  if (winner == (1, 1)).all():
    # Mutate colour
    colour_mutation = 0.01 if mutation else 0.0
    state_mutation = 0.002 if mutation else 0.0
    preserved_state = neighbourhood[winner[0], winner[1], 0]
    state_and_colour = neighbourhood[winner[0], winner[1]] + (colour_mutation *
        np.random.normal(size=neighbourhood[winner[0], winner[1]].shape))
    state_and_colour[0] = (
        preserved_state + state_mutation * np.random.standard_cauchy())
  else:
    # Inherit colour and decayed state
    state_and_colour = neighbourhood[winner[0], winner[1]].copy()
    colour_mutation = 0.01 if mutation else 0.0
    state_mutation = 0.00002 if mutation else 0.0
    preserved_state = neighbourhood[winner[0], winner[1], 0]
    state_and_colour = neighbourhood[winner[0], winner[1]] + (colour_mutation *
        np.random.normal(size=neighbourhood[winner[0], winner[1]].shape))
    state_and_colour[0] = (
        0.99 * preserved_state + state_mutation * np.random.standard_cauchy())
  return state_and_colour.clip(0, 1)

def minimal_diffuser_1d(neighbourhood, mutation=True):
  # Expects state values: [Energy, R, G, B]
  # Cell inherits mutated RGB and decayed energy from neighbourhood winner
  if len(neighbourhood) != 3:
    raise ValueError('neighbourhood must be 3 size')
  if len(neighbourhood[1]) != 4:
    raise ValueError('cell state length expected to be 4')
  winner = 2
  if neighbourhood[2, 0] < neighbourhood[0, 0]:
    winner = 0
  orig_state_and_colour = neighbourhood[1].copy()
  # Inherit colour and decayed state
  state_and_colour = neighbourhood[winner].copy()
  colour_mutation = 0.00001 if mutation else 0.0
  state_mutation = 0.00002 if mutation else 0.0
  preserved_state = neighbourhood[winner, 0]
  state_and_colour = neighbourhood[winner] + (colour_mutation *
      np.random.normal(size=neighbourhood[winner].shape))
  state_and_colour[0] = (
      0.999 * preserved_state + state_mutation * np.random.standard_cauchy())
  return state_and_colour - orig_state_and_colour




class TestWorld(unittest.TestCase):
  def test_1D_rule_30(self):
    rule30 = CreateBinaryLife1DRule(30, state_index=0)
    self.assertTrue(rule30(np.array([0,0,0]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([0,0,1]).reshape(3,1)) == 1)
    self.assertTrue(rule30(np.array([0,1,0]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([0,1,1]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([1,0,0]).reshape(3,1)) == 1)
    self.assertTrue(rule30(np.array([1,0,1]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([1,1,0]).reshape(3,1)) == -1)
    self.assertTrue(rule30(np.array([1,1,1]).reshape(3,1)) == -1)

  def test_1D_rule_0(self):
    rule30 = CreateBinaryLife1DRule(0, state_index=0)
    self.assertTrue(rule30(np.array([0,0,0]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([0,0,1]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([0,1,0]).reshape(3,1)) == -1)
    self.assertTrue(rule30(np.array([0,1,1]).reshape(3,1)) == -1)
    self.assertTrue(rule30(np.array([1,0,0]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([1,0,1]).reshape(3,1)) == 0)
    self.assertTrue(rule30(np.array([1,1,0]).reshape(3,1)) == -1)
    self.assertTrue(rule30(np.array([1,1,1]).reshape(3,1)) == -1)

  def test_1D_2_states(self):
    rule30 = CreateBinaryLife1DRule(30, state_index=1)
    io = [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[0, 0], [0, 0], [0, 1]], [0, 1]),
        ([[0, 0], [0, 1], [0, 0]], [0, 0]),
        ([[0, 0], [0, 1], [0, 1]], [0, 0]),
        ([[0, 1], [0, 0], [0, 0]], [0, 1]),
        ([[0, 1], [0, 0], [0, 1]], [0, 0]),
        ([[0, 1], [0, 1], [0, 0]], [0, -1]),
        ([[0, 1], [0, 1], [0, 1]], [0, -1]),

        ([[1, 0], [1, 0], [1, 0]], [0, 0]),
        ([[1, 0], [1, 0], [1, 1]], [0, 1]),
        ([[1, 0], [1, 1], [1, 0]], [0, 0]),
        ([[1, 0], [1, 1], [1, 1]], [0, 0]),
        ([[1, 1], [1, 0], [1, 0]], [0, 1]),
        ([[1, 1], [1, 0], [1, 1]], [0, 0]),
        ([[1, 1], [1, 1], [1, 0]], [0, -1]),
        ([[1, 1], [1, 1], [1, 1]], [0, -1]),

        ]
    for cells, target in io:
      new_state = rule30(np.array(cells).reshape(3,2))
      self.assertTrue((new_state == target).all())


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

  def test_diffuser_1d(self):
    input = np.array([
      [0.0,0.1,0.2,0.3],
      [0.1,0.4,0.5,0.6],
      [0.0,0.7,0.8,0.9]])
    # Winner
    self.assertTrue(np.allclose(diffuser_1d(input,
      mutation=False)[0], 0.1, 1.0))
    self.assertTrue((diffuser_1d(input)[1:] != input[1,1:]).all())
    self.assertTrue(np.allclose(diffuser_1d(input, mutation=False),
      np.array([0.1,0.4,0.5,0.6])))

    input = np.array([
      [0.2,0.1,0.2,0.3],
      [0.1,0.4,0.5,0.6],
      [0.0,0.7,0.8,0.9]])
    self.assertTrue(np.allclose(diffuser_1d(input,
      mutation=False)[0], 0.2*0.99, 0.1))
    self.assertTrue((diffuser_1d(input,
      mutation=False)[1:] == input[0,1:]).all())

  def test_diffuser_2d(self):
    input = np.array([
      [[0.0, 0.1, 0.1, 0.2],
      [0.0, 0.2, 0.2, 0.3],
      [0.0, 0.3, 0.3, 0.4]],

      [[0.0, 0.4, 0.4, 0.5],
      [0.1, 0.5, 0.5, 0.6],
      [0.0, 0.6, 0.6, 0.7]],

      [[0.0, 0.7, 0.7, 0.8],
      [0.0, 0.8, 0.8, 0.9],
      [0.05, 0.9, 0.9, 0.0]]]
      )
    self.assertTrue((diffuser_2d(input, mutation=False)[0] == 0.1).all())
    self.assertTrue((diffuser_2d(input)[1:] != input[1, 1, 1:]).all())
    self.assertTrue((diffuser_2d(input, mutation=False) == np.array([0.1, 0.5,
      0.5, 0.6])).all())

    input = np.array([
      [[0.0, 0.1, 0.1, 0.2],
      [0.2, 0.2, 0.2, 0.3],
      [0.0, 0.3, 0.3, 0.4]],

      [[0.0, 0.4, 0.4, 0.5],
      [0.1, 0.5, 0.5, 0.6],
      [0.0, 0.6, 0.6, 0.7]],

      [[0.0, 0.7, 0.7, 0.8],
      [0.0, 0.8, 0.8, 0.9],
      [0.15, 0.9, 0.9, 0.0]]]
      )
    self.assertTrue(diffuser_2d(input, mutation=False)[0] == 0.2*0.99)
    self.assertTrue((diffuser_2d(input,
      mutation=False)[1:] == input[0, 1, 1:]).all())

if __name__ == '__main__':
  unittest.main()
