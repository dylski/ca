import argparse
import numpy as np
import pygame
from renderer import Renderer1D
import rules
from pygame import surfarray
from world import EdgeEffect
from world import World1D

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-n', '--num_cells', help='Cells wide', default=128)
  ap.add_argument(
      '-l', '--history_length', help='Length of history', default=96)
  ap.add_argument(
      '-r', '--red_rule_set', help='Rule byte code, e.g. 158', default=None)
  ap.add_argument(
      '-g', '--green_rule_set', help='Rule byte code, e.g. 187', default=None)
  ap.add_argument(
      '-b', '--blue_rule_set', help='Rule byte code, e.g. 90', default=None)
  args = vars(ap.parse_args())
  num_cells = int(args.get('num_cells'))
  history = int(args.get('history_length'))
  red_rule_set = None
  green_rule_set = None
  blue_rule_set = None
  if args.get('red_rule_set'):
    red_rule_set = int(args.get('red_rule_set'))
  if args.get('green_rule_set'):
    green_rule_set = int(args.get('green_rule_set'))
  if args.get('blue_rule_set'):
    blue_rule_set = int(args.get('blue_rule_set'))
  display_size = (640, 480)

  if not red_rule_set and not green_rule_set and not blue_rule_set:
    raise ValueError('Set at least one rule')

  num_states = 3
  mutation = False  # 0.025
  world = World1D(num_cells, num_states=num_states,
      edge_effect=EdgeEffect.reflect)
  states = np.random.uniform(size=(num_cells, num_states))
  if red_rule_set:
    world.set_rules(rules.CreateBinaryLife1DRule(
      red_rule_set, state_index=0, other_state_mutation=mutation))
    states[:, 0] = 0
  if green_rule_set:
    world.set_rules(rules.CreateBinaryLife1DRule(
      green_rule_set, state_index=1, other_state_mutation=mutation))
    states[:, 1] = 0
  if blue_rule_set:
    world.set_rules(rules.CreateBinaryLife1DRule(
      blue_rule_set, state_index=2, other_state_mutation=mutation))
    states[:, 2] = 0
  states[int(num_cells/2), :] = 1
  world.set_states(states)

  renderer = Renderer1D(num_cells=num_cells, history_size=history,
      display_size=display_size, state_depth=num_states)
  renderer.next_gen(world.cells)
  renderer.display()

  while True:
    world.step()
    renderer.next_gen(world.cells)
    renderer.display()

    stop = False
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        stop = True

    if stop:
      break


