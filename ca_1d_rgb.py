import argparse
import numpy as np
import pygame
from renderer import Renderer1D
import rules
from pygame import surfarray
import world

boundary_condition = [x.name for x in world.Boundary]

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-n', '--num_cells', help='Cells wide', default=127)
  ap.add_argument(
      '-l', '--history_length', help='Length of history', default=96)
  ap.add_argument(
      '-r', '--red_rule_set', help='Rule byte code, e.g. 161', default=None)
  ap.add_argument(
      '-g', '--green_rule_set', help='Rule byte code, e.g. 182', default=None)
  ap.add_argument(
      '-b', '--blue_rule_set', help='Rule byte code, e.g. 90, 165', default=None)
  ap.add_argument('-f', '--save_frames', action='store_true',
      help='Save frames to outout_v directory')
  ap.add_argument('-c', '--boundary', help='Boundary condition [{}]'.format(
    boundary_condition), default=boundary_condition[0])
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
  save_frames = args.get('save_frames', False)
  boundary = args.get('boundary')
  display_size = (640, 480)

  if red_rule_set is None and green_rule_set is None and blue_rule_set is None:
    raise ValueError('Set at least one rule')

  num_states = 3
  boundary = world.Boundary[boundary]
  world = world.World1D(num_cells, num_states=num_states, boundary=boundary)
  states = np.zeros(shape=(num_cells, num_states))
  # states = np.random.uniform(size=(num_cells, num_states))
  if red_rule_set is not None:
    world.set_rules(rules.CreateBinaryLife1DRule(red_rule_set, state_index=0))
    states[:, 0] = 0
  if green_rule_set is not None:
    world.set_rules(rules.CreateBinaryLife1DRule(green_rule_set, state_index=1))
    states[:, 1] = 0
  if blue_rule_set is not None:
    world.set_rules(rules.CreateBinaryLife1DRule(blue_rule_set, state_index=2))
    states[:, 2] = 0

  states[int(num_cells/2), :] = 1
  world.set_states(states)

  renderer = Renderer1D(num_cells=num_cells, history_size=history,
      display_size=display_size, state_depth=num_states)
  renderer.next_gen(world.cells)
  renderer.display()
  if save_frames:
    renderer.save_frame()

  while True:
    world.step()
    renderer.next_gen(world.cells)
    renderer.display()
    if save_frames:
      renderer.save_frame()

    stop = False
    for event in pygame.event.get():
      if event.type == pygame.QUIT:
        stop = True

    if stop:
      break


