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
  ap.add_argument('-g', '--history', help='Size of history', default=96)
  ap.add_argument('-r', '--rule_set', help='Rule byte code', default=158)
  ap.add_argument('-f', '--save_frames', action='store_true',
      help='Save frames to outout_v directory')
  ap.add_argument('-b', '--boundary', help='Boundary condition [{}]'.format(
    boundary_condition), default=boundary_condition[0])
  args = vars(ap.parse_args())
  num_cells = int(args.get('num_cells'))
  history = int(args.get('history'))
  rule_set = int(args.get('rule_set'))
  save_frames = args.get('save_frames', False)
  boundary = args.get('boundary')
  display_size = (640, 480)

  boundary = world.Boundary[boundary]
  world = world.World1D(num_cells, boundary=boundary)
  centre = int(num_cells/2)
  world.set_cell_state(centre, 1)
  world.set_rules(rules.CreateBinaryLife1DRule(rule_set))

  renderer = Renderer1D(num_cells=num_cells, history_size=history,
      display_size=display_size)
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


