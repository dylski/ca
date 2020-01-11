import argparse
import numpy as np
import pygame
from renderer import Renderer1D
import rules
from pygame import surfarray
from world import World1D

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-n', '--num_cells', help='Cells wide', default=127)
  ap.add_argument('-g', '--history', help='Size of history', default=96)
  ap.add_argument('-r', '--rule_set', help='Rule byte code', default=158)
  args = vars(ap.parse_args())
  num_cells = int(args.get('num_cells'))
  history = int(args.get('history'))
  rule_set = int(args.get('rule_set'))
  display_size = (640, 480)

  world = World1D(num_cells)
  world.set_cell_state(int(num_cells/2), 1)
  world.set_rules(rules.CreateBinaryLife1DRule(rule_set))

  renderer = Renderer1D(num_cells=num_cells, history_size=history,
      display_size=display_size)
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

