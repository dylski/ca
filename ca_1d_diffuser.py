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

  num_states = 4
  world = World1D(num_cells, num_states=num_states)
  states = np.random.uniform(size=(num_cells, num_states))
  states[:, :] = 0
  # highs = list(set(np.random.randint(num_cells, size=(int(num_cells/80)))))
  # states[highs, 0] = 1
  states[int(num_cells/2), :] = 1
  world.set_states(states)
  world.set_rules(rules.diffuser_1d)

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


