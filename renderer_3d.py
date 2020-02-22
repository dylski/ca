''' CA3D base class, examples and Rendering class.

Create CA3DRenderer with a CA3DBase-derived class; it calls and render steps.

Examples:
Render a random 'CA'
  $ python ca3d_renderer.py -r

Render ca states saved in a numpy file (t,x,y,z)
  $ python ca3d_renderer.py -n appraisals/glider/Muncher_ca_state.npy

Make video
  $ python ca3d_renderer.py -r -f
  $ ffmpeg -r 60 -f image2 -s 640x480 -i ca3d_renderer_frames/frame_%04d.png \\
      -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
'''

import argparse
import world

if __name__ == '__main__':
  ap = argparse.ArgumentParser()
  ap.add_argument('-f', '--render_to_file',
      help='render to file', action='store_true')
  ap.add_argument('-x', '--width', help='Cells wide', default=20)
  ap.add_argument('-y', '--height', help='Cells height', default=20)
  ap.add_argument('-z', '--depth', help='Cells depth', default=20)
  ap.add_argument('-b', '--boundary', help='Boundary condition [{}]'.format(
    world.boundary_type), default=world.boundary_type[0])
  args = vars(ap.parse_args())

  width = int(args.get('width'))
  height = int(args.get('height'))
  depth = int(args.get('depth'))
  boundary = args.get('boundary')
  np_file = args.get('np_file', None)
  render_to_file = args.get('render_to_file', False)

if render_to_file:
  from panda3d.core import loadPrcFileData
  # CPU rendering 50% speed on MacBook Pro

  # loadPrcFileData("",
  # """
  #    load-display p3tinydisplay # to force CPU only rendering (to make it available as an option if everything else fail, use aux-display p3tinydisplay)
  #    window-type offscreen # Spawn an offscreen buffer (use window-type none if you don't need any rendering)
  #    audio-library-name null # Prevent ALSA errors
  #    show-frame-rate-meter 0
  #    sync-video 0
  # """)
  loadPrcFileData("",
  """
     window-type offscreen # Spawn an offscreen buffer (use window-type none if you don't need any rendering)
     audio-library-name null # Prevent ALSA errors
     show-frame-rate-meter 0
     sync-video 0
  """)
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import *
import numpy as np
from pathlib import Path
import rules
import shutil
import sys


class Renderer3D(ShowBase):
  def __init__(self, state_generator, render_to_file=False):
    # CA3D class that generates 3D states.
    ShowBase.__init__(self)

    # if you leave mouse mode enabled camera position will be governed by Panda
    # mouse control
    # self.disableMouse()
    # Move camera for a better view
    # self.camera.setY(-20)
    if render_to_file:
      path = 'renderer_3d_frames'
      duration = 10
      shutil.rmtree('{}'.format(path))
      Path(path).mkdir(parents=True, exist_ok=True)
      self.movie(namePrefix='{}/frame'.format(path),
          duration=duration, fps=24, format='png')
      print('Saving {} seconds of frames'.format(duration))

    # Enable fast exit
    self.accept("escape", sys.exit)

    self._state_generator = state_generator
    self._structure_shape = self._state_generator.dim
    self._create_structure()
    self._structure_root.reparentTo(self.render)
    self._structure_root.setPos(0, 20, 0)
    self._structure_root.setHpr(40, -30, 20)

    # Add a simple point light
    plight = PointLight('plight')
    plight.setColor(VBase4(0.5, 0.5, 0.5, 1))
    #plight.setAttenuation(Point3(0, 0, 0.5))
    plnp = self.render.attachNewNode(plight)
    plnp.setPos(4, -4, 4)
    self.render.setLight(plnp)
    # Add an ambient light
    alight = AmbientLight('alight')
    alight.setColor(VBase4(0.9, 0.9, 0.9, 1))
    alnp = self.render.attachNewNode(alight)
    self.render.setLight(alnp)

    self._next_time = 0
    self._frame_index = 0
    self.taskMgr.add(self.set_ca_states,
        self._state_generator.__class__.__name__)

  def set_ca_states(self, task):
    if task.time > self._next_time:
      states = self._state_generator.step()
      for x in range(self._structure_shape[0]):
        for y in range(self._structure_shape[1]):
          for z in range(self._structure_shape[2]):
            state = states[x, y, z]
            alpha = max(0.1, min(1.0, state))
            new_color = (state, state, state, alpha)
            self._structure_cubes[x][y][z].setColor(new_color)
      self._next_time = task.time + 0.017
    return Task.cont

  def _create_structure(self, cube_scale=0.1):
    self._structure_cubes = []
    self._structure_root = NodePath('structure')
    width = self._structure_shape[0] * 3 * cube_scale
    height = self._structure_shape[1] * 3 * cube_scale
    depth = self._structure_shape[2] * 3 * cube_scale
    x_pos = np.linspace(0, width, self._structure_shape[0])
    y_pos = np.linspace(0, height, self._structure_shape[1])
    z_pos = np.linspace(0, depth, self._structure_shape[2])
    for x in x_pos:
      self._structure_cubes.append([])
      x_list = self._structure_cubes[-1]
      for y in y_pos:
        x_list.append([])
        y_list = x_list[-1]
        for z in z_pos:
          cube_node_path = self._structure_root.attachNewNode(create_cube())
          cube_node_path.setScale(0.1)
          cube_node_path.setPos(x - width/2, (z - depth/2), -(y - height/2))
          # cube_node_path.setPos(x - width/2, y - height/2, z - depth/2)
          cube_node_path.setTransparency(TransparencyAttrib.MAlpha)
          cube_node_path.setAlphaScale(0.5)
          y_list.append(cube_node_path)
    return self._structure_root


def create_cube():
  format = GeomVertexFormat.getV3n3c4()
  vertexData = GeomVertexData('cube', format, Geom.UHStatic)

  vertexData.setNumRows(24)

  vertices = GeomVertexWriter(vertexData, 'vertex')
  normals = GeomVertexWriter(vertexData, 'normal')
  colors = GeomVertexWriter(vertexData, 'color')

  vertices.addData3f(-1, -1, -1)
  vertices.addData3f(-1, -1, -1)
  vertices.addData3f(-1, -1, -1)
  vertices.addData3f(1, -1, -1)
  vertices.addData3f(1, -1, -1)
  vertices.addData3f(1, -1, -1)
  vertices.addData3f(1, 1, -1)
  vertices.addData3f(1, 1, -1)
  vertices.addData3f(1, 1, -1)
  vertices.addData3f(-1, 1, -1)
  vertices.addData3f(-1, 1, -1)
  vertices.addData3f(-1, 1, -1)
  vertices.addData3f(-1, -1, 1)
  vertices.addData3f(-1, -1, 1)
  vertices.addData3f(-1, -1, 1)
  vertices.addData3f(1, -1, 1)
  vertices.addData3f(1, -1, 1)
  vertices.addData3f(1, -1, 1)
  vertices.addData3f(1, 1, 1)
  vertices.addData3f(1, 1, 1)
  vertices.addData3f(1, 1, 1)
  vertices.addData3f(-1, 1, 1)
  vertices.addData3f(-1, 1, 1)
  vertices.addData3f(-1, 1, 1)

  normals.addData3f(0, -1, 0)
  normals.addData3f(-1, 0, 0)
  normals.addData3f(0, 0, -1)
  normals.addData3f(0, -1, 0)
  normals.addData3f(1, 0, 0)
  normals.addData3f(0, 0, -1)
  normals.addData3f(1, 0, 0)
  normals.addData3f(0, 1, 0)
  normals.addData3f(0, 0, -1)
  normals.addData3f(0, 1, 0)
  normals.addData3f(-1, 0, 0)
  normals.addData3f(0, 0, -1)
  normals.addData3f(0, -1, 0)
  normals.addData3f(-1, 0, 0)
  normals.addData3f(0, 0, 1)
  normals.addData3f(0, -1, 0)
  normals.addData3f(1, 0, 0)
  normals.addData3f(0, 0, 1)
  normals.addData3f(1, 0, 0)
  normals.addData3f(0, 1, 0)
  normals.addData3f(0, 0, 1)
  normals.addData3f(0, 1, 0)
  normals.addData3f(-1, 0, 0)
  normals.addData3f(0, 0, 1)

  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)
  colors.addData4f(1, 1, 1, 1)

  # Store the triangles, counter clockwise from front
  primitive = GeomTriangles(Geom.UHStatic)
  primitive.addVertices(0, 3, 15)
  primitive.addVertices(0, 15, 12)
  primitive.addVertices(4, 6, 18)
  primitive.addVertices(4, 18, 16)
  primitive.addVertices(7, 9, 21)
  primitive.addVertices(7, 21, 19)
  primitive.addVertices(10, 1, 13)
  primitive.addVertices(10, 13, 22)
  primitive.addVertices(2, 11, 8)
  primitive.addVertices(2, 8, 5)
  primitive.addVertices(14, 17, 20)
  primitive.addVertices(14, 20, 23)

  geom = Geom(vertexData)
  geom.addPrimitive(primitive)

  node = GeomNode('cube gnode')
  node.addGeom(geom)
  return node



if __name__ == '__main__':
  boundary = world.Boundary[boundary]
  world = world.World3D((width, height, depth), boundary=boundary)
  # centre = int(num_cells/2)dd
  # world.set_cell_state(centre, 1)
  states = np.random.uniform(size=(width, height, depth))
  states[states < 0.8] = 0.
  states[states >= 0.8] = 1.
  world.set_states(states)
  world.set_rules(rules.GOL3D)
  app = Renderer3D(world, render_to_file=render_to_file)
  app.run()
  # elif np_file:
  #   app = CA3DRenderer(ca3d_base.CA3DFromFile(filename=np_file),
  #       render_to_file=render_to_file)
  #   app.run()



