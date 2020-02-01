# Multi-state Cellular Automata - [A-Life On Pi](http://www.alifeonpi.com) project

![cover_gif](https://github.com/dylski/loki/blob/master/rgb_energydown.gif)
A cellular automota simulation library that I intend to coax into producing
pretty patterns.

The idea is that each cell will be able to contain multiple states. For now this
has been implemented in the 1D case. Multiple rules can be run per cell.
In theory rules will be able to read and write to
multiple states although the current 1D rules read and write to a single state.

I'm currently focused on the 1D simulation although I have started a little
on the 2D and 3D versions.

## Description
Go to [CA 1D](http://www.alifeonpi.com/ca_1d.html) for more details.
Go to [Wolfram](http://mathworld.wolfram.com/ElementaryCellularAutomaton.html) for
infromation on 1D CAs and how to specify their rules.

## Example
Runs nicely on a Raspberry Pi, using Python3 and PyGame.
I think you just need to run install numpy and pygame, i.e. run the following:

    pip3 install numpy
    pip3 install pygame

## Quick Start
###Binary 1D CA

    $ python ca_1d.py -r 30 -b wrap

###Three coloured binary 1d CAs
Three state 1D CA assigned to RGB colours. Separate rule assigned to each state.

    $ python ca_1d_rgb -r 161 -g 182 -b 126 -c dead
    $ python loki1Dv.py

###Diffuse CA
Cells inherit neighbour qualities if the neighbour has a higher **energy**
state. I've only just started looking at this and it's not set up to be
particularly configurable yet. There's a lot to be played around with (and it's
currently a little broken).

    $ python3 ca_1d_diffuser.py

Run with -h arg for more options.

