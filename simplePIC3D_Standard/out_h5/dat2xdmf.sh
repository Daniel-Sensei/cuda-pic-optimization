#!/bin/sh
python3 read_fields.py
./h5Xdmf h5Xdmf.inp
paraview fields.xmf &
