#!/bin/bash

python ../../bin/ligandff.py -l inputs/ligands.sdf -o ligands -w

python ../../bin/plan.py -l inputs/ligands.sdf -o network -w -m lomap -n minimal
