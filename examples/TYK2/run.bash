#!/bin/bash

# Build alchemical ligand network
python ../../bin/plan.py -l inputs/ligands.sdf -o network -w -m lomap -n minimal

# Optimize placement of ligand 2
python ../../bin/optimize.py -p inputs/receptor.pdb -l1 inputs/ligands.sdf -l2 inputs/ligands.sdf -o optimize -w

# Generate force field parameters for ligands
python ../../bin/ligff.py -l optimize/ligands_1.sdf -o ligands -w

# Make complexes for MD
mkdir complexes 2> /dev/null
python ../../bin/sysprep.py -opt optimize -l ligands -n network -o complexes -w