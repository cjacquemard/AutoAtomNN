# WARNING! Run this script in the directory where files are!
delete all

load receptor_aln.pdb, receptor

load ligands_1.sdf, ligand_1
split_states ligand_1, prefix=l1
delete ligand_1
group ligand_1, l1*

load ligands_2.sdf, ligand_2
split_states ligand_2, prefix=l2
delete ligand_2
group ligand_2, l2*

load populations.pdb, population, discrete=1

hide everything, all
show cartoon, receptor
color gray80, receptor and e. c

show sticks, ligand_1 and not (h. and (e. c extend 1))
color green, ligand_1 and e. c

show sticks, ligand_2  and not (h. and (e. c extend 1))
color magenta, ligand_2 and e. c

show spheres, population
spectrum q, red_white_blue, population, 0, 100
ramp_new colorbar, none, [0, 50, 100], [red, white, blue]

run original_system_box.pml
run new_system_box.pml
run max_system_box.pml

center
