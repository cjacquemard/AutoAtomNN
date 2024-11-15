import sys
import os

import numpy as np
from rdkit.Chem import AllChem, rdFMCS, Draw, rdMolTransforms

import warnings
warnings.filterwarnings('error')


_ANGLE_STEP = np.pi / 180
_SCAN_ANGLES = np.arange(-np.pi, np.pi+_ANGLE_STEP, _ANGLE_STEP)
_SCAN_INDICES = np.arange(_SCAN_ANGLES.shape[0]).astype(int)

def MMFF94_torsion(v1, v2, v3, a):
	return 0.5 * (v1 * (1 + np.cos(a)) + v2 * (1 - np.cos(2*a)) + v3 * (1 + np.cos(3*a)))


def MMFF94_vdw(Rij, e, d):
	return e * ((1.07 * Rij) / (d + 0.07 * Rij))**7 * ((1.12 * Rij**7) / (d**7 + 0.12 * Rij**7) - 2)


def MMFF94_elec(q1, q2, d, e=1):
	return 332.0716 * q1 * q2 / (e * (d + 0.05))


def MMFF4_1_4_distance(dij, djk, dkl, aijk, ajkl, tijkl):
	x = -dij * np.cos(aijk) + djk - dkl * np.cos(ajkl)
	y = dij * np.sin(aijk) - dkl * np.sin(ajkl) * np.cos(tijkl)
	z = dkl * np.sin(ajkl) * np.sin(tijkl)
	return np.sqrt(x**2 + y**2 + z**2)


def boltzmann(dE, t):
	# NOTE: Unit in kcal/mol and Kelvin
	return np.exp(-dE/(1.987E-3 * t))


def two_sides_boltzmann_probabilities(E, i, t):
	left_dE = (E[:i] - E[1:i+1])[::-1]
	right_dE = E[i+1:] - E[i:-1]

	# Cap minium dE otherwise number is too big (overflow)
	left_dE[left_dE < -423.1] = -423.1
	right_dE[right_dE < -423.1] = -423.1

	left_probs = boltzmann(left_dE, t)
	right_probs = boltzmann(right_dE, t)

	return left_probs, right_probs


def get_dihedral_atoms_from_bond(rdbond):
	if not rdbond.HasOwningMol():
		raise ValueError("rdkit Bond must belongs to a rdkit Mol")

	rdmol = rdbond.GetOwningMol()
	atom1 = rdbond.GetBeginAtom()
	atom2 = rdbond.GetEndAtom()

	# Get atom1 neighbors (except atom2) to define the first index of the torsion
	atom1_neighbors = []
	for neighbor_atom in atom1.GetNeighbors():
		if neighbor_atom.GetIdx() != atom2.GetIdx():
			atom1_neighbors.append(neighbor_atom)

	# Get atom2 neighbors (except atom1) to define the last index of the torsion
	atom2_neighbors = []
	for neighbor_atom in atom2.GetNeighbors():
		if neighbor_atom.GetIdx() != atom1.GetIdx():
			atom2_neighbors.append(neighbor_atom)

	# Check if atom1 and atom2 are bonded to other atoms
	# Skip if not (e.g. bond with termini atom like H)
	if len(atom1_neighbors) and len(atom2_neighbors):
		return atom1_neighbors[0], atom1, atom2, atom2_neighbors[0]
	else:
		return None
		

def mol_rigidity(rdconf):
	if not rdconf.HasOwningMol():
		raise ValueError("rdkit conformer must belongs to a rdkit Mol")

	# Init output
	bond_indices = []
	bond_strain_scores = []
	bond_rigidity_scores = []

	# Copy the molecule because coordinates may change
	rdmol = AllChem.Mol(rdconf.GetOwningMol())
	AllChem.SanitizeMol(rdmol) # Just in case!

	# Check if number of bonds changed
	if rdmol.GetNumBonds() != rdconf.GetOwningMol().GetNumBonds():
		print("ERROR: Number of bonds changed after sanitization")

	# Copy initial positions
	conf_atom_positions = rdconf.GetPositions() 

	mp = AllChem.MMFFGetMoleculeProperties(rdmol, mmffVariant='MMFF94s')
	ff = AllChem.MMFFGetMoleculeForceField(rdmol, mp)

	bond_indices = []
	bond_strain_scores = []
	bond_rigidity_scores = []

	for rdbond in rdmol.GetBonds():
		bond_index = rdbond.GetIdx()
		if bond_index >= rdconf.GetOwningMol().GetNumBonds():
			continue # Due to sanitization

		if rdbond.IsInRing():
			strain_score, rigidity_score = 1.0, 1.0
		else:
			dihedral_atoms = get_dihedral_atoms_from_bond(rdbond)

			# Check if bond is a dihedral
			if dihedral_atoms is not None:
				a1, a2, a3, a4 = dihedral_atoms
				strain_score, rigidity_score = dihedral_scan(rdmol.GetConformer(0), ff, a1.GetIdx(), a2.GetIdx(), a3.GetIdx(), a4.GetIdx())
			else:
				# Bond with termini atom
				continue

		bond_indices.append(bond_index)
		bond_strain_scores.append(strain_score)
		bond_rigidity_scores.append(rigidity_score)

	return np.array(bond_indices), np.array(bond_strain_scores), np.array(bond_rigidity_scores)


def dihedral_scan(rdconf, ff, index_1, index_2, index_3, index_4):
	# Get the current dihe
	ref_angle = rdMolTransforms.GetDihedralRad(rdconf, index_1, index_2, index_3, index_4)

	# Center angles on to the torsion angle value of the current bond
	tested_angles = _SCAN_ANGLES + ref_angle

	energies = []
	for angle in tested_angles:
		rdMolTransforms.SetDihedralRad(rdconf, index_1, index_2, index_3, index_4, angle)
		ff.Initialize() # This must be call! Otherwise results are wrong (don't know why!!!!)
		energy = ff.CalcEnergy()
		energies.append(energy)

	energies = np.array(energies)

	# Integrate left and right
	middle_index = int((len(energies) - 1) / 2)

	left_probs, right_probs = two_sides_boltzmann_probabilities(energies, middle_index, 300)

	# Find a range (99% of left and right probs)
	# Set all values above 1 to 1 (capped) to avoid cumulative effect when the starting point is not at a local minima
	capped_left_probs = np.copy(left_probs)
	capped_left_probs[capped_left_probs > 1] = 1

	capped_right_probs = np.copy(right_probs)
	capped_right_probs[capped_right_probs > 1] = 1

	left_cumprobs = np.cumprod(capped_left_probs)
	right_cumprobs = np.cumprod(capped_right_probs)

	left_prob_mask = left_cumprobs <= 0.01
	right_prob_mask = right_cumprobs <= 0.01

	# Check if there are barriers both sides
	if left_prob_mask.any():
		left_index = middle_index - left_prob_mask.argmax()
	else:
		left_index = 0

	if right_prob_mask.any():
		right_index = middle_index + right_prob_mask.argmax()
	else:
		right_index = energies.shape[0] - 1

	# Compute the strain torsion score
	# First find the global minimum within previously defined range
	minimum_sub_index = np.argmin(energies[left_index:right_index+1])
	minimum_index = _SCAN_INDICES[left_index:right_index+1][minimum_sub_index]

	# Then, compute the strain score
	de = energies[middle_index] - energies[minimum_index]

	# This score is the "probability" of a dihedral to move from its initial angle
	# 0 it will move without external forces to maintain the conformation
	# 1 meens it will not move
	strain_score =  boltzmann(de, 300)

	# Compute the flexibility score
	left_angle = _SCAN_ANGLES[left_index]
	right_angle = _SCAN_ANGLES[right_index]

	rigidity_score = 1 - (right_angle - left_angle) / (np.pi*2)

	return strain_score, rigidity_score