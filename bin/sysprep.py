import sys
import os
import itertools
from io import BytesIO

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rdkit.Chem import AllChem, rdFMCS, Draw, rdMolTransforms
from scipy.spatial import KDTree
from scipy.signal import argrelmin, argrelmax
from PIL import Image

import plan
import optimize
import forcefield
from common import RdkitHelper
from filehandler import PDB

# https://gist.github.com/fangkuoyu/dc785218e5d4d94c752e80f1aaba4fad
def rdkit_to_nx(mol):
	G = nx.Graph()

	for atom in mol.GetAtoms():
		G.add_node(atom.GetIdx(),
				   atomic_num=atom.GetAtomicNum(),
				   is_aromatic=atom.GetIsAromatic(),
				   atom_symbol=atom.GetSymbol(),
				   **atom.GetPropsAsDict())
		
	for bond in mol.GetBonds():
		G.add_edge(bond.GetBeginAtomIdx(),
				   bond.GetEndAtomIdx(),
				   bond_type=bond.GetBondType(),
				   **bond.GetPropsAsDict())
		
	return G


def molid(mol):
	for i, atom in enumerate(mol.GetAtoms()):
		atom.SetAtomMapNum(i)
	return mol


def plot_mol(rdmol):
	# def get_props(rdkit_objects, prop_name, ignore_missing=True):
	# 	# You can pass GetBonds() or GetAtoms() directly
	# 	rdkit_objects = [x for x in rdkit_objects]

	# 	indices = []
	# 	values = []
	# 	annotated_rdkit_objects = []
	# 	for ro in rdkit_objects:
	# 		d = ro.GetPropsAsDict()

	# 		if prop_name not in d:
	# 			if not ignore_missing:
	# 				raise ValueError("Prop name does not exist")
	# 			else:
	# 				continue

	# 		indices.append(ro.GetIdx())
	# 		values.append(d[prop_name])
	# 		annotated_rdkit_objects.append(ro)

	# 	return indices, values, annotated_rdkit_objects

	def annotate(rdkit_objects, prop_name):
		# You can pass GetBonds() or GetAtoms() directly
		rdkit_objects = [x for x in rdkit_objects]

		if not rdkit_objects:
			raise ValueError("rdkit_objects empty")

		anno_prop_name = None
		if isinstance(rdkit_objects[0], AllChem.rdchem.Atom):
			anno_prop_name = "atomNote"
		elif isinstance(rdkit_objects[0], AllChem.rdchem.Bond):
			anno_prop_name = "bondNote"
		else:
			raise ValueError("rdkit_objects must be Atom or Bond")

		indices, values, annotated_rdkit_objects = RdkitHelper.props_to_array(rdkit_objects, prop_name)

		for ro, value in zip(annotated_rdkit_objects, values):
			ro.SetProp(anno_prop_name, f"{value.item():.2f}")


	def highlight(rdkit_objects, prop_name, min_color=(1, 0, 0), max_color=(0, 0, 1)):
		indices, values, annotated_rdkit_objects = RdkitHelper.props_to_array(rdkit_objects, prop_name)

		# Must store a list with Python int otherwise rdkit complains
		indices = [x.item() for x in indices]

		norm_values = (values - values.min()) / ((values.max() - values.min()))

		min_color = np.array(min_color)
		max_color = np.array(max_color)
		if max_color.shape[0] != 3:
			raise ValueError("Color must be defined as RGB ")
		min_color = min_color - min_color.min() / (min_color.max() - min_color.min()) # RGB values are between 0 and 1
		max_color = max_color - max_color.min() / (max_color.max() - max_color.min()) # RGB values are between 0 and 1
		colors = min_color + (max_color - min_color) * norm_values[:,np.newaxis]

		d_colors = {}
		for index, color in zip(indices, colors):
			d_colors[index] = tuple(color)

		return indices, d_colors

	def rdmol_to_image(rdmol, **kwargs):
		d2d = Draw.MolDraw2DCairo(500,500)
		Draw.rdMolDraw2D.PrepareAndDrawMolecule(d2d, rdmol, **kwargs)

		# Keep a trace of arguments
		# Draw.rdMolDraw2D.PrepareAndDrawMolecule(d2d, rdmol, kekulize=False,
		# 	highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
		# 	highlightBonds=highlight_bonds,highlightBondColors=highlight_bond_colors
		# )

		d2d.FinishDrawing()
		sio = BytesIO(d2d.GetDrawingText())
		return Image.open(sio)

	def anno_light(draw_mol,
			anno_atom_prop_name=None,
			highlight_atom_prop_name=None,
			anno_bond_prop_name=None,
			highlight_bond_prop_name=None):
		draw_mol = AllChem.Mol(draw_mol) # copy

		if anno_atom_prop_name is not None:
			annotate(draw_mol.GetAtoms(), anno_atom_prop_name)

		if anno_bond_prop_name is not None:
			annotate(draw_mol.GetBonds(), anno_bond_prop_name)

		draw_dict = {"kekulize": False}
		if highlight_atom_prop_name is not None:
			atom_indices, atom_colors = highlight(draw_mol.GetAtoms(), highlight_atom_prop_name)
			draw_dict.update({"highlightAtoms": atom_indices, "highlightAtomColors": atom_colors})

		if highlight_bond_prop_name is not None:
			bond_indices, bond_colors = highlight(draw_mol.GetBonds(), highlight_bond_prop_name)
			draw_dict.update({"highlightBonds": bond_indices, "highlightBondColors": bond_colors})

		return rdmol_to_image(draw_mol, **draw_dict)


	draw_mol = AllChem.Mol(rdmol) # copy

	# Remove molAtomMapNumber, otherwise atom indices are drawn
	for atom in draw_mol.GetAtoms():
		atom.ClearProp("molAtomMapNumber")

	# Draw combo scores
	# bond_combo_img = anno_light(draw_mol, anno_bond_prop_name="combo_score", highlight_bond_prop_name="combo_score")
	# bond_rigidity_img = anno_light(draw_mol, anno_bond_prop_name="rigidity_score", highlight_bond_prop_name="rigidity_score")
	# bond_strain_img = anno_light(draw_mol, anno_bond_prop_name="strain_score", highlight_bond_prop_name="strain_score")

	# center_path_img = anno_light(draw_mol, anno_atom_prop_name="center_path_score", highlight_atom_prop_name="center_path_score")
	weighted_center_path_img = anno_light(draw_mol, anno_atom_prop_name="weighted_center_path_score", highlight_atom_prop_name="weighted_center_path_score")

	weighted_center_path_img.show()
	breakpoint()
	# bond_combo_mol_draw = AllChem.Mol(draw_mol)
	# annotate(bond_combo_mol_draw.GetBonds(), "combo_score")
	# bond_indices, bond_colors = highlight(bond_combo_mol_draw.GetBonds(), "combo_score")

	# kwargs = {"kekulize": False, "highlightBonds": bond_indices, "highlightBondColors": bond_colors}
	# bond_combo_img = rdmol_to_image(bond_combo_mol_draw, **kwargs)

	# # Draw rigidity scores
	# bond_rigidity_mol_draw = AllChem.Mol(draw_mol)
	# annotate(bond_combo_mol_draw.GetBonds(), "combo_score")
	# bond_indices, bond_colors = highlight(bond_combo_mol_draw.GetBonds(), "combo_score")

	# kwargs = {"kekulize": False, "highlightBonds": bond_indices, "highlightBondColors": bond_colors}
	# bond_combo_img = rdmol_to_image(bond_combo_mol_draw, **kwargs)

	# highlight_atoms = []
	# highlight_atom_colors = {}
	# for atom_id in atom_scores:
	# 	score = longest_path_scores[atom_id]
	# 	highlight_atoms.append(atom_id)

	# 	if atom_id == mcs_closest_centroid_index:
	# 		highlight_atom_colors[atom_id] = (0, 0, 1)
	# 	else:
	# 		highlight_atom_colors[atom_id] = (0, score / max_longest_score, 0)

	# 	draw_mol.GetAtomWithIdx(atom_id).SetProp('atomNote', f"{score:.2f}")


	# highlight_bonds = []
	# highlight_bond_colors = {}
	# for bond_id in bond_scores:
	# 	s, r = bond_scores[bond_id]
	# 	score = r * s
	# 	highlight_bonds.append(bond_id)
	# 	highlight_bond_colors[bond_id] = (score, 0, 0)
	# 	draw_mol.GetBondWithIdx(bond_id).SetProp('bondNote', f"{score:.2f}")

	# for atom in draw_mol.GetAtoms():
	# 	atom.ClearProp("molAtomMapNumber")

	# d2d = Draw.MolDraw2DCairo(500,500)
	# Draw.rdMolDraw2D.PrepareAndDrawMolecule(d2d, draw_mol, kekulize=False,
	# 	highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
	# 	highlightBonds=highlight_bonds,highlightBondColors=highlight_bond_colors
	# )

	# d2d.FinishDrawing()
	# sio = BytesIO(d2d.GetDrawingText())
	# Image.open(sio).show()


_DEFAULTS = {
	"binding_site_threshold": 6.5,
}

# class TransformationInfo:
# 	def __init__(self, id_1, id_2, ligname_1=None, ligname_2=None, sim=None):
# 		self.ids = id_1, id_2
# 		self.lignames = ligname_1, ligname_2
# 		self.sim = sim

# 		self.rcpt_cm_atom_indices = None # Ca atom indices that define the binding site
# 		self.pos_restrained_atom_indices = None # Atoms restrained during MD. Usually, the Ca atoms of the protein are restrained.

# 		# Each attribute is tuple corresponding to ligand 1 and ligand 2
# 		self.mols = None # RDKIT mols
# 		self.cm_atom_indices = None # Atom(s) that is(are) used as centroid for binding site restraint
# 		self.align_ligand_indices = None # Triplet of atom indices used to align ligand 2 to ligand 1 during transformation


class NetworkInfo:
	def __init__(self, filepath):
		self.filepath = filepath
		self.df = None

	def load(self):
		self.df = pd.read_csv(self.filepath, sep='\s+')

	def get_rows(self):
		return self.df.iterrows()

	def get_pair_ligids(self):
		return tuple([(row[plan._ID_A_COLNAME], row[plan._ID_B_COLNAME]) for row in self.get_rows()])

class MolobjectInfo:
	def __init__(self, struct_filepath):
		self.struct_filepath = struct_filepath

	def load(self):
		raise NotImplementedError

	def get_positions(self):
		raise NotImplementedError


class CommonLigandFrame:
	# NOTE: rdmols should be sanitized otherwise the forcefield parameters may be wrong
	# Most of computed values are stored as rdkit Property
	def __init__(self, rdmols):
		self.rdmols = rdmols

		self.mcs_result = None
		self.mcs_query = None
		self.mcs_mol = None

	def mcs(self):
		# Get Maximum Common Substructure
		self.mcs_result = rdFMCS.FindMCS(self.rdmols)

		# The ligands must share at least a substructure containing 3 or more atoms.
		# We need 3 atoms to restraint ligand 1 and ligand 2 during the simulation.
		if self.mcs_result.numAtoms < 3:
			# TODO: Write a method that treat pairs of ligand instead
			raise ValueError(f"MCS must contain 3 atoms or more")

		# Get the MCS molecule
		self.mcs = AllChem.MolFromSmarts(self.mcs_result.smartsString)
		#self.mcs_query = AllChem.Mol(tmp_mcs) # Copy
		self.mcs_query = AllChem.Mol(self.mcs) # Copy
		# AllChem.SanitizeMol(tmp_mcs) # Inplace
		# self.mcs = AllChem.rdmolops.AddHs(tmp_mcs)

		# Get MCS atom indices for mols
		self.rdmols_mcs_atom_indices = np.zeros((len(self.rdmols), self.mcs_query.GetNumAtoms())).astype(int)
		for i, mol in enumerate(self.rdmols):
			self.rdmols_mcs_atom_indices[i] = mol.GetSubstructMatch(self.mcs_query)

		# Get MCS bond indices for mols
		rdmols_mcs_bond_indices = []
		for i, mol in enumerate(self.rdmols):
			rdmol_mcs_atom_indices = self.rdmols_mcs_atom_indices[i]
			rdmol_mcs_bond_indices = []
			for bond in self.mcs_query.GetBonds():
				idx1, idx2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
				rdmol_mcs_bond_indices.append(mol.GetBondBetweenAtoms(rdmol_mcs_atom_indices[idx1].item(), rdmol_mcs_atom_indices[idx2].item()).GetIdx())
			rdmols_mcs_bond_indices.append(rdmol_mcs_bond_indices)
		self.rdmols_mcs_bond_indices = np.array(rdmols_mcs_bond_indices)

		# Compute the RMSD of all ligands MCS
		self.rdmols_mcs_atom_positions = np.zeros((len(self.rdmols), self.mcs_query.GetNumAtoms(), 3))
		for i, rdmol in enumerate(self.rdmols):
			# Molecule must contain only 1 conformer
			rdconf = rdmol.GetConformer(0)

			# Get the position for atoms in the MCS
			rdmol_mcs_atom_indices = self.rdmols_mcs_atom_indices[i]
			self.rdmols_mcs_atom_positions[i] = rdconf.GetPositions()[rdmol_mcs_atom_indices]

		# Compute mean positions
		self.mcs_mean_atom_positions = np.mean(self.rdmols_mcs_atom_positions, axis=0)

		# Compute standard deviation of atoms around mean positions (kinda like a RMSF)
		self.mcs_std_atom_positions = np.sqrt(np.mean(np.sum((self.rdmols_mcs_atom_positions - self.mcs_mean_atom_positions)**2, axis=2), axis=0))
		if np.any(self.mcs_std_atom_positions > 1.0):
			print("WARNING: The MCS core of ligands atoms seems not well aligned!")

		# Get the nearest atom of mols from centroid
		self.mcs_centroid = self.mcs_mean_atom_positions.mean(axis=0)
		centroid_to_atom_distances = np.linalg.norm(self.mcs_mean_atom_positions - self.mcs_centroid, axis=1)
		self.mcs_center_atom_index = centroid_to_atom_distances.argmin()

		# Set properties (is in MCS and center atom)
		for i, rdmol in enumerate(self.rdmols):
			rdmol_mcs_atom_indices = self.rdmols_mcs_atom_indices[i]
			rdmol_mcs_bond_indices = self.rdmols_mcs_bond_indices[i]

			# Is atom part of the MCS?
			for atom in rdmol.GetAtoms():
				if atom.GetIdx() in rdmol_mcs_atom_indices:
					atom.SetBoolProp("is_mcs", True)
				else:
					atom.SetBoolProp("is_mcs", False)

			# Is bond part of the MCS?
			for bond in rdmol.GetBonds():
				if bond.GetIdx() in rdmol_mcs_bond_indices:
					bond.SetBoolProp("is_mcs", True)
				else:
					bond.SetBoolProp("is_mcs", False)

			# The center atom index
			rdmol_center_atom_index = rdmol_mcs_atom_indices[self.mcs_center_atom_index]
			rdmol.SetIntProp("atom_center_index", rdmol_center_atom_index.item())


	def find(self):
		self.mcs()

		rdmols_mcs_bond_combo_scores = np.zeros((len(self.rdmols), self.rdmols_mcs_bond_indices.shape[1]))
		rdmols_mcs_bond_rigidity_scores = np.zeros((len(self.rdmols), self.rdmols_mcs_bond_indices.shape[1]))
		rdmols_mcs_bond_strain_scores = np.zeros((len(self.rdmols), self.rdmols_mcs_bond_indices.shape[1]))

		for mol_id, mol in enumerate(self.rdmols):
			# The molecule must have 1 conformer only
			conf = mol.GetConformer(0)

			# Compute dihedral rigidity. Scores are stored as rdkit bond props
			forcefield.mol_rigidity(conf)

			bond_combo_indices, bond_combo_scores, _ = RdkitHelper.props_to_array(mol.GetBonds(), "combo_score", filtering=False)
			bond_rigidity_indices, bond_rigidity_scores, _ = RdkitHelper.props_to_array(mol.GetBonds(), "rigidity_score", filtering=False)
			bond_strain_indices, bond_strain_scores, _ = RdkitHelper.props_to_array(mol.GetBonds(), "strain_score", filtering=False)

			rdmol_mcs_bond_indices = self.rdmols_mcs_bond_indices[mol_id]

			rdmols_mcs_bond_combo_scores[mol_id] = bond_combo_scores[rdmol_mcs_bond_indices]
			rdmols_mcs_bond_rigidity_scores[mol_id] = bond_rigidity_scores[rdmol_mcs_bond_indices]
			rdmols_mcs_bond_strain_scores[mol_id] = bond_strain_scores[rdmol_mcs_bond_indices]

			# # Compute the cumulative rigidity between central atom and all other ones
			# cumrigidity(mol, mol.GetIntProp("atom_center_index"))

			# plot_mol(mol)
		min_rdmols_mcs_bond_combo_scores = rdmols_mcs_bond_combo_scores.min(axis=0)
		min_rdmols_mcs_bond_rigidity_scores = rdmols_mcs_bond_rigidity_scores.min(axis=0)
		min_rdmols_mcs_bond_strain_scores = rdmols_mcs_bond_strain_scores.min(axis=0)

		# Assign bond scores to MCS
		for mcs_bond_id, scores in enumerate(zip(min_rdmols_mcs_bond_combo_scores, min_rdmols_mcs_bond_rigidity_scores, min_rdmols_mcs_bond_strain_scores)):
			combo_bond_score, rigidity_bond_score, strain_bond_score = scores
			mcs_bond = self.mcs.GetBondWithIdx(mcs_bond_id)

			if not np.isnan(combo_bond_score):
				mcs_bond.SetDoubleProp("combo_score", combo_bond_score)
				mcs_bond.SetDoubleProp("rigidity_score", rigidity_bond_score)
				mcs_bond.SetDoubleProp("strain_score", strain_bond_score)

		# Compute the cumulative rigidity between central atom and all other ones
		cumrigidity(self.mcs, self.mcs_center_atom_index)

		plot_mol(self.mcs)

			# continue

	@classmethod
	def atom_path_scoring(rdmol, start_atom_index, bond_prop_name):
		def compute_path_score(path, rdmol, bond_prop_name):
			path_score = 1.0
			# NOTE: The first bond and the last bond are ignored because they are not dihedral
			for i in range(1, len(path) - 2):
				atom_index_1 = path[i]
				atom_index_2 = path[i+1]

				bond = rdmol.GetBondBetweenAtoms(atom_index_1, atom_index_2)

				# If bond does not have prop, there is a termini atom (no dihedral)
				if bond.HasProp("combo_score"):
					path_score *= bond.GetDoubleProp("combo_score")

			return path_score

		G = rdkit_to_nx(rdmol) # All bond props are copied
		paths = []
		for atom in rdmol.GetAtoms():
			atom_index = atom.GetIdx()
			path = nx.shortest_path(G, source=start_atom_index, target=atom_index)

			# Convert np.uint64 to int (np.uint64 not supported by rdkit)
			path = [int(x) for x in path]
			paths.append(path)

			# NOTE: Distance of two bonds or less is obviously ridig
			if len(path) < 4:
				path_score = 1.0
			else:
				path_score = compute_path_score(path, rdmol, bond_prop_name)

			atom.SetDoubleProp("center_path_score", path_score)

		# Weight path_score according to distance. Longer is the distance higher is the weigth
		max_weighted_path_score = 0.0 
		for path in paths:
			cum_score = 1.0
			for atom_index in path:
				atom = rdmol.GetAtomWithIdx(atom_index)
				cum_score *= atom.GetDoubleProp("center_path_score")

			weighted_path_score = cum_score * len(path)
			last_atom = rdmol.GetAtomWithIdx(path[-1])
			last_atom.SetDoubleProp("weighted_center_path_score", weighted_path_score)

			if weighted_path_score > max_weighted_path_score:
				max_weighted_path_score = weighted_path_score


class LigandsInfo(MolobjectInfo):
	def __init__(self, struct_filepath, network_filepath):
		super().__init__(struct_filepath)
		self.raw_rdmols = None
		self.clean_rdmols = None
		self.network = None
		self.common_ligand_frame = None

	def load(self):
		suppl = AllChem.SDMolSupplier(self.struct_filepath, sanitize=False, removeHs=False, strictParsing=False)
		self.raw_rdmols = []
		self.clean_rdmols = []
		for mol_id, raw_mol in enumerate(suppl):
			if raw_mol.GetNumConformers() != 1:
				# TODO: Treate case when mol has 2 or more conformers
				raise ValueError(f"Molecule {mol_id} must have 1 conformer")

			clean_mol = AllChem.Mol(raw_mol) # Copy
			AllChem.SanitizeMol(clean_mol) # Inplace

			# Check if structure changed
			if raw_mol.GetNumAtoms() != clean_mol.GetNumAtoms():
				raise ValueError("Molecule changed after sanitization")

			self.raw_rdmols.append(raw_mol)
			self.clean_rdmols.append(clean_mol)

	def load_network(self):
		network = NetworkInfo(self.network_filepath)
		transformations = []
		for row in network.get_rows():
			transformations.append(TransformationInfo(
				row[plan._ID_A_COLNAME],
				row[plan._ID_B_COLNAME],
				row[plan._NAME_A_COLNAME],
				row[plan._NAME_B_COLNAME],
				row[plan._NAME_B_COLNAME],
			))

	def common_atom_frame(self):
		self.common_ligand_frame = CommonLigandFrame(self.clean_rdmols)
		self.common_ligand_frame.find()

		step = np.pi / 180
		scan_angles = np.arange(-np.pi, np.pi+step, step)
		scan_indices = np.arange(scan_angles.shape[0]).astype(int)

		# Get Maximum Common Substructure
		self.mcs_result = rdFMCS.FindMCS(self.clean_rdmols)

		# The ligands must share at least a substructure containing 3 or more atoms.
		# We need 3 atoms to restraint ligand 1 and ligand 2 during the simulation.
		if self.mcs_result.numAtoms < 3:
			# TODO: Write a method that treat pairs of ligand instead
			raise ValueError(f"MCS must contain 3 atoms or more")

		# Get the MCS and sanitize it to get FF parameters.
		# If it is not sanatized, aromatic atoms may not be properly assigned as aromatic
		tmp_mcs = AllChem.MolFromSmarts(self.mcs_result.smartsString)
		self.mcs_query = AllChem.Mol(tmp_mcs) # Copy
		AllChem.SanitizeMol(tmp_mcs) # Inplace
		self.mcs = AllChem.rdmolops.AddHs(tmp_mcs)

		# Get MCS atom indices for mols
		all_mcs_atom_indices = np.zeros((len(self.clean_rdmols), self.mcs_query.GetNumAtoms())).astype(int)
		for i, mol in enumerate(self.clean_rdmols):
			all_mcs_atom_indices[i] = mol.GetSubstructMatch(self.mcs_query)
			
		# Compute the RMSD of all ligands MCS
		mcs_atom_positions = np.zeros((len(self.clean_rdmols), self.mcs_query.GetNumAtoms(), 3))
		for i, mol in enumerate(self.clean_rdmols):
			conf = mol.GetConformer(0)

			# Get the position for atoms in the MCS
			mcs_atom_indices = all_mcs_atom_indices[i]
			mcs_atom_positions[i] = conf.GetPositions()[mcs_atom_indices]

		# Compute mean positions
		mean_atom_positions = np.mean(mcs_atom_positions, axis=0)

		# Compute standard deviation of atoms around mean positions (kinda like a RMSF)
		std_atoms = np.sqrt(np.mean(np.sum((mcs_atom_positions - mean_atom_positions)**2, axis=2), axis=0))
		if np.any(std_atoms > 1.0):
			print("WARNING: The MCS core of ligands atoms seems not well aligned!")

		# Get the nearest atom of mols from centroid
		mcs_centroid = mean_atom_positions.mean(axis=0)
		centroid_distances = np.linalg.norm(mean_atom_positions - mcs_centroid, axis=1)
		mcs_closest_centroid_index = centroid_distances.argmin()

		# for mol in self.clean_rdmols:
		# Get bond rigidity
		for mol_id, mol in enumerate(self.clean_rdmols):
			# The molecule must have 1 conformer only
			conf = mol.GetConformer(0)

			# Compute dihedral rigidity. Scores are stored as rdkit bond props
			forcefield.mol_rigidity(conf)

			plot_mol(mol)

			breakpoint()

			continue

			# Copy the molecule because coordinates may change
			tested_mol = AllChem.Mol(mol)
			mcs_atom_indices = all_mcs_atom_indices[mol_id]

			bond_scores = {} # bondIdx: (strain_score, rigidity_score)
			for bond in tested_mol.GetBonds():

				# Check if the bond is part of the MCS
				if not bond.GetBeginAtomIdx() in mcs_atom_indices or not bond.GetEndAtomIdx() in mcs_atom_indices:
					continue

				# Check if the bond is in a cycle (aromatic). If yes, the bond get a rigidity score of 1 (max)
				# NOTE: It is not possible to change the dihedral angle in saturated cycle. The rigidity
				# score is set to 1 as well
				if bond.IsInRing():
					bond_scores[bond.GetIdx()] = (1.0, 1.0)
				# Else need to check bond rigidity
				# TODO: Write a function man!
				else:
					atom1 = bond.GetBeginAtom()
					atom2 = bond.GetEndAtom()

					# Get atom1 neighbors (except atom2) to define the first index of the torsion
					# TODO: Write a function
					atom1_neighbor_indices = []
					for neighbor_atom in atom1.GetNeighbors():
						if neighbor_atom.GetIdx() != atom2.GetIdx():
							atom1_neighbor_indices.append(neighbor_atom.GetIdx())

					# Get atom2 neighbors (except atom1) to define the last index of the torsion
					atom2_neighbor_indices = []
					for neighbor_atom in atom2.GetNeighbors():
						if neighbor_atom.GetIdx() != atom1.GetIdx():
							atom2_neighbor_indices.append(neighbor_atom.GetIdx())

					# Check if atom1 and atom2 are bonded to other atoms
					# Skip if not (e.g. bond with H)
					if not atom1_neighbor_indices or not atom2_neighbor_indices:
						continue

					# Get get the torsion atom indices
					tid1 = atom1_neighbor_indices[0]
					tid2 = atom1.GetIdx()
					tid3 = atom2.GetIdx()
					tid4 = atom2_neighbor_indices[0]

					# NOTE: This loop was used to get all combinations of torsion indices					
					# torsion_indices = []
					# for na1 in atom1_neighbor_indices:
					# 	for na2 in atom2_neighbor_indices:
					# 		torsion_indices.append(tuple([na1, atom1.GetIdx(), atom2.GetIdx(), na2]))				

					# Scan torsion angle and compute energy profile using the force field
					AllChem.SanitizeMol(tested_mol) # Just in case!
					mp = AllChem.MMFFGetMoleculeProperties(tested_mol, mmffVariant='MMFF94s')
					ff = AllChem.MMFFGetMoleculeForceField(tested_mol, mp)
					conf = tested_mol.GetConformer(0)
					ref_angle = rdMolTransforms.GetDihedralRad(conf, tid1, tid2, tid3, tid4)
					tested_angles = scan_angles + ref_angle

					energies = []
					for angle in tested_angles:
						rdMolTransforms.SetDihedralRad(conf, tid1, tid2, tid3, tid4, angle)
						ff.Initialize() # This must be call! Otherwise results are wrong (don't know why!!!!)
						energy = ff.CalcEnergy()
						energies.append(energy)

					energies = np.array(energies)

					def two_sides_boltzmann_probabilities(E, i, t):
						left_dE = (E[:i] - E[1:i+1])[::-1] 
						right_dE = E[i+1:] - E[i:-1]

						left_probs = boltzmann(left_dE, t)
						right_probs = boltzmann(right_dE, t)

						return left_probs, right_probs

					# Integrate left and right
					middle_index = int((len(energies) - 1) / 2)
					left_probs, right_probs = two_sides_boltzmann_probabilities(energies, middle_index, 300)

					# Find a range (99% of left and right probs)
					# Set all values above 1 to 1 to avoid cumulative effect when the starting point is not at a local minima
					# NOTE: If you find a better name to replace "flat", it would be great
					flat_left_probs = np.copy(left_probs)
					flat_left_probs[flat_left_probs > 1] = 1

					flat_right_probs = np.copy(right_probs)
					flat_right_probs[flat_right_probs > 1] = 1

					left_cumprobs = np.cumprod(flat_left_probs)
					right_cumprobs = np.cumprod(flat_right_probs)

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
					minimum_index = scan_indices[left_index:right_index+1][minimum_sub_index]

					# Then, compute the strain score
					de = energies[middle_index] - energies[minimum_index]

					# This score is the "probability" of a dihedral to move from its initial angle
					# 0 it will move without external forces to maintain the conformation
					# 1 meens it will not move
					strain_score =  boltzmann(de, 300)

					# Compute the flexibility score
					left_angle = scan_angles[left_index]
					right_angle = scan_angles[right_index]

					rigidity_score = 1 - (right_angle - left_angle) / (np.pi*2)

					bond_scores[bond.GetIdx()] = (strain_score, rigidity_score)

					# if True:
					# 	plt.ion()
					# 	plt.plot(tested_angles, energies)
					# 	plt.axvline(ref_angle, linestyle="dotted")
					# 	plt.axvline(tested_angles[right_index], linestyle="dotted")
					# 	plt.axvline(tested_angles[left_index], linestyle="dotted")
					# 	plt.draw()

					# 	print("Torsion indices:", tid1, tid2, tid3, tid4)
					# 	print("Ref angle:", ref_angle)

					# 	min_indices = argrelmin(energies)
					# 	max_indices = argrelmax(energies)

					# 	print("Minima angles", tested_angles[min_indices])
					# 	print("Minima energies", energies[min_indices])

					# 	print("Maxima angles", tested_angles[max_indices])
					# 	print("Maxima energies", energies[max_indices])

					# 	print("Barrier indices", left_index, middle_index, right_index)

					# 	draw_mol = AllChem.Mol(tested_mol)
					# 	AllChem.Compute2DCoords(draw_mol)
					# 	# Draw.MolToImage(molid(draw_mol), size=(600, 600)).show()

			# Compute the cumulative rigidity between central atom and all other ones
			mol_graph = rdkit_to_nx(tested_mol)
			atom_scores = {}
			paths = []
			for atom_index in mcs_atom_indices:
				path = nx.shortest_path(mol_graph, source=mcs_closest_centroid_index, target=atom_index)
				paths.append(path)

				# NOTE: Distance of two bonds or less is obviously ridig
				if len(path) < 4:
					atom_scores[atom_index.item()] = 1.0
					continue

				path_score = 1.0
				# NOTE: The first bond and the last bond are ignored because they are not dihedral
				for i in range(1, len(path) - 2):
					# NOTE: int casting is due to rdkit function signature (np.int64 not supported)
					a1 = int(path[i])
					a2 = int(path[i+1])

					atom_1 = tested_mol.GetAtomWithIdx(a1)
					atom_2 = tested_mol.GetAtomWithIdx(a2)

					# Ignore hydrogens
					# if tested_mol.GetAtomWithIdx(a1).GetSymbol() == 'H' or tested_mol.GetAtomWithIdx(a2).GetSymbol() == 'H':
					# 	continue

					bond_id = tested_mol.GetBondBetweenAtoms(a1,a2).GetIdx()

					# If bond id does not exist, it means that there is a termini atom
					if bond_id in bond_scores:
						s, r = bond_scores[bond_id]
						path_score *= s * r


				atom_scores[atom_index.item()] = path_score


			longest_path_scores = {}
			max_longest_score = 0.0 
			for path in paths:
				score = 1.0
				for atom_index in path:
					score *= atom_scores[atom_index]

				longest_path_scores[path[-1]] = score * len(path)
				if longest_path_scores[path[-1]] > max_longest_score:
					max_longest_score = longest_path_scores[path[-1]]

			draw_mol = AllChem.Mol(tested_mol)
			AllChem.Compute2DCoords(draw_mol)

			highlight_atoms = []
			highlight_atom_colors = {}
			for atom_id in atom_scores:
				score = longest_path_scores[atom_id]
				highlight_atoms.append(atom_id)

				if atom_id == mcs_closest_centroid_index:
					highlight_atom_colors[atom_id] = (0, 0, 1)
				else:
					highlight_atom_colors[atom_id] = (0, score / max_longest_score, 0)

				draw_mol.GetAtomWithIdx(atom_id).SetProp('atomNote', f"{score:.2f}")


			highlight_bonds = []
			highlight_bond_colors = {}
			for bond_id in bond_scores:
				s, r = bond_scores[bond_id]
				score = r * s
				highlight_bonds.append(bond_id)
				highlight_bond_colors[bond_id] = (score, 0, 0)
				draw_mol.GetBondWithIdx(bond_id).SetProp('bondNote', f"{score:.2f}")

			for atom in draw_mol.GetAtoms():
				atom.ClearProp("molAtomMapNumber")

			d2d = Draw.MolDraw2DCairo(500,500)
			Draw.rdMolDraw2D.PrepareAndDrawMolecule(d2d, draw_mol, kekulize=False,
				highlightAtoms=highlight_atoms, highlightAtomColors=highlight_atom_colors,
				highlightBonds=highlight_bonds,highlightBondColors=highlight_bond_colors
			)

			d2d.FinishDrawing()
			sio = BytesIO(d2d.GetDrawingText())
			Image.open(sio).show()

			# draw = Draw.rdMolDraw2D.MolDraw2DCairo(500, 500)
			# Draw.rdMolDraw2D.PrepareAndDrawMolecule(draw, mol, highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)
			# d2d.DrawMolecule(mol, highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors)

			# Draw.MolToImage(draw_mol, size=(600, 600), kekulize=False, highlightBonds=highlight_bonds, highlightBondColors=highlight_bond_colors).show()

			breakpoint()

			# Assign a score to atom according to bond strain and rigidity scores


			# FOR NEXT TIME
			# Find the three reference atoms for ATM simulations
			# Find rigid part of MCS


	def get_positions(self):
		positions = []

		for mol in self.raw_rdmols:
			for conf in mol.GetConformers():
				positions.extend(conf.GetPositions())

		return np.array(positions)


class ProteinInfo(MolobjectInfo):
	def __init__(self, struct_filepath):
		super().__init__(struct_filepath)

	def load(self):
		self.parser = PDB(self.struct_filepath)
		self.parser.parse()

	def get_positions(self):
		return self.parser.positions


def main(args):
	if not os.path.isdir(args.ligands_ff_dirpath) or not os.access(args.ligands_ff_dirpath, os.R_OK):
		print(f"ERROR: Ligands force field directory {args.ligands_ff_dirpath} not found or not readable!")
		return 1

	if not os.path.isdir(args.optimize_dirpath) or not os.access(args.optimize_dirpath, os.R_OK):
		print(f"ERROR: Optimize directory {args.optimize_dirpath} not found or not readable!")
		return 1

	if not os.path.isdir(args.network_dirpath) or not os.access(args.network_dirpath, os.R_OK):
		print(f"ERROR: Network directory {args.network_dirpath} not found or not readable!")
		return 1

	# if not os.path.isfile(args.protein_filepath) or not os.access(args.protein_filepath, os.R_OK):
	# 	print(f"ERROR: receptor file {args.protein_filepath} not found or not readable!")
	# 	return 1

	# if not os.path.isfile(args.network_filepath) or not os.access(args.network_filepath, os.R_OK):
	# 	print(f"ERROR: network file {args.network_filepath} not found or not readable!")
	# 	return 1

	if not os.path.isdir(args.output_dirpath):
		os.mkdir(args.output_dirpath)
	elif args.overwrite:
		print(f"WARNING: Files in the output directory '{args.output_dirpath}' will be overwritten.")
	else:
		print(f"CRITICAL: The output directory '{args.output_dirpath}' already exists. Use '-w' to force overwrite.")
		return 1

	# Get filepaths
	optimize_protein_filepath = os.path.join(args.optimize_dirpath, optimize.System._PROTEIN_FILENAME)
	optimize_info_filepath = os.path.join(args.optimize_dirpath, optimize.CONST.INFO_FILENAME)
	network_filepath = os.path.join(args.network_dirpath, plan._NETWORK_FILENAME)

	# if not os.path.isfile(args.info_filepath) or not os.access(args.info_filepath, os.R_OK):
	# 	print(f"ERROR: info file {args.info_filepath} not found or not readable!")
	# 	return 1

	# Load input
	protein_info = ProteinInfo(optimize_protein_filepath)
	try:
		protein_info.load()
	except BaseException as e:
		print(f"CRITICAL: The protein file is not readable! The program raised the following error: {e}")
		return 1

	ligands_filepath = os.path.join(args.optimize_dirpath, optimize.System._LIGAND_1_FILENAME)
	ligands_info = LigandsInfo(ligands_filepath, network_filepath)
	try:
		ligands_info.load()
	except BaseException as e:
		print(f"CRITICAL: The ligands file is not readable! The program raised the following error: {e}")
		return 1

	# Step 1: Definition of the binding site
	print(f"INFO: Defining binding site.")
	kdtree_binding_site = KDTree(ligands_info.get_positions())
	distances, _ = kdtree_binding_site.query(protein_info.get_positions())
	protein_bs_mask = (distances < _DEFAULTS["binding_site_threshold"]) & (protein_info.parser.names == "CA")

	if not protein_bs_mask.sum():
		print("CRITICAL: No binding site residue was found!")
		return 1

	bs_resnames = protein_info.parser.resnames[protein_bs_mask]
	bs_resids = protein_info.parser.resids[protein_bs_mask]
	bs_positions = protein_info.parser.positions[protein_bs_mask]
	bs_centroid = bs_positions.mean(axis=0)

	print(f"INFO: {len(bs_resnames)} residues defined as binding site.")

	# Step 2: Retrieve the displacement vector from optimize.py
	with open(optimize_info_filepath) as f:
		optimize_data = yaml.load(f, Loader=yaml.Loader)
	displacement_vector = optimize_data["displacement_vector"]

	print("INFO: MCS")
	ligands_info.common_atom_frame()
	breakpoint()

	# Make transformation directories and setup systems
	for index, row in network.iterrows():
		id_a = plan.format_id(row[plan._ID_A_COLNAME])
		id_b = plan.format_id(row[plan._ID_B_COLNAME])
		subdirname = f"{id_a}_{id_b}"
		subdirpath = os.path.join(args.output_dirpath, subdirname)

		if not os.path.isdir(subdirpath):
			os.mkdir(subdirpath)

		# Load ligands
		ligand_a_filepath = os.path.join(args.ligands_ff_dirpath, id_a+".sdf")
		if not os.path.isfile(ligand_a_filepath):
			print("CRITICAL: Ligand A {ligand_a_filepath} is not found! Run 'ligff.py' before!")
			return 1

		ligand_b_filepath = os.path.join(args.ligands_ff_dirpath, id_b+".sdf")
		if not os.path.isfile(ligand_b_filepath):
			print("CRITICAL: Ligand B {ligand_b_filepath} is not found! Run 'ligff.py' before!")
			return 1

		lig_a = AllChem.Mol(ligand_a_filepath)
		lig_b = AllChem.SDMolSupplier(ligand_b_filepath)

		breakpoint()


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--ligands", required=True, dest="ligands_ff_dirpath")
	parser.add_argument("-opt", "--optimize", required=True, dest="optimize_dirpath")
	parser.add_argument("-n", "--network", required=True, dest="network_dirpath")
	# parser.add_argument("-i", "--info", required=True, dest="info_filepath")
	parser.add_argument("-o", "--output", required=True, dest="output_dirpath")
	parser.add_argument("-w", "--overwrite", action="store_true")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)