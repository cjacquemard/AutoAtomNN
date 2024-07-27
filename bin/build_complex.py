import sys
import os

# MDAanalysis warns about missing mass for some elements. Can be totally ignored but catch it to not worried user.
import warnings

import pygad
import numpy as np
import MDAnalysis as mda
from scipy.spatial.transform import Rotation as R
from rdkit import Chem


def near_to(reference_positions, configuration_positions, max_cutoff, min_cutoff=None, reverse=False):
	"""
	Return reference atoms that are between min_cutoff and max_cutoff of configuration atoms
	"""
	pairs, distances = mda.lib.distances.capped_distance(
		reference_positions, configuration_positions, max_cutoff, min_cutoff
	)

	if reverse:
		indices = np.setdiff1d(np.arange(len(reference_positions)), np.unique(pairs[:, 0]))
	else:
		indices = np.unique(pairs[:, 0])

	return reference_atoms[indices]

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
	   return v
	return v / norm


def eigen(points):
	# Center the point cloud around the origin
	centroid = np.mean(points, axis=0)
	centered_point_cloud = points - centroid

	# Compute the covariance matrix
	cov_matrix = np.cov(centered_point_cloud, rowvar=False)

	# Compute eigenvalues and eigenvectors
	eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

	# Sort the eigenvectors by eigenvalues in descending order
	idx = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[idx]
	eigenvectors = eigenvectors[:, idx]

	return eigenvalues, eigenvectors

class GA:
	_MIN_DIST = 15

	def __init__(self, prot_u, lig1_u, lig2_u):
		self.prot_u = prot_u
		self.lig1_u = lig1_u
		self.lig1_com = lig1_u.select_atoms("not element H").center_of_geometry()

		self.lig2_u = lig2_u

		self.complex_atom_positions = np.vstack([
			prot_u.select_atoms("not element H").positions,
			lig1_u.select_atoms("not element H").positions
		])

		self.complex_bbox = np.array([
			np.min(self.complex_atom_positions, axis=0),
			np.max(self.complex_atom_positions, axis=0),
		])

		self.complex_dimension = self.complex_bbox[0] - self.complex_bbox[1]
		self.complex_volume = np.prod(self.complex_dimension)

		self.lig2_atoms = lig2_u.select_atoms("not element H")
		self.lig2_atoms.translate(-self.lig2_atoms.center_of_geometry())
		self.lig2_guessed_com = np.array([0, 0, 0])

		self.ga_instance = pygad.GA(num_generations=50,
					   num_parents_mating=4,
					   fitness_func=self.fitness_function,
					   sol_per_pop=8,
					   num_genes=3,
					   init_range_low=self.complex_bbox[0].min(),
					   init_range_high=self.complex_bbox[1].max(),
					   parent_selection_type="sss",
					   keep_parents=1,
					   crossover_type="single_point",
					   mutation_type="random",
					   mutation_percent_genes=10)
		self.ga_instance.run()

	@property
	def lig2_guessed_positions(self):
		return self.lig2_atoms.positions + self.lig2_guessed_com
	

	def _fitness_farest_ligand(self):
		indices = near_to(self.complex_atom_positions, self.lig2_guessed_com)
		if indices:
			return 0
		else:
			return 100

	def _fitness_smallest_bbox(self):
		lig2_bbox = np.array([
			np.min(self.lig2_guessed_positions),
			np.max(self.lig2_guessed_positions)
		])

		new_bbox = np.array([
			np.min([self.complex_dimension[0], lig2_bbox[0]]),
			np.min([self.complex_dimension[1], lig2_bbox[1]]),
		])

		new_dimension = new_bbox[1] - new_bbox[0]
		new_volume = np.prod(new_dimension)

		increase_factor = (new_volume - self.complex_volume) / self.complex_volume

		if increase_factor <= 0:
			return 100
		else:
			return 100 * np.exp(-increase_factor)


	def _fitness_lig_direction(self):
		"""Cosine similarity"""
		return 100 * np.dot(self.lig1_com, self.lig2_guessed_com) / (np.linalg.norm(self.lig1_com) * np.linalg.norm(self.lig2_guessed_com))


	def fitness_function(self):
		return np.array([
			self._fitness_farest_ligand(),
			self._fitness_smallest_bbox(),
			self._fitness_lig_direction(),
		])


def main(args):
	if os.path.isdir(args.output_dirpath) and not args.overwrite:
		print(f"ERROR: directory {args.output_dirpath} already exists! Use '-w' to force overwrite.")
		return 1

	if not os.path.isfile(args.ligand1_filepath) or not os.access(args.ligand1_filepath, os.R_OK):
		print(f"ERROR: Ligand 1 file {args.ligand1_filepath} not found or not readable!")
		return 1

	if not os.path.isfile(args.ligand2_filepath) or not os.access(args.ligand2_filepath, os.R_OK):
		print(f"ERROR: Ligand 2 file {args.ligand2_filepath} not found or not readable!")
		return 1

	if not os.path.isfile(args.protein_filepath) or not os.access(args.protein_filepath, os.R_OK):
		print(f"ERROR: Protein file {args.protein_filepath} not found or not readable!")
		return 1

	# Catch MDAanalysis warn about missing mass for some elements that is not use in this script.
	with warnings.catch_warnings(record=True) as w:
		# Load molecules
		prot_u = mda.Universe(args.protein_filepath)
		# lig1_u = mda.Universe(args.ligand1_filepath)
		# lig2_u = mda.Universe(args.ligand2_filepath)

		if len(w):
			if not issubclass(w[-1].category, UserWarning):
				print("DEV: Some unexpected warnings here!")

	# Load ligand using rdkit because MDAnalysis is not that good
	mol1 = Chem.MolFromSDFFile(args.ligand1_filepath)
	mol2 = Chem.MolFromSDFFile(args.ligand2_filepath)

	breakpoint()

	# Center protein and ligands according to protein center of geometry
	prot_cog = prot_u.atoms.center_of_geometry()
	prot_u.atoms.translate(-prot_cog)
	lig1_u.atoms.translate(-prot_cog)
	lig2_u.atoms.translate(-prot_cog)

	# Compute principal axis of the protein
	eigenvalues, eigenvectors = eigen(prot_u.atoms.positions)

	# Align the first principal axis onto the diagonal of cube
	first_principal_axis = eigenvectors[:, 0]
	ref_vector = normalize([1, 1, 1])
	diag_align, rmsd = R.align_vectors(ref_vector[np.newaxis,:], first_principal_axis[np.newaxis,:])
	prot_u.atoms.rotate(diag_align.as_matrix())
	lig1_u.atoms.rotate(diag_align.as_matrix())
	lig2_u.atoms.rotate(diag_align.as_matrix())

	ga = GA(prot_u, lig1_u, lig2_u)

	breakpoint()

	# Run GA to find the best placement for the second ligand


	# # Align the first principal axis onto the z axis
	# first_principal_axis = eigenvectors[:, 0]
	# z_axis = np.array([0, 0, 1])
	# z_align = R.align_vectors(z_axis[np.newaxis,:], first_principal_axis[np.newaxis,:])
	# prot_u.atoms.rotate(z_align.as_matrix())
	# lig1_u.atoms.rotate(z_align.as_matrix())
	# lig2_u.atoms.rotate(z_align.as_matrix())

	# # Get the vector where the bound ligand points out projected on xy plane
	# ligand1_vector = normalize(lig1_u.atoms.center_of_geometry())

	# # Align the second principal to that vector
	# second_principal_axis = eigenvectors[:, 1]
	# lig_align = R.align_vectors(ligand1_vector[np.newaxis,:], second_principal_axis[np.newaxis,:])
	# prot_u.atoms.rotate(lig_align.as_matrix())
	# lig1_u.atoms.rotate(lig_align.as_matrix())
	# lig2_u.atoms.rotate(lig_align.as_matrix())

	return 0

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-l1", "--ligand1", required=True, dest="ligand1_filepath", help="NOTE: Do not provide the file generated by ANTECHAMBER!")
	parser.add_argument("-l2", "--ligand2", required=True, dest="ligand2_filepath", help="NOTE: Do not provide the file generated by ANTECHAMBER!")
	parser.add_argument("-p", "--protein", required=True, dest="protein_filepath")
	parser.add_argument("-o", "--output", required=True, dest="output_dirpath")
	parser.add_argument("-w", "--overwrite", action="store_true")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)
