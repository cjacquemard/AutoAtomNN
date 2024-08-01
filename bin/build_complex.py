import sys
import os
from copy import copy
from time import time

# MDAanalysis warns about missing mass for some elements. Can be totally ignored but catch it to not worried user.
import warnings

import pygad
import numba
import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist
from rdkit import Chem
from rdkit.Chem.AllChem import TransformMol
from sklearn.cluster import SpectralClustering, KMeans

import cgo

def translate_mol(mol, translation_vector):
	transformation_matrix = np.eye(4)
	transformation_matrix[:3, 3] = translation_vector

	# Apply the transformation
	TransformMol(mol, transformation_matrix)


def rotate_mol(mol, rotation_matrix):
	transformation_matrix = np.eye(4)
	transformation_matrix[:3, :3] = rotation_matrix

	# Apply the transformation
	TransformMol(mol, transformation_matrix)


# def near_to(reference_positions, configuration_positions, max_cutoff, min_cutoff=None, reverse=False):
# 	"""
# 	Return reference atoms that are between min_cutoff and max_cutoff of configuration atoms
# 	"""
# 	pairs, distances = mda.lib.distances.capped_distance(
# 		reference_positions, configuration_positions, max_cutoff, min_cutoff
# 	)

# 	if reverse:
# 		indices = np.setdiff1d(np.arange(len(reference_positions)), np.unique(pairs[:, 0]))
# 	else:
# 		indices = np.unique(pairs[:, 0])

# 	return indices


def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
	   return v
	return v / norm


def get_rdkit_ha_positions(mol):
		positions = mol.GetConformer().GetPositions()
		ha_indices = np.array([atom.GetSymbol() != 'H' for atom in mol.GetAtoms()])
		return positions[ha_indices]


@numba.njit
def wrap(points, bbox, vector):
	box_size = bbox[1] - bbox[0]
	box_center = (bbox[0] + bbox[1]) / 2
	box_shift = vector - box_center
	cell_indices = (box_shift / box_size).astype(np.int64)
	relative_box_shift = box_shift + box_shift * cell_indices
	new_bbox = bbox + relative_box_shift
	wrapped_points = np.zeros(points.shape)

	for i in range(points.shape[0]):
		for axis in range(points.shape[1]):
			if points[i, axis] < new_bbox[0][axis]:
				wrapped_points[i, axis] = points[i, axis] + box_size[axis]
			elif points[i, axis] > new_bbox[1][axis]:
				wrapped_points[i, axis] = points[i, axis] - box_size[axis]
			else:
				wrapped_points[i, axis] = points[i, axis]

	return wrapped_points


class Box:
	def __init__(self, bbox):
		self._bbox = bbox

	@property
	def bbox(self):
		return self._bbox

	@property
	def lower(self):
		return self._bbox[0]

	@property
	def upper(self):
		return self._bbox[1]

	@property
	def size(self):
		return self._bbox[1] - self._bbox[0]

	@property
	def volume(self):
		return np.prod(self.size)

	@property
	def center(self):
		return (self.upper + self.lower) / 2


class MolObject:
	def __init__(self, *positions):
		self.positions = np.vstack([pos for pos in positions])
		self.original_positions = self.positions.copy()

	@property
	def box(self):
		return Box(np.array([
			np.min(self.positions, axis=0),
			np.max(self.positions, axis=0)
		]))

	@property
	def cog(self):
		return np.mean(self.positions, axis=0)

	@property
	def original_cog(self):
		return np.mean(self.original_positions, axis=0)

	def eigen(self):
		# NOTE: To get principal axis, positions must be centered

		# Compute the covariance matrix
		cov_matrix = np.cov(self.positions, rowvar=False)

		# Compute eigenvalues and eigenvectors
		eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

		# Sort the eigenvectors by eigenvalues in descending order
		idx = eigenvalues.argsort()[::-1]
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[:, idx]

		return eigenvalues, eigenvectors

	def translate(self, vector):
		self.positions += vector

	def center(self, method="bbox"):
		if method == "cog":
			vector = -self.cog
			self.translate(vector)

			return vector
		elif method == "bbox":
			vector = -self.box.center
			self.translate(vector)

			return vector			
		else:
			raise NotImplementedError

	# def pose(self, position, method="cog"):
	# 	self.center(method)
	# 	self.translate(position)

	def rotate(self, rotation_matrix):
		self.positions = np.dot(rotation_matrix, self.positions.T).T


class System:
	_MIN_DIST = 20
	_SOLVENT_THICKNESS = 10
	_MIN_PARENT_MATING = 10
	_GA_OPTS = {
		"num_generations": 100,
		"num_parents_mating": 10,
		# "sol_per_pop": 50,
		# "num_genes": 3,
		"parent_selection_type": "sss",
		# "mutation_type": "random",
		"mutation_probability": 0.3,
		# "keep_parents": 10,
		# "crossover_type": "single_point",
		# "mutation_type": "random",
		# "mutation_percent_genes": 10
	}

	# FILES
	_POPULATION_FILENAME = "populations.pdb"

	def __init__(self, protein_atom_positions, ligand_1_atom_positions, ligand_2_atom_positions, output_dirpath):
		self.complex = MolObject(protein_atom_positions, ligand_1_atom_positions)
		self.ligand_1 = MolObject(ligand_1_atom_positions)
		self.ligand_2 = MolObject(ligand_2_atom_positions)

		self.output_dirpath = output_dirpath

	@property
	def population_filepath(self):
		return os.path.join(self.output_dirpath, self._POPULATION_FILENAME)

	@classmethod
	def save_population(cls, filepath, population):
		name = "DU"
		altloc = ''
		resname = "POP"
		chainid = 'A'
		resid = 1
		icode = ''
		occ = 1.0
		temp = 0.0
		element = "Du"
		charge = "0"

		with open(filepath, 'a') as f:
			f.write("MODEL\n")
			for i, individual in enumerate(population, 1):
				x, y, z = individual
				f.write(f"HETATM{i: >5}{name: >4}{altloc:1}{resname: >3}{chainid: >2}{resid: >4}{icode: >4}{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{temp:6.2f}            {element: >2}{charge: >2}\n")
			f.write("ENDMDL\n")

	def clean_output(self):
		# Remove population file
		if os.path.isfile(self.population_filepath):
			os.remove(self.population_filepath)

	@property
	def max_box(self):
		"""
		Compute the size of the system in the worst case.
		Worst case corresponds to the ligand 2 laying on the diagonal of the 
		box with a distance _MIN_DIST away from any atom of the complex.
		"""
		min_diagonal_distance = self._MIN_DIST / np.sqrt(3)

		max_box = Box(np.array([
			self.complex.box.lower - min_diagonal_distance - self.ligand_2.box.size,
			self.complex.box.upper + min_diagonal_distance + self.ligand_2.box.size,
		]))

		return max_box

	def _init_population(self):
		"""Evenly place ligand 2 poses in the system"""
		self.ligand_2.center()
		max_box = self.max_box

		# Positions refer to initial ligand poses based on center of geometry
		# self.init_all_pop_positions =  np.array(np.meshgrid(
		# 	np.arange(max_box.lower[0], max_box.upper[0]+0.1, self.ligand_2.box.size[0]),
		# 	np.arange(max_box.lower[1], max_box.upper[1]+0.1, self.ligand_2.box.size[1]),
		# 	np.arange(max_box.lower[2], max_box.upper[2]+0.1, self.ligand_2.box.size[2])
		# )).T.reshape(-1, 3)


		# TODO: Take into account system that are not a cube (e.g. long parallelepiped)
		# To distribute initial positions
		n_init_sol_per_axis = np.array([10, 10, 10])

		self.init_all_pop_positions =  np.array(np.meshgrid(
			np.linspace(max_box.lower[0], max_box.upper[0], n_init_sol_per_axis[0], endpoint=True),
			np.linspace(max_box.lower[1], max_box.upper[1], n_init_sol_per_axis[1], endpoint=True),
			np.linspace(max_box.lower[2], max_box.upper[2], n_init_sol_per_axis[2], endpoint=True)
		)).T.reshape(-1, 3)

		self.init_unit = np.array([
			(max_box.upper[0] - max_box.lower[0]) / (n_init_sol_per_axis[0] - 1),
			(max_box.upper[1] - max_box.lower[1]) / (n_init_sol_per_axis[1] - 1),
			(max_box.upper[2] - max_box.lower[2]) / (n_init_sol_per_axis[2] - 1),
		])
		self.init_unit_diag_size = np.linalg.norm(self.init_unit)

		# Check initial positions to close from complex
		keep_init_indices = []
		complex_kdtree = KDTree(self.complex.positions)
		for i, position in enumerate(self.init_all_pop_positions):
			distance = complex_kdtree.query(self.ligand_2.positions + position)[0].min()

			if distance >= self._MIN_DIST:
				keep_init_indices.append(i)

		self.keep_init_indices = np.array(keep_init_indices)
		self.init_pop_positions = self.init_all_pop_positions[self.keep_init_indices]

		return self.init_pop_positions

	def optimize(self, **kwargs):
		self.clean_output()

		self.ligand_2.center()

		ga_opts = copy(self._GA_OPTS)
		ga_opts.update(kwargs)

		max_box = self.max_box
		ga_opts["init_range_low"] = max_box.lower
		ga_opts["init_range_high"] = max_box.upper
		ga_opts["initial_population"] = self._init_population()
		ga_opts["fitness_func"] = self.fitness_function
		ga_opts["mutation_type"] = self.mutation_func
		ga_opts["crossover_type"] = self.crossover_func
		# ga_opts["parent_selection_type"] = self.selection_func
		# ga_opts["num_parents_mating"] = int(self.init_pop_positions.shape[0] / 10) + 1

		ga_opts["on_start"] = self.on_generation
		ga_opts["on_generation"] = self.on_generation
		ga_opts["on_stop"] = self.on_stop

		self.ligand_1_kdtree = KDTree(self.ligand_1.positions)
		self.complex_solvated_box = Box(np.array([
			self.complex.box.lower - self._SOLVENT_THICKNESS,
			self.complex.box.upper + self._SOLVENT_THICKNESS,
		]))

		self.ga_instance = pygad.GA(**ga_opts)
		self.ga_instance.run()

		self.best_solution, self.best_solution_fitness, self.best_solution_idx = self.ga_instance.best_solution()

		# Compute scores using best solution
		self.fitness_function(None, self.best_solution, None)


	def on_generation(self, ga_instance):
		print(f"Diversity: {ga_instance.population.std():.4f}")
		self.save_population(self.population_filepath, ga_instance.population)
		# fig = plt.figure()
		# ax = fig.add_subplot(projection='3d')
		# ax.scatter(ga_instance.population[:,0], ga_instance.population[:,1], ga_instance.population[:,2])
		# fig.show()
		# breakpoint()
		# plt.cla()
		# plt.clf()

	def on_stop(self, ga_instance, fitness):
		print(f"Diversity: {ga_instance.population.std():.4f}")
		# fig = plt.figure()
		# ax = fig.add_subplot(projection='3d')
		# ax.scatter(ga_instance.population[:,0], ga_instance.population[:,1], ga_instance.population[:,2])
		# fig.show()
		# breakpoint()
		# plt.cla()
		# plt.clf()

	def mutation_func(self, offspring, ga_instance):
		# Randmoly select 33% of population
		number = int(offspring.shape[0]*0.33)+1
		selected_indices = np.random.choice(offspring.shape[0], number, replace=False)

		for i in selected_indices:
			offspring[i] += np.random.normal(size=3) * 2

		return offspring

	def selection_func(self, fitness, num_parents, ga_instance):
		sorted_indices = np.argsort(fitness)[::-1]
		sorted_fitness = fitness[sorted_indices]

		# Keep positive fitness only
		good_solution_mask = sorted_fitness > 0
		good_sorted_indices = sorted_indices[good_solution_mask]
		good_sorted_fitness = fitness[good_sorted_indices]
		good_population = ga_instance.population[good_sorted_indices]

		# clustering = SpectralClustering(n_clusters=10, eigen_solver="arpack", affinity="nearest_neighbors").fit(good_population)
		clustering = KMeans(n_clusters=10).fit(good_population)
		

		breakpoint()

		# Change the number of parent mating or PyGAD will complains
		ga_instance.num_parents_mating = good_sorted_indices.shape[0]

		if ga_instance.num_parents_mating < self._MIN_PARENT_MATING:
			raise ValueError(f"Number of parents is too low ({ga_instance.num_parents_mating} < {self._MIN_PARENT_MATING})")

		ga_instance.good_fitness = good_sorted_fitness

		return ga_instance.population[good_sorted_indices], good_sorted_indices


	def crossover_func(self, parents, offspring_size, ga_instance):
		parent_distances = pdist(parents)
		offspring = []
		idx = 0
		N = parents.shape[0]
		while len(offspring) != offspring_size[0]:
			i = idx % parents.shape[0]
			j = (idx + 1) % parents.shape[0]
			k = int((N*(N-1)/2) - (N-i)*((N-i)-1)/2 + j - i - 1)

			if parent_distances[k] < self.init_unit_diag_size*2:
				parent_1 = parents[i]
				parent_2 = parents[j]

				direction = parent_2 - parent_1
				distance = np.linalg.norm(direction)
				norm_direction = normalize(direction)
				scale = distance * 0.2
				offspring.append(((parent_1 + parent_2) / 2) + norm_direction * np.random.normal(scale=scale)) 
			idx += 1

		return np.array(offspring)

	# def crossover_func(self, parents, offspring_size, ga_instance):
	# 	parent_distances = pdist(parents)
	# 	fitness_scores = ga_instance.good_fitness
	# 	sorted_indices = np.argsort(fitness_scores)[::-1]
	# 	parents = parents[sorted_indices]

	# 	offspring = []
	# 	idx = 0
	# 	#np.ravel_multi_index([40, 13], ())
	# 	# Keep the top 3 parents
	# 	try:
	# 		offspring.extend(parents[sorted_indices[:3]])
	# 	except IndexError:
	# 		breakpoint()

	# 	# for i in range(offspring_size - 3):
	# 	# 	method = np.random.randint(1, 4)

	# 	# 	if method == 1:
	# 	# 		# Randomly moves a parent around a sphere


	# 	N = parents.shape[0]

	# 	while len(offspring) != offspring_size[0]:
	# 		i = idx % parents.shape[0]
	# 		j = (idx + 1) % parents.shape[0]

	# 		k = int((N*(N-1)/2) - (N-i)*((N-i)-1)/2 + j - i - 1)

	# 		if parent_distances[k] < self.init_unit_diag_size:
	# 			parent_1 = parents[i]
	# 			parent_2 = parents[j]
	# 			offspring.append((parent_1 + parent_2) / 2)
	# 		idx += 1

	# 	return np.array(offspring)


	# Define three functions to optimize:
	#   1) The ligand 2 must be far away from any atoms of the complex (protein and ligand 1)
	#   2) The ligand 2 cannot be too far away otherwise the system size increase (computational cost for the RBFE estimation)
	#   3) The ligand 2 should be close to ligand 1 as possible
	def fitness_function(self, ga_instance, solution, solution_idx):
		self.solution = solution
		self.solution_positions = self.ligand_2.positions + solution

		# Compute the new system box
		solution_ligand_2_box = Box(np.array([
			np.min(self.solution_positions, axis=0),
			np.max(self.solution_positions, axis=0),
		]))

		self.new_box = Box(np.array([
			np.min([self.complex.box.lower, solution_ligand_2_box.lower], axis=0) - self._SOLVENT_THICKNESS,
			np.max([self.complex.box.upper, solution_ligand_2_box.upper], axis=0) + self._SOLVENT_THICKNESS,
		]))


		self.increase_factor = (self.new_box.volume - self.complex_solvated_box.volume) / self.complex_solvated_box.volume

		# Wrapped complex atoms around ligand 2
		self.wrapped_complex_positions = wrap(
			self.complex.positions,
			self.new_box.bbox,
			solution,
		)

		# bbox_selection = np.array([
		# 	self.ligand_2.box.lower - self._MIN_DIST - self.new_box.center,
		# 	self.ligand_2.box.upper + self._MIN_DIST - self.new_box.center
		# ])

		# keep_indices = np.all(np.logical_and(
		# 	self.wrapped_complex_positions > bbox_selection[0],
		# 	self.wrapped_complex_positions < bbox_selection[1]
		# ), axis=1)

		# self.close_wrapped_positions = self.wrapped_complex_positions[keep_indices]
		self.close_wrapped_positions = self.wrapped_complex_positions

		if len(self.close_wrapped_positions) == 0:
			# Find the closest point
			self.score_1 = 100
		else:
			# Intrisingly, cdist (brute force) is faster than KDTree
			self.min_distance_to_complex = cdist(self.close_wrapped_positions, self.solution_positions).min()	
			# complex_kdtree = KDTree(self.close_wrapped_positions)
			# self.min_distance_to_complex = complex_kdtree.query(self.solution_positions)[0].min()

			if self.min_distance_to_complex > self._MIN_DIST:
				self.score_1 = 100
			else:
				self.score_1 = 0

			# self.score_1 = 100 * (1 - np.exp(2 * (-self.min_distance_to_complex + self._MIN_DIST)))

		# Compute distance to ligand 1
		self.min_distance_to_ligand_1 = self.ligand_1_kdtree.query(self.solution_positions)[0].min()

		# Compute scores
		if self.increase_factor <= 0:
			self.score_2 = 100
		else:
			self.score_2 = 100 * np.exp(-2 * self.increase_factor)

		if self.min_distance_to_ligand_1 < self._MIN_DIST:
			self.score_3 = 100 * (1 - np.exp(2 * (-self.min_distance_to_ligand_1 + self._MIN_DIST)))
		else:
			self.score_3 = 100 * np.exp(0.1 * (-self.min_distance_to_ligand_1 + self._MIN_DIST))

		scores = np.array([
			self.score_1, self.score_2, self.score_3
		])

		if np.all(scores > 0):
			unary = 1
		else:
			unary = -1

		self.fitness_score = unary * np.abs(np.prod(scores))

		return self.fitness_score
		# return self.score_1, self.score_2, self.score_3


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


	if not os.path.isdir(args.output_dirpath):
		os.mkdir(args.output_dirpath)
	elif args.overwrite:
		print("WARNING: Files in the output directory will be overwriten.")

	prot_u = mda.Universe(args.protein_filepath)
	mol1 = Chem.MolFromMolFile(args.ligand1_filepath)
	mol2 = Chem.MolFromMolFile(args.ligand2_filepath)

	system = System(
		prot_u.select_atoms("not element H").positions,
		get_rdkit_ha_positions(mol1),
		get_rdkit_ha_positions(mol2),
		args.output_dirpath
	)

	# Center protein and ligands according to protein center of geometry
	translation_vector = system.complex.center(method="cog")
	system.ligand_1.translate(translation_vector)
	system.ligand_2.translate(translation_vector)

	# Compute principal axis of the protein
	eigenvalues, eigenvectors = system.complex.eigen()

	# Align the first principal axis onto the diagonal of cube
	first_principal_axis = eigenvectors[:, 0]
	ref_vector = normalize([1, 1, 1])
	diag_align, rmsd = R.align_vectors(ref_vector[np.newaxis,:], first_principal_axis[np.newaxis,:])
	rotation_matrix = diag_align.as_matrix()

	system.complex.rotate(rotation_matrix)
	system.ligand_1.rotate(rotation_matrix)
	system.ligand_2.rotate(rotation_matrix)

	# Run GA to find the best placement for the second ligand
	system.optimize()
	system.ga_instance.plot_fitness()

	# Save files
	prot_u.atoms.translate(translation_vector)
	prot_u.atoms.rotate(diag_align.as_matrix())
	prot_u.atoms.write(os.path.join(args.output_dirpath, "receptor_aln.pdb"))


	translate_mol(mol1, translation_vector)
	rotate_mol(mol1, diag_align.as_matrix())
	with open(os.path.join(args.output_dirpath, "ligand_1.sdf"), 'w') as f:
		f.write(Chem.MolToMolBlock(mol1))

	translate_mol(mol2, translation_vector)
	rotate_mol(mol2, diag_align.as_matrix())
	displacement_vector = system.solution - get_rdkit_ha_positions(mol2).mean(axis=0)
	translate_mol(mol2, displacement_vector)
	with open(os.path.join(args.output_dirpath, "ligand_2.sdf"), 'w') as f:
		f.write(Chem.MolToMolBlock(mol2))

	# Draw some shape using CGO in PyMOL
	_SYSTEM_OLD_CELL_CGO_FORMATTER = cgo.CGOFormatter(
		os.path.join(args.output_dirpath, "original_cell.pml"), 'original_cell', (0.0, 1.0, 0.0))
	cgo.export_box_to_cgo(system.complex.box.lower, system.complex.box.upper, _SYSTEM_OLD_CELL_CGO_FORMATTER)

	_SYSTEM_NEW_CELL_CGO_FORMATTER = cgo.CGOFormatter(
		os.path.join(args.output_dirpath, "new_cell.pml"), 'new_cell', (1.0, 0.0, 0.0))
	cgo.export_box_to_cgo(system.new_box.lower, system.new_box.upper, _SYSTEM_NEW_CELL_CGO_FORMATTER)

	# Save wrapped complex atoms around ligand 2
	wrapped_pos = system.wrapped_complex_positions
	with open(os.path.join(args.output_dirpath, "wrapped.xyz"), 'w') as f:
		f.write(f"{wrapped_pos}\n\n")
		for position in wrapped_pos:
			f.write(f"H\t{position[0]:.4f}\t{position[1]:.4f}\t{position[2]:.4f}\n")

	_SYSTEM_WRAPPED_CELL_CGO_FORMATTER = cgo.CGOFormatter(
		os.path.join(args.output_dirpath, "wrapped_cell.pml"), 'wrapped_cell', (0.0, 0.0, 1.0))
	cgo.export_box_to_cgo(np.min(wrapped_pos, axis=0), np.max(wrapped_pos, axis=0), _SYSTEM_WRAPPED_CELL_CGO_FORMATTER)

	breakpoint()
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
