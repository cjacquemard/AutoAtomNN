import sys
import os
import itertools
from copy import copy
from pathlib import Path

import numba
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.spatial.transform import Rotation as R
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans

import sbs
import cgo
from filehandler import PDB, SDF

_PYMOL_TEMPLATE = """# WARNING! Run this script in the directory where files are!
delete all

load {protein_filename}, receptor

load {ligand_1_filename}, ligand_1
split_states ligand_1, prefix=l1
delete ligand_1
group ligand_1, l1*

load {ligand_2_filename}, ligand_2
split_states ligand_2, prefix=l2
delete ligand_2
group ligand_2, l2*

load {population_filename}, population, discrete=1

hide everything, all
show cartoon, receptor
color gray80, receptor and e. c

show sticks, ligand_1 and not (h. and (e. c extend 1))
color green, ligand_1 and e. c

show sticks, ligand_2  and not (h. and (e. c extend 1))
color magenta, ligand_2 and e. c

show spheres, population
spectrum q, red_white_blue, population, {min_value}, {max_value}
ramp_new colorbar, none, [{min_value}, {mean_value}, {max_value}], [red, white, blue]

run {original_system_box_filename}
run {new_system_box_filename}
run {max_system_box_filename}

center
"""

class CONST:
	INFO_FILENAME = "info.yml"

def normalize(v):
	norm = np.linalg.norm(v)
	if norm == 0: 
	   return v
	return v / norm


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


class ParentSelection:
	def _d_check(func):
		def _check_wrapper(fitness, num_parents):
			if np.any(fitness < 0):
				raise ValueError("Fitness values must be non-negative")

			if num_parents > len(fitness):
				raise ValueError("Number of parents cannot be greater than the number of individuals")

			if num_parents < 0:
				raise ValueError("Number of parents must be positive")

			return func(fitness, num_parents)

		return _check_wrapper

	@_d_check
	@staticmethod
	def maximum(fitness, num_parents):
		return np.argsort(fitness)[:-num_parents-1:-1]

	@_d_check
	@staticmethod
	def roulette_wheel_selection(fitness, num_parents):
		"""
		Selects an individual based on roulette wheel selection.

		Parameters:
		fitness (list or np.array): Array or list of fitness values of the individuals.

		Returns:
		int: The index of the selected individual.
		"""
		fitness = np.array(fitness)

		# Total fitness
		total_fitness = np.sum(fitness)
		if total_fitness <= 0:
			raise ValueError("Sum of fitness must be greater than 0")

		# Calculate the probability of selection for each individual
		probabilities = fitness / total_fitness

		# Select indices
		selected_indices = np.random.choice(fitness.shape[0], size=num_parents, p=probabilities, replace=False)
		
		return selected_indices

	@_d_check
	@staticmethod
	def stochastic_universal_sampling(fitness, num_parents):
		"""
		Selects multiple individuals based on Stochastic Universal Sampling (SUS).

		Parameters:
		fitness (list or np.array): Array or list of fitness values of the individuals.
		num_parents (int): Number of parents to select.

		Returns:
		list: Indices of the selected individuals.
		"""
		fitness = np.array(fitness)

		# Total fitness
		total_fitness = np.sum(fitness)
		
		# Calculate the probability of selection for each individual
		probabilities = fitness / total_fitness
		
		# Cumulative probability distribution
		cumulative_probabilities = np.cumsum(probabilities)
		
		# Determine the spacing between the sampling points
		step = 1.0 / num_parents
		start = np.random.uniform(0, step)
		
		# Initialize list for selected indices
		selected_indices = []
		
		# Generate the sampling points and select indices
		current_point = start
		for _ in range(num_parents):
			# Find the index where the current point falls in the cumulative distribution
			selected_index = np.searchsorted(cumulative_probabilities, current_point)
			selected_indices.append(selected_index)
			# Move to the next point
			current_point += step

		return np.array(selected_indices)


class Box:
	_LOC_VERTICES = np.array([
			[-1, -1, -1],
			[-1, -1,  1],
			[-1,  1, -1],
			[ 1, -1, -1],
			[-1,  1,  1],
			[ 1, -1,  1],
			[ 1,  1, -1],
			[ 1,  1,  1],
	])

	_LOC_FACES = np.array([
			[-1,  0,  0],
			[ 0, -1,  0],
			[ 0,  0, -1],
			[ 1,  0,  0],
			[ 0,  1,  0],
			[ 0,  0,  1],
	])

	_LOC_EDGES = np.array([
			[-1, -1,  0],
			[-1,  0, -1],
			[ 0, -1, -1],
			[ 1,  1,  0],
			[ 1,  0,  1],
			[ 0,  1,  1],
			[-1,  1,  0],
			[-1,  0,  1],
			[ 0, -1,  1],
			[ 1, -1,  0],
			[ 1,  0, -1],
			[ 0,  1, -1],
	])

	_LOCS = np.vstack([
		_LOC_VERTICES,
		_LOC_FACES,
		_LOC_EDGES,
	])

	def __init__(self, bbox):
		self._bbox = bbox
		self._cell_size = None

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

	@property
	def cell_size(self):
		return self._cell_size

	@property
	def n_cells(self):
		if self._cell_size is None:
			raise ValueError("You have to define the size of a cell first!")
		return (self.size / self._cell_size).astype(int)

	@property
	def grid_points(self):
		"""
		Get a grid represented by points. Each point conrespond to the center 
		of a cell
		"""

		X = self.grid_axis(0)
		Y = self.grid_axis(1)
		Z = self.grid_axis(2)

		return np.array(np.meshgrid(X, Y, Z)).T.reshape(-1, 3)

	def increase_size(self, values, method="grid", anchor="center"):
		"""
		grid method increase the size of the box to fit cells of size values
		"""
		if method == "grid":
			# Get number of cells
			self._cell_size = values
			n_cells = self.size / self._cell_size
			target_n_cells = self.n_cells + 1
			increase_vector = (target_n_cells - n_cells) * self.cell_size
		else:
			raise NotImplementedError

		if anchor == "center":
			self._bbox[0] -= increase_vector / 2
			self._bbox[1] += increase_vector / 2

	def grid_axis(self, axis):
		half_cell_size = self._cell_size / 2

		return np.linspace(
			self.lower[axis] + half_cell_size[axis],
			self.upper[axis] - half_cell_size[axis],
			self.n_cells[axis],
			endpoint=True
		)

	@classmethod
	def subdivide_grid_points(cls, grid_points, cell_size):
		"""
		Becareful! The points should be evenly spaced. If not, this function is
		not suitable.
		"""
		new_cell_size = cell_size / 2

		new_grid_points = (grid_points[:, np.newaxis, :] + (Box._LOC_VERTICES[np.newaxis, :, :] * new_cell_size / 2)).reshape(-1, 3)

		return new_grid_points, new_cell_size


class Sphere:
	def __init__(self, center, radius):
		self._center = center
		self._radius = radius

	@property
	def center(self):
		return self._center

	@property
	def radius(self):
		return self._radius


class MolObject:
	def __init__(self, *positions):
		self.positions = []
		self.mol_indices = []

		for i, subset_positions in enumerate(positions):
			self.positions.extend(subset_positions)
			self.mol_indices.extend([i] * len(subset_positions))

		self.positions = np.array(self.positions)
		self.mol_indices = np.array(self.mol_indices)
		self.original_positions = self.positions.copy()

		self._box = None
		self._sphere = None

	@property
	def box(self):
		if self._box is None:
			self._box = Box(np.array([
				np.min(self.positions, axis=0),
				np.max(self.positions, axis=0)
			]))

		return self._box

	@property
	def sphere(self):
		if self._sphere is None:
			center, radius = sbs.Welz.get_bounding_ball(self.positions)
			self._sphere = Sphere(center, radius)

		return self._sphere

	@property
	def cog(self):
		return np.mean(self.positions, axis=0)

	@property
	def original_cog(self):
		return np.mean(self.original_positions, axis=0)

	def mol_positions(self, mol_index):
		return self.positions[self.mol_indices == mol_index]

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
		# TODO: Use decorator to clear these variables
		self._box = None
		self._sphere = None

		self.positions += vector

	def center(self, method="bbox"):
		# TODO: Use decorator to clear these variables
		self._box = None
		self._sphere = None

		if method == "cog":
			vector = -self.cog
		elif method == "bbox":
			vector = -self.box.center
		elif method == "sphere":
			vector = -self.sphere.center
		else:
			raise NotImplementedError

		self.translate(vector)
		return vector

	# def pose(self, position, method="cog"):
	# 	self.center(method)
	# 	self.translate(position)

	def rotate(self, rotation_matrix):
		# TODO: Use decorator to clear these variables
		self._box = None
		self._sphere = None

		self.positions = np.dot(self.positions, rotation_matrix)


class SolutionResult:
	def __init__(self):
		pass


class GAFunc:
	@staticmethod
	def do_start(system, ga):
		ga.fitness = None
		ga.parents = None
		ga.parent_indices = None
		ga.pop_island_labels = None

		ga.elit_indices = None
		ga.elits = None
		ga.elit_fitness = None

		ga.parthenogenesis = None
		ga.parthenogenesis_indices = None

		ga.children = None

	@staticmethod
	def do_scoring(system, ga, solution):
		result = SolutionResult()
		result.solution = solution

		result.solution_positions = system.ligand_2_centered_positions + solution

		# Compute the new system box based on the guessed ligand 2 position
		result.solution_ligand_2_box = Box(np.array([
			np.min(result.solution_positions, axis=0),
			np.max(result.solution_positions, axis=0),
		]))

		result.new_box = Box(np.array([
			np.min([system.complex.box.lower, result.solution_ligand_2_box.lower], axis=0) - system._SOLVENT_THICKNESS,
			np.max([system.complex.box.upper, result.solution_ligand_2_box.upper], axis=0) + system._SOLVENT_THICKNESS,
		]))


		result.system_increase_factor = (result.new_box.volume - system.complex_solvated_box.volume) / system.complex_solvated_box.volume

		# Wrapped complex atoms around ligand 2
		result.wrapped_complex_positions = wrap(
			system.complex.positions,
			result.new_box.bbox,
			solution,
		)

		# Keep wrapped complex atoms close to ligand 2 solution
		result.wrapped_bound_checked = (
			  (result.wrapped_complex_positions > result.solution_ligand_2_box.lower - system._MIN_DIST)
			& (result.wrapped_complex_positions < result.solution_ligand_2_box.upper + system._MIN_DIST)
		).all(axis=1)

		result.close_wrapped_positions = result.wrapped_complex_positions[result.wrapped_bound_checked]

		if len(result.close_wrapped_positions) == 0:
			# Find the closest point
			result.score_1 = 100
		else:
			result.min_distance_to_complex = cdist(result.close_wrapped_positions, result.solution_positions).min()	
			if result.min_distance_to_complex > system._MIN_DIST:
				result.score_1 = 100
			else:
				result.score_1 = 0

		# Compute distance to ligand 1
		result.min_distance_to_ligand_1 = cdist(system.ligand_1.positions, result.solution_positions).min()

		# Compute scores
		if result.system_increase_factor <= 0:
			result.score_2 = 100
		else:
			result.score_2 = 100 * np.exp(-2 * result.system_increase_factor)

		if result.min_distance_to_ligand_1 < system._MIN_DIST:
			result.score_3 = 100 * (1 - np.exp(2 * (-result.min_distance_to_ligand_1 + system._MIN_DIST)))
		else:
			result.score_3 = 100 * np.exp(0.1 * (-result.min_distance_to_ligand_1 + system._MIN_DIST))

		result.scores = np.array([
			result.score_1, result.score_2, result.score_3
		])


		result.fitness_score = result.scores.prod() / 10000

		return result

	@staticmethod
	def do_fitness(system, ga):
		fitness = []
		for individual in ga.population:
			result = ga.scoring_func(system, ga, individual)
			fitness.append(result.fitness_score)

		ga.fitness = np.array(fitness)

		# Keep trace of the best fitness for this generation
		ga.best_generation_fitness.append(ga.fitness.max())

	@staticmethod
	def do_selection(system, ga, func=ParentSelection.maximum):
		island_parent_sizes = np.array([ga.n_parents // ga.n_islands + (i < ga.n_parents % ga.n_islands) for i in range(ga.n_islands)])

		if ga.n_islands == 1:
			pop_island_labels = np.zeros(len(ga.population), dtype=int)
		else:
			clustering = KMeans(n_clusters=ga.n_islands).fit(ga.population)
			pop_island_labels = clustering.labels_

		parents = []
		parent_indices = []
		parent_island_labels = []

		for island_id in range(ga.n_islands):
			island_parent_size = island_parent_sizes[island_id]

			island_pop_indices = np.where(pop_island_labels == island_id)[0]
			
			# Check if there are enough individuals in cluster
			if island_pop_indices.shape[0] < ga.n_min_parents_island:
				# Some clusters may be very similar.
				# Reduce the number of clusters and redo clustering
				print("WARNING: Decreasing the number of islands...")
				ga.n_islands -= 1

				return GAFunc.do_selection(system, ga, func)

			island_population = ga.population[island_pop_indices]
			island_fitness = ga.fitness[island_pop_indices]

			# Select parents
			island_parent_indices = func(island_fitness, island_parent_size)
			island_parents = island_population[island_parent_indices]

			parents.extend(island_parents)
			parent_indices.extend(island_pop_indices[island_parent_indices])

		ga.parents = np.array(parents)
		ga.parent_indices = np.array(parent_indices)
		ga.pop_island_labels = pop_island_labels

	@staticmethod
	def do_elitism(system, ga, func=ParentSelection.maximum):
		parent_fitness = ga.fitness[ga.parent_indices]
		parent_elit_indices = func(parent_fitness, ga.n_elits)
		ga.elit_indices = ga.parent_indices[parent_elit_indices]
		ga.elits = ga.population[ga.elit_indices]

	@staticmethod
	def do_parthenogenesis(system, ga, func=ParentSelection.maximum):
		parent_fitness = ga.fitness[ga.parent_indices]
		parent_parthenogenesis_indices = func(parent_fitness, ga.n_parthenogenesis)
		ga.parthenogenesis_indices = ga.parent_indices[parent_parthenogenesis_indices]

		parthenogenesis = []
		# Local search
		for parent_index in ga.parthenogenesis_indices:
			parent = ga.population[parent_index]
			parent_score = ga.fitness[parent_index]

			# TODO: Do something about the 1.5 distance (e.g. constant)
			# Create new points around parent 
			new_points = (parent[np.newaxis,np.newaxis,:] + Box._LOCS[np.newaxis, :, :] * 1.5).reshape(-1, 3)
			best_point = parent
			best_score = parent_score

			# NOTE: Keep parent if no better solution is found
			for new_point in new_points:
				result = ga.scoring_func(system, ga, new_point)
				score = result.fitness_score
				if score > best_score:
					best_point = new_point
					best_score = score
					# Do not check other positions if a better one is found
					break

			# NOTE: The parent is mutated if it is the best point
			# if better_found:
			# 	best_point += np.random.normal(scale=3, size=3)

			parthenogenesis.append(best_point)

		ga.parthenogenesis = np.array(parthenogenesis)


	@staticmethod
	def do_mating(system, ga):
		parent_distances = pdist(ga.parents)
		N = ga.n_parents
		children = []

		# Parents mating
		# TODO: I have to rework the mating procedure
		while len(children) < ga.n_children:
			# TODO: Choose randomly two parents
			for i, j in itertools.combinations(range(ga.n_parents), 2):
				parent_index_1 = ga.parent_indices[i]
				parent_index_2 = ga.parent_indices[j]

				parent_1 = ga.population[i]
				parent_2 = ga.population[j]

				k = int((N*(N-1)/2) - (N-i)*((N-i)-1)/2 + j - i - 1)

				# TODO: Optimize the following block
				if ga.pop_island_labels[i] == ga.pop_island_labels[j]:

					# Do not mate too close parents (no incest allowed)
					if parent_distances[k] > 1.5:
						distance = parent_distances[k]
						barycenter = (parent_1 + parent_2) / 2
						children.append(barycenter + np.random.normal(scale=2, size=3))

						if len(children) == ga.n_children:
							break


		if ga.n_children - len(children) > 0:
			raise RuntimeError("Something wrong happens during mating procedure")

		ga.children = np.array(children)	

	@staticmethod
	def do_mutation(system, ga):
		ga.mutated_children = ga.children.copy()

	@staticmethod
	def do_generation(system, ga):
		ga.previous_population = ga.population.copy()
		ga.population = np.vstack([
			ga.elits,
			ga.parthenogenesis,
			ga.mutated_children,
		])

	@staticmethod
	def on_parents(system, ga):
		# Save population
		PDB.save_population(
			system.population_filepath,
			ga.population,
			np.arange(len(ga.population)).astype(int),
			ga.fitness,
			ga.pop_island_labels
		)
		
		# Save selected parents
		PDB.save_population(
			system.population_filepath,
			ga.parents,
			ga.parent_indices,
			ga.fitness[ga.parent_indices],
			ga.pop_island_labels[ga.parent_indices]
		)


class GA:
	"""
	Based on PyGAD module.
	"""
	_DEFAULTS = {
		"n_generations": 20,
		"n_parents": 20,
		"n_islands": 5,
		"n_min_parents_island": 4,
		"n_elits": 0,
		"n_parthenogenesis": 20,

		"start_func": GAFunc.do_start,
		"scoring_func": GAFunc.do_scoring,
		"fitness_func": GAFunc.do_fitness,
		"selection_func": GAFunc.do_selection,
		"elitism_func": GAFunc.do_elitism,
		"parthenogenesis_func": GAFunc.do_parthenogenesis,
		"mating_func": GAFunc.do_mating,
		"mutation_func": GAFunc.do_mutation,
		"generation_func": GAFunc.do_generation,

		"on_parents": GAFunc.on_parents,
	}

	def __init__(self, population, system, **kwargs):
		self.system = system
		self.population = np.array(population)
		self.fitness = np.full(len(self.population), -np.inf)
		self.best_generation_fitness = []
		self.is_done = False

		opts = {}
		for key in self._DEFAULTS:
			opts[key] = kwargs.get(key, self._DEFAULTS[key])

		# Check opts
		if opts["n_generations"] < 1:
			raise ValueError("n_generations must be greater than 0")

		if opts["n_parents"] < 1:
			raise ValueError("n_parents must be greater than 0")

		if opts["n_parents"] > len(population):
			raise ValueError("n_parents cannot be greater than the population")

		if opts["n_islands"] < 1:
			raise ValueError("n_islands must be greater than 0")

		if opts["n_elits"] < 0:
			raise ValueError("n_elits must be 0 or a positive integer")			

		if opts["n_parthenogenesis"] < 0:
			raise ValueError("n_parthenogenesis must be 0 or a positive integer")	

		if opts["n_parthenogenesis"] > opts["n_parents"]:
			raise ValueError("n_parthenogenesis cannot be greater than n_parents ")

		if opts["n_parents"] < opts["n_min_parents_island"]:
			raise ValueError("The number of selected parents cannot be lower than the minimal number of parents per islands")

		# Create instance attributes
		for key in opts:
			if key in self.__dict__:
				raise KeyError(f"Attribute '{key}' already exits")

			self.__dict__[key] = opts[key]

		# Deduced attributes
		self.n_pop = len(population)
		self.n_children = self.n_pop - self.n_elits - self.n_parthenogenesis

		# Check
		if self.n_children < 0:
			raise ValueError("The number of children must be 0 or greater. Check the n_elits and n_parthenogenesis attribute values")


	def run(self):
		for generation_id in range(self.n_generations):
			self.current_generation = generation_id
			self.start_func(self.system, self)
			self.fitness_func(self.system, self)
			self.selection_func(self.system, self)
			self.on_parents(self.system, self)
			self.elitism_func(self.system, self)
			self.parthenogenesis_func(self.system, self)
			self.mating_func(self.system, self)
			self.mutation_func(self.system, self)
			self.generation_func(self.system, self)

		# Compute fitness on last generation
		self.fitness_func(self.system, self)

		self.is_done = True

	def best_solution(self):
		if not self.is_done:
			raise RuntimeError("The GA was not executed")

		best_individual_index = np.argmax(self.fitness)
		return self.scoring_func(self.system, self, self.population[best_individual_index])

	def plot_fitness(self):
		if not self.is_done:
			raise RuntimeError("The GA was not executed")

		fig, ax = plt.subplots(1, 1)
		ax.xaxis.set_major_locator(MaxNLocator(integer=True))
		ax.set_xlabel("generation")
		ax.set_ylabel("fitness")

		ax.plot(range(1, self.n_generations+2), self.best_generation_fitness)

		return fig, ax


class System:
	_N_INDIVIDUALS = 200
	_MIN_DIST = 20.0
	_SOLVENT_THICKNESS = 10
	_INIT_CELL_SIZE = 15 # TODO: Use sphere radius of ligand 2

	# FILES
	_POPULATION_FILENAME = "populations.pdb"
	_PROTEIN_FILENAME = "receptor_aln.pdb"
	_LIGAND_1_FILENAME = "ligands_1.sdf"
	_LIGAND_2_FILENAME = "ligands_2.sdf"

	_ORIGINAL_SYSTEM_BOX_FILENAME = "original_system_box.pml"
	_NEW_SYSTEM_BOX_FILENAME = "new_system_box.pml"
	_MAX_SYSTEM_BOX_FILENAME = "max_system_box.pml"

	_PYMOL_SCRIPT_FILENAME = "visualize.pml"

	def __init__(self, protein_atom_positions, ligand_1_atom_positions, ligand_2_atom_positions, output_dirpath):
		self.complex = MolObject(protein_atom_positions, ligand_1_atom_positions)
		self.ligand_1 = MolObject(ligand_1_atom_positions)
		self.ligand_2 = MolObject(ligand_2_atom_positions)

		# Directory and file paths
		self.output_dirpath = output_dirpath

		self.protein_filepath = Path(output_dirpath, self._PROTEIN_FILENAME)
		self.ligand_1_filepath = Path(output_dirpath, self._LIGAND_1_FILENAME)
		self.ligand_2_filepath = Path(output_dirpath, self._LIGAND_2_FILENAME)

		self.population_filepath = Path(output_dirpath, self._POPULATION_FILENAME)

		self.original_system_box_filepath = Path(output_dirpath, self._ORIGINAL_SYSTEM_BOX_FILENAME)
		self.new_system_box_filepath = Path(output_dirpath, self._NEW_SYSTEM_BOX_FILENAME)
		self.max_system_box_filepath = Path(output_dirpath, self._MAX_SYSTEM_BOX_FILENAME)

		self.pymol_script_filepath = Path(output_dirpath, self._PYMOL_SCRIPT_FILENAME)

		self.output_filepaths = [
			self.protein_filepath,
			self.ligand_1_filepath,
			self.ligand_2_filepath,
			self.population_filepath,
			self.original_system_box_filepath,
			self.new_system_box_filepath,
			self.max_system_box_filepath,
			self.pymol_script_filepath,
		]

	def clean_output(self):
		for output_filepath in self.output_filepaths:
			if os.path.isfile(output_filepath):
				os.remove(output_filepath)

	@property
	def max_box(self):
		"""
		Compute the size of the system in the worst case.
		Worst case corresponds to the ligand 2 laying on the diagonal of the 
		box with a distance _MIN_DIST away from any atom of the complex.
		"""
		# min_diagonal_distance = self._MIN_DIST / np.sqrt(3)
		lower = self.complex.box.lower - self._MIN_DIST - self.ligand_2.box.size
		upper = self.complex.box.upper + self._MIN_DIST + self.ligand_2.box.size

		max_box = Box(np.array([lower, upper]))

		return max_box

	def start_population(self, n_subdivide=1):
		"""Evenly place ligand 2 poses in the system"""

		# TODO: This algo does not work well. Rewrite it (with Numba)!
		self.grid_box = self.max_box
		self.grid_box.increase_size(np.full(3, self._INIT_CELL_SIZE))

		init_positions = self.grid_box.grid_points
		PDB.save_population(self.population_filepath, init_positions)

		kdtree = KDTree(self.complex.positions)

		# Remove initial position of ligand 2 too close from complex
		distances, _ = kdtree.query(init_positions)
		distance_checked = distances > self._MIN_DIST + self.ligand_2.sphere.radius

		good_points = init_positions[distance_checked]
		good_point_distances = distances[distance_checked]
		bad_points = init_positions[~distance_checked]

		# Subdivide grid
		cell_size = self.grid_box.cell_size
		for n in range(n_subdivide):
			new_points, cell_size = Box.subdivide_grid_points(bad_points, cell_size)
			distances, indices = kdtree.query(new_points)
			distance_checked = distances > self._MIN_DIST + self.ligand_2.sphere.radius

			# Update good and bad points
			good_points = np.vstack([good_points, new_points[distance_checked]])
			good_point_distances = np.hstack([good_point_distances, distances[distance_checked]])
			bad_points = new_points[~distance_checked]


		# Save good solutions
		PDB.save_population(self.population_filepath, good_points)

		# Keep only the best solutions (closest to the complex)
		best_indices = np.argsort(good_point_distances)
		self.init_pop_positions = good_points[best_indices[:self._N_INDIVIDUALS]]
		PDB.save_population(self.population_filepath, self.init_pop_positions)	

		return self.init_pop_positions

	def prepare(self):
		"""
		Center and align complex
		"""
		# Remove old data from the output dir
		self.clean_output()

		# Center protein and ligands according to protein center of geometry
		self.translation_vector = self.complex.center(method="cog")
		self.ligand_1.translate(self.translation_vector)
		self.ligand_2.translate(self.translation_vector)

		# Compute principal axis of the protein
		eigenvalues, eigenvectors = self.complex.eigen()

		# Align the first principal axis onto the diagonal of cube
		first_principal_axis = eigenvectors[:, 0]
		ref_vector = normalize([1, 1, 1])
		diag_align, rmsd = R.align_vectors(ref_vector[np.newaxis,:], first_principal_axis[np.newaxis,:])
		self.rotation_matrix = diag_align.as_matrix().T

		self.complex.rotate(self.rotation_matrix)
		self.ligand_1.rotate(self.rotation_matrix)
		self.ligand_2.rotate(self.rotation_matrix)

		# Keep trace of the position of ligand 2 and center it
		self.ligand_2_center = self.ligand_2.sphere.center
		self.ligand_2_centered_positions = self.ligand_2.positions - self.ligand_2_center

		# Compute the current complex bounding box
		self.complex_solvated_box = Box(np.array([
			self.complex.box.lower - self._SOLVENT_THICKNESS,
			self.complex.box.upper + self._SOLVENT_THICKNESS,
		]))

	def save(self, results):
		# Draw some shape using CGO for PyMOL
		# Original system box (complex of protein and ligand 1)
		cgo.export_box_to_cgo(
			self.complex_solvated_box.lower, self.complex_solvated_box.upper,
			cgo.CGOFormatter(
				self.original_system_box_filepath,
				self.original_system_box_filepath.stem, (0.0, 1.0, 0.0)
			)
		)

		# New system box (with the complex + ligand 2)
		cgo.export_box_to_cgo(
			results.new_box.lower, results.new_box.upper,
			cgo.CGOFormatter(
				self.new_system_box_filepath,
				self.new_system_box_filepath.stem, (1.0, 0.0, 0.0)
			)
		)

		# Maximal guessed system box
		cgo.export_box_to_cgo(
			self.grid_box.lower, self.grid_box.upper,
			cgo.CGOFormatter(
				self.max_system_box_filepath,
				self.max_system_box_filepath.stem, (1.0, 1.0, 1.0)
			)
		)

		# Save pymol script
		with open(self.pymol_script_filepath, 'w') as f:
			f.write(_PYMOL_TEMPLATE.format(
				protein_filename=self.protein_filepath.name,
				ligand_1_filename=self.ligand_1_filepath.name,
				ligand_2_filename=self.ligand_2_filepath.name,
				population_filename=self.population_filepath.name,
				original_system_box_filename=self.original_system_box_filepath.name,
				new_system_box_filename=self.new_system_box_filepath.name,
				max_system_box_filename=self.max_system_box_filepath.name,
				min_value=0,
				mean_value=50,
				max_value=100,
			))


def main(args):
	if args.ligand_2_filepath is None:
		args.ligand_2_filepath = args.ligand_1_filepath
		same_ligand = True
	else:
		same_ligand = False

	if os.path.isdir(args.output_dirpath) and not args.overwrite:
		print(f"ERROR: directory {args.output_dirpath} already exists! Use '-w' to force overwrite.")
		return 1

	if not os.path.isfile(args.ligand_1_filepath) or not os.access(args.ligand_1_filepath, os.R_OK):
		print(f"ERROR: Ligand 1 file {args.ligand1_filepath} not found or not readable!")
		return 1

	if not os.path.isfile(args.ligand_2_filepath) or not os.access(args.ligand_2_filepath, os.R_OK):
		print(f"ERROR: Ligand 2 file {args.ligand2_filepath} not found or not readable!")
		return 1

	if not os.path.isfile(args.protein_filepath) or not os.access(args.protein_filepath, os.R_OK):
		print(f"ERROR: Protein file {args.protein_filepath} not found or not readable!")
		return 1

	if not os.path.isdir(args.output_dirpath):
		os.mkdir(args.output_dirpath)
	elif args.overwrite:
		print(f"WARNING: Files in the output directory '{args.output_dirpath}' will be overwritten.")
	else:
		print(f"CRITICAL: The output directory '{args.output_dirpath}' already exists. Use '-w' to force overwrite.")
		return 1

	protein_file = PDB(args.protein_filepath)
	protein_file.parse()
	heavy_atom_protein_positions = protein_file.positions[protein_file.elements != 'H']

	ligand_1_file = SDF(args.ligand_1_filepath)
	ligand_1_file.parse()
	heavy_atom_ligand_1_positions = ligand_1_file.positions[ligand_1_file.elements != 'H']

	if same_ligand:
		heavy_atom_ligand_2_positions = heavy_atom_ligand_1_positions.copy()
	else:
		ligand_2_file = SDF(args.ligand_2_filepath)
		ligand_2_file.parse()
		heavy_atom_ligand_2_positions = ligand_2_file.positions[ligand_2_file.elements != 'H']		

	system = System(
		heavy_atom_protein_positions,
		heavy_atom_ligand_1_positions,
		heavy_atom_ligand_2_positions,
		args.output_dirpath
	)

	system.prepare()

	# Save protein and ligand 1 new positions
	protein_new_positions = protein_file.positions
	protein_new_positions += system.translation_vector
	protein_new_positions = np.dot(protein_new_positions, system.rotation_matrix)
	protein_file.set_positions(protein_new_positions)
	protein_file.rewrite(system.protein_filepath)

	ligand_1_new_positions = ligand_1_file.positions
	ligand_1_new_positions += system.translation_vector
	ligand_1_new_positions = np.dot(ligand_1_new_positions, system.rotation_matrix)
	ligand_1_file.set_positions(ligand_1_new_positions)
	ligand_1_file.rewrite(system.ligand_1_filepath)

	# Run GA to find the best placement for the ligand 2
	ga = GA(system.start_population(), system)
	ga.run()
	best_results = ga.best_solution()
	displacement_vector = best_results.solution - system.ligand_2_center

	# Plot GA results
	fig, ax = ga.plot_fitness()
	fig.savefig(os.path.join(args.output_dirpath, "fitness.png"), dpi=200, bbox_inches='tight')

	# Save the top solution for ligand 2
	# TODO: Save more solutions
	ligand_2_new_positions = ligand_2_file.positions
	ligand_2_new_positions += system.translation_vector
	ligand_2_new_positions = np.dot(ligand_2_new_positions, system.rotation_matrix)
	ligand_2_new_positions += displacement_vector
	ligand_2_file.set_positions(ligand_2_new_positions)
	ligand_2_file.rewrite(system.ligand_2_filepath)

	system.save(best_results)

	# Save information that will be use later to setup the systems
	data = {
		"displacement_vector": [float(x) for x in system.translation_vector] # Need this otherwise number type is numpy.float and pyaml does not like this
	}
	info_filepath = os.path.join(args.output_dirpath, CONST.INFO_FILENAME)
	with open(info_filepath, 'w') as f:
		breakpoint()
		f.write(yaml.dump(data, default_flow_style=False))

	return 0


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-l1", "--ligand1", required=True, dest="ligand_1_filepath")
	parser.add_argument("-l2", "--ligand2", required=False, dest="ligand_2_filepath")
	parser.add_argument("-p", "--protein", required=True, dest="protein_filepath")
	parser.add_argument("-o", "--output", required=True, dest="output_dirpath")
	parser.add_argument("-w", "--overwrite", action="store_true")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)
