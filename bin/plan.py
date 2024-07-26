import sys
import os
import json
# import math
# from pathlib import Path
# from typing import Iterable, Callable, Optional, Union
# import itertools
# from collections import Counter
# import functools
# import warnings

# import networkx as nx
# from tqdm.auto import tqdm

# from gufe import SmallMoleculeComponent, AtomMapper
# from openfe.setup import LigandNetwork
# from openfe.setup.atom_mapping import LigandAtomMapping

# from lomap import generate_lomap_network, LomapAtomMapper
# from lomap.dbmol import _find_common_core

# from openff.toolkit import Molecule
# from openfe import SmallMoleculeComponent
# from openfe.setup import LomapAtomMapper
# from kartograf import KartografAtomMapper
import pandas as pd
from openfe.setup.ligand_network_planning import generate_lomap_network
from openfe.setup.ligand_network_planning import generate_minimal_spanning_network
from openfe.setup.ligand_network_planning import generate_maximal_network
from openfe import lomap_scorers
from openfe.utils.atommapping_network_plotting import plot_atommapping_network


def generate_lomap_network_wrapper(ligands, mappers, scorer):
	return generate_lomap_network(molecules=ligands, mappers=mappers, scorer=scorer)


def format_id(index):
	return f"{index:0>3d}"


_ID_WIDTH = 5
_NAME_WIDTH = 5
_SCORE_WIDTH = 8

def main(args):
	_MAPPERS = {
		"lomap": LomapAtomMapper,
		"kartograph": LomapAtomMapper,
	}

	_NETWORKS = {
		"lomap": generate_lomap_network_wrapper,
		"minimal": generate_minimal_spanning_network,
		"maximal": generate_maximal_network,
	}


	if os.path.isdir(args.output_dirpath) and not args.overwrite:
		print(f"ERROR: directory {args.output_dirpath} already exists! Use '-w' to force overwrite.")
		return 1

	if not os.path.isfile(args.ligands_filepath) or not os.access(args.ligands_filepath, os.R_OK):
		print(f"ERROR: Ligands file {args.ligands_filepath} not found or not readable!")
		return 1

	if not os.path.isdir(args.output_dirpath):
		os.mkdir(args.output_dirpath)

	# Load ligands and assign unique id for each ligand (order in the file)
	ligands_sdf = Molecule.from_file(args.ligands_filepath)
	if len(ligands_sdf) >= 1000:
		print(f"ERROR: Does not support 1000 molecules or more ({len(ligands_sdf)})!")
		return 1

	ligand_indices = {} # Link names with ids
	for ligand in enumerate(ligands_sdf, 1):
		if not ligand.name:
			print(f"WARNING: Ligand {i} has no name defined in the file! Set default.")
			ligand.name = format_id(i)

		ligand_indices[ligand] = ligand.name

	# Generate network
	ligand_mols = [SmallMoleculeComponent.from_openff(sdf) for sdf in ligands_sdf]
	mapper = _MAPPERS[args.mapper]()
	network = _NETWORKS[args.network](
		ligands=ligand_mols,
		scorer=lomap_scorers.default_lomap_score,
		mappers=[mapper,]
	)


	# Save figures to visualize network
	network_fig = plot_atommapping_network(network)
	network_fig.savefig(os.path.join(args.output_dirpath, "graph.png", dpi=200, bbox_inches='tight'))

	# Save network and each edge
	data = {
		"idA": [],
		"idB": [],
		"nameA": [],
		"nameB": [],
		"score": [],
	}

	for edge in network.edges:
		id_a = ligand_indices[edge.componentA.name]
		id_b = ligand_indices[edge.componentB.name]

		fmt_id_a = format_id(i)
		fmt_id_b = format_id(i)

		data["idA"].append(fmt_id_a)
		data["idB"].append(fmt_id_b)
		data["nameA"].append(edge.componentA.name)
		data["nameB"].append(edge.componentB.name)
		data["score"].append(f"{edge.score:.3f}")

		# Plot
		map_plot_filepath = os.path.join(args.output_dirpath, f"{fmt_id_a}_{fmt_id_b}.png")
		edge.draw_to_file(map_plot_filepath)

	df = pd.DataFrame(data)
	df.to_string(os.path.join(arg.output_dirpath, "network.csv"), col_space=[_ID_WIDTH, _ID_WIDTH, _NAME_WIDTH, _NAME_WIDTH, _SCORE_WIDTH], index=None)

	print("DONE")

	return 0

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--ligands", required=True, dest="ligands_filepath")
	parser.add_argument("-o", "--output", required=True, dest="output_dirpath")
	parser.add_argument("-w", "--overwrite", action="store_true")
	parser.add_argument("-m", "--mapper", choices=["lomap", "kartograph"], default="lomap")
	parser.add_argument("-n", "--network", choices=["lomap", "minimal", "maximal"], default="lomap")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)
