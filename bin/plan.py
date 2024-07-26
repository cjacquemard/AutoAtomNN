import sys
import os
import math
from pathlib import Path
from typing import Iterable, Callable, Optional, Union
import itertools
from collections import Counter
import functools
import warnings

import networkx as nx
from tqdm.auto import tqdm

from gufe import SmallMoleculeComponent, AtomMapper
from openfe.setup import LigandNetwork
from openfe.setup.atom_mapping import LigandAtomMapping

from lomap import generate_lomap_network, LomapAtomMapper
from lomap.dbmol import _find_common_core

from openff.toolkit import Molecule
from openfe import SmallMoleculeComponent
from openfe.setup import LomapAtomMapper
from kartograf import KartografAtomMapper
from openfe.setup.ligand_network_planning import generate_lomap_network
#from openfe.setup.ligand_network_planning import generate_minimal_spanning_network
#from openfe.setup.ligand_network_planning import generate_maximal_network
from openfe import lomap_scorers


def generate_maximal_network(
	ligands: Iterable[SmallMoleculeComponent],
	mapper: AtomMapper,
	scorer: Optional[Callable[[LigandAtomMapping], float]] = None,
) -> LigandNetwork:
	"""
	This function is adapted from the OpenFE package because it does not work!!!!!!
	"""

	nodes = list(ligands)

	mapping_generator = itertools.chain.from_iterable(
		mapper.suggest_mappings(molA, molB)
		for molA, molB in itertools.combinations(nodes, 2)
	)
	if scorer:
		mappings = [mapping.with_annotations({'score': scorer(mapping)})
					for mapping in mapping_generator]
	else:
		mappings = list(mapping_generator)

	network = LigandNetwork(mappings, nodes=nodes)

	return network


def generate_minimal_spanning_network(
    ligands: Iterable[SmallMoleculeComponent],
	mapper: AtomMapper,
    scorer: Callable[[LigandAtomMapping], float],
) -> LigandNetwork:
    """
	This function is adapted from the OpenFE package because it does not work!!!!!!
    """

    # First create a network with all the proposed mappings (scored)
    network = generate_maximal_network(ligands, mapper, scorer)

    # Flip network scores so we can use minimal algorithm
    g2 = nx.MultiGraph()
    for e1, e2, d in network.graph.edges(data=True):
        g2.add_edge(e1, e2, weight=-d['score'], object=d['object'])

    # Next analyze that network to create minimal spanning network. Because
    # we carry the original (directed) LigandAtomMapping, we don't lose
    # direction information when converting to an undirected graph.
    min_edges = nx.minimum_spanning_edges(g2)
    min_mappings = [edge_data['object'] for _, _, _, edge_data in min_edges]
    min_network = LigandNetwork(min_mappings)
    missing_nodes = set(network.nodes) - set(min_network.nodes)
    if missing_nodes:
        raise RuntimeError("Unable to create edges to some nodes: "
                           f"{list(missing_nodes)}")

    return min_network


def generate_lomap_network_wrapper(ligands, mapper, scorer):
	return generate_lomap_network(molecules=ligands, mappers=[mapper,], scorer=scorer)


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


	ligands_sdf = Molecule.from_file(args.ligands_filepath)
	ligand_mols = [SmallMoleculeComponent.from_openff(sdf) for sdf in ligands_sdf]

	breakpoint()

	mapper = _MAPPERS[args.mapper]
	network = _NETWORKS[args.network](
		ligands=ligand_mols,
		scorer=lomap_scorers.default_lomap_score,
		mapper=mapper
	)

	breakpoint()

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
