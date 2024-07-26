import sys
import os

def main(args):
	from openff.toolkit import Molecule
	from openfe import SmallMoleculeComponent
	from openfe.setup import LomapAtomMapper
	from kartograf import KartografAtomMapper
	from openfe.setup.ligand_network_planning import generate_lomap_network
	from openfe.setup.ligand_network_planning import generate_minimal_spanning_network
	from openfe.setup.ligand_network_planning import generate_maximal_network
	from openfe import lomap_scorers

	_MAPPERS = {
		"lomap": LomapAtomMapper,
		"kartograph": LomapAtomMapper,
	}

	_NETWORKS = {
		"lomap": generate_lomap_network,
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
		mappers=[mapper,]
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
