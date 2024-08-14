import sys
import os
import subprocess
from time import time

from openff.toolkit import Molecule
from openff.toolkit.utils.rdkit_wrapper import UndefinedStereochemistryError
from rdkit import Chem

def clean_antechamber(dirpath):
	try:
		os.remove(os.path.join(dirpath, "ANTECHAMBER_AC.AC"))
		os.remove(os.path.join(dirpath, "ANTECHAMBER_AC.AC0"))
		os.remove(os.path.join(dirpath, "ANTECHAMBER_BOND_TYPE.AC"))
		os.remove(os.path.join(dirpath, "ANTECHAMBER_BOND_TYPE.AC0"))
		os.remove(os.path.join(dirpath, "ANTECHAMBER.FRCMOD"))
		os.remove(os.path.join(dirpath, "ATOMTYPE.INF"))
	except FileNotFoundError as err:
		print("DEV: Something wrong\n{err}")

def main(args):
	if os.path.isdir(args.output_dirpath) and not args.overwrite:
		print(f"ERROR: directory {args.output_dirpath} already exists! Use '-w' to force overwrite.")
		return 1

	if not os.path.isfile(args.ligands_filepath) or not os.access(args.ligands_filepath, os.R_OK):
		print(f"ERROR: Ligands file {args.ligands_filepath} not found or not readable!")
		return 1

	if not os.path.isdir(args.output_dirpath):
		os.mkdir(args.output_dirpath)
	elif args.overwrite:
		print("WARNING: Files in the output directory will be overwriten.")


	suppl = Chem.SDMolSupplier(args.ligands_filepath)
	rdmols = []
	for mol in suppl:
		rdmols.append(mol)

	if len(rdmols) >= 1000:
		print(f"ERROR: Does not support 1000 molecules or more ({len(rdmols)})!")
		return 1

	print(f"INFO: {len(rdmols)} molecule(s) loaded")

	# Go to the output folder to not mess cwd with ANTECHAMBER intermediate output files
	cwd = os.getcwd()
	os.chdir(args.output_dirpath)

	for i, rdmol in enumerate(rdmols, 1):
		try:
			molecule = Molecule.from_rdkit(rdmol)
		except UndefinedStereochemistryError:
			print(f"WARNING: Unspecified stereo center was detected in molecule {i}! It could be a bug from OpenFF. Double check your structure!")
			molecule = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

		prefix = f"{i:0>3d}"

		if not molecule.name:
			print(f"WARNING: Molecule {i} has no name defined in the file! Set default.")
			molecule.name = f"{i:0>3d}"

		mol_start = time()
		print(f"INFO: Treating molecule {molecule.name} {prefix}...")
		print("INFO: Assigning charges...")

		net_charge = molecule.total_charge
		molecule.assign_partial_charges(partial_charge_method="am1bcc")

		input_sdf_filename = prefix + ".sdf"
		gaff_mol2_filename = prefix + ".gaff.mol2"
		frcmod_filename = prefix + ".frcmod"

		molecule.to_file(input_sdf_filename, file_format="sdf")

		# Run ANTECHAMBER
		print("INFO: Running ANTECHAMBER...")

		cmd = f"antechamber -i {input_sdf_filename} -fi mdl -o {gaff_mol2_filename} -fo mol2 -at gaff2"
		output = subprocess.getoutput(cmd)

		if not os.path.isfile(gaff_mol2_filename):
			print(f"WARNING: ANTECHAMBER failed for molecule {molecule.name} in {os.path.basname(input_sdf_filename)}")
			continue

		# Run PARMCHK
		print("INFO: Running PARMCHK...")
		cmd = f"parmchk2 -i {gaff_mol2_filename} -f mol2 -o {frcmod_filename} -s gaff2 -a Y"
		output = subprocess.getoutput(cmd)

		if not os.path.isfile(gaff_mol2_filename):
			print(f"WARNING: PARMCHK failed for molecule {molecule.name} in {os.path.basname(input_sdf_filename)}")
			continue

		print(f"INFO: Molecule {prefix} done in {time() - mol_start:.1f} s")


	# Clean directory
	clean_antechamber(os.getcwd())
	os.chdir(cwd)

	print("DONE")
	return 0


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--ligands", required=True, dest="ligands_filepath")
	parser.add_argument("-o", "--output", required=True, dest="output_dirpath")
	parser.add_argument("-w", "--overwrite", action="store_true")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)
