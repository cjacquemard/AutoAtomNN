import sys
import os

def main(args):
	import subprocess
	from openff.toolkit import Molecule

	if os.path.isdir(args.output_dirpath) and not args.overwrite:
		print(f"ERROR: directory {args.output_dirpath} already exists! Use '-w' to force overwrite.")
		return 1

	if not os.path.isfile(args.ligands_filepath) or not os.access(args.ligands_filepath, os.R_OK):
		print(f"ERROR: Ligands file {args.ligands_filepath} not found or not readable!")
		return 1

	if not os.path.isdir(args.output_dirpath):
		os.mkdir(args.output_dirpath)

	molecules = Molecule.from_file(args.ligands_filepath)

	for i, molecule in enumerate(molecules, 1):

		net_charge = molecule.total_charge
		molecule.assign_partial_charges(partial_charge_method="am1bcc")

		prefix = f"{i:0>3d}"

		input_sdf_filename = os.path.join(args.output_dirpath, prefix + ".sdf")
		gaff_mol2_filename = os.path.join(args.output_dirpath, prefix + ".gaff.mol2")
		frcmod_filename = os.path.join(args.output_dirpath, prefix + ".frcmod")

		molecule.to_file(input_sdf_filename, file_format="sdf")

		# Run ANTECHAMBER
		cmd = f"antechamber -i {input_sdf_filename} -fi mdl -o {gaff_mol2_filename} -fo mol2 -at gaff2"
		output = subprocess.getoutput(cmd)

		if not os.path.isfile(gaff_mol2_filename):
			print(f"WARNING: ANTECHAMBER failed for molecule {molecule.name} in {os.path.basname(input_sdf_filename)}")
			continue

		# Run PARMCHK
		cmd = f"parmchk2 -i {gaff_mol2_filename} -f mol2 -o {frcmod_filename} -s gaff2 -a Y"
		output = subprocess.getoutput(cmd)

		if not os.path.isfile(gaff_mol2_filename):
			print(f"WARNING: PARMCHK failed for molecule {molecule.name} in {os.path.basname(input_sdf_filename)}")
			continue

	print("DONE")
	return 0


if __name__ == "__main__":
	import argparse

	parser = ArgumentParser()
	parser.add_argument("-l", "--ligands", required=True, dest="ligands_filepath")
	parser.add_argument("-o", "--output", required=True, dest="output_dirpath")
	parser.add_argument("-w", "--overwrite", action="store_true")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)