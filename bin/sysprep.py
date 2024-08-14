import sys
import os

from rdkit import Chem

from filehandler import PDB

_DEAFULTS = {
	"binding_site_threshold": 6.5,
}


class MolobjectInfo:
	def __init__(self, filepath):
		self.filepath = filepath

	def load(self):
		raise NotImplementedError


class LigandInfo(MolobjectInfo):
	def __init__(self, filepath):
		super().__init__(filepath)


class ReceptorInfo(MolobjectInfo):
	def __init__(self, filepath):
		super().__init__(filepath)

	def load(self):


def main(args):
	if not os.path.isfile(args.ligands_dirpath) or not os.access(args.ligands_dirpath, os.R_OK):
		print(f"ERROR: Ligands directory {args.ligands_dirpath} not found or not readable!")
		return 1

	if not os.path.isfile(args.receptor_filepath) or not os.access(args.receptor_filepath, os.R_OK):
		print(f"ERROR: receptor file {args.receptor_filepath} not found or not readable!")
		return 1

	if not os.path.isfile(args.network_filepath) or not os.access(args.network_filepath, os.R_OK):
		print(f"ERROR: network file {args.network_filepath} not found or not readable!")
		return 1

	# if not os.path.isfile(args.info_filepath) or not os.access(args.info_filepath, os.R_OK):
	# 	print(f"ERROR: info file {args.info_filepath} not found or not readable!")
	# 	return 1

	# Definition of the binding site
	receptor_info = ReceptorInfo(args.receptor_filepath)


if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument("-l", "--ligands", required=True, dest="ligands_dirpath")
	parser.add_argument("-r", "--receptor", required=True, dest="receptor_filepath")
	parser.add_argument("-n", "--network", required=True, dest="network_filepath")
	parser.add_argument("-i", "--info", required=True, dest="info_filepath")

	parser.set_defaults(func=main)

	args = parser.parse_args()

	# Run
	status = args.func(args)
	sys.exit(status)