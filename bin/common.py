import sys
import os
import warnings

import numpy as np
from rdkit.Chem import AllChem
from openff.toolkit import Molecule
from openff.toolkit.utils.rdkit_wrapper import UndefinedStereochemistryError

class LigandError(Exception):
	"""Exception raised for custom error scenarios.

	Attributes:
		message -- explanation of the error
	"""

	def __init__(self, message):
		self.message = message
		super().__init__(self.message)


class RdkitHelper:
	@staticmethod
	def props_to_array(rdkit_objects, prop_name, ignore_missing=True, filtering=True):
		# You can pass GetBonds() or GetAtoms() directly
		rdkit_objects = [x for x in rdkit_objects]

		indices = []
		values = []
		annotated_rdkit_objects = []
		for ro in rdkit_objects:
			d = ro.GetPropsAsDict()

			if prop_name not in d:
				if not ignore_missing:
					raise ValueError("Prop name does not exist")
				else:
					if filtering:
						continue
					else:
						indices.append(ro.GetIdx())
						values.append(np.nan)
						annotated_rdkit_objects.append(ro)						
			else:
				indices.append(ro.GetIdx())
				values.append(d[prop_name])
				annotated_rdkit_objects.append(ro)

		return np.array(indices), np.array(values), np.array(annotated_rdkit_objects)		

	@staticmethod
	def indices_to_rdbonds(rdmol, indices):
		rdbonds = []
		for index in indices:
			if index >= rmdol.GetNumBonds():
				raise IndexError("Index {index} out of range for rdbond")

			rdbonds.append(rdmol.GetBondWithIdx(index))

		return rdbonds

	



class LigandLoader:
	_MAX_MOLS = 1000

	def __init__(self, ligands_filepath):
		if not os.path.isfile(ligands_filepath):
			raise FileNotFoundError(f"Ligands file not found {ligands_filepath}")

		self.ligands_filepath = ligands_filepath
		self._load()
		self._openff_mols()

	def _load(self):
		suppl = AllChem.SDMolSupplier(self.ligands_filepath, sanitize=False, removeHs=False, strictParsing=False)
		self.raw_rdmols = []
		self.clean_rdmols = []
		for mol_id, raw_mol in enumerate(suppl):
			if raw_mol.GetNumConformers() != 1:
				# TODO: Treate case when mol has 2 or more conformers
				raise LigandError(f"Molecule n°{mol_id} must have 1 conformer only")

			clean_mol = AllChem.Mol(raw_mol) # Copy
			AllChem.SanitizeMol(clean_mol) # Inplace

			# Check if structure changed
			if raw_mol.GetNumAtoms() != clean_mol.GetNumAtoms():
				raise LigandError("Molecule changed after sanitization")

			self.raw_rdmols.append(raw_mol)
			self.clean_rdmols.append(clean_mol)

		if len(self.raw_rdmols) >= self._MAX_MOLS:
			raise LigandError(f"Does not support {self._MAX_MOLS} molecules or more ({len(self.raw_rdmols)})!")

		if len(self.raw_rdmols) != len(self.clean_rdmols):
			raise LigandError("Different number of raw rdmols and clean rdmols")

	def _openff_mols(self):
		self.raw_openff_mols = []
		self.clean_openff_mols = []

		for raw_rdmol, clean_rdmol in zip(self.raw_rdmols, self.clean_rdmols):
			self.raw_openff_mols.append(self.rdkit_to_openff(raw_rdmol))
			self.clean_openff_mols.append(self.rdkit_to_openff(clean_rdmol))

	@classmethod
	def rdkit_to_openff(cls, rdmol):
		try:
			openff_mol = Molecule.from_rdkit(rdmol)
		except UndefinedStereochemistryError:
			warnings.warn(f"Unspecified stereo center was detected in rdmol! It could be a bug from OpenFF. Double check your structure!")
			openff_mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

		return openff_mol

	@classmethod
	def format_id(cls, index):
		return f"{index:0>3d}"