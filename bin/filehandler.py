import numpy as np

class AtomField:
	def __init__(self, name, ftype, col_loc, fformat, mandatory, default=''):
		self._name = name
		self._ftype = ftype
		self._col_loc = col_loc
		self._mandatory = mandatory
		self._fformat = fformat
		self._default = default

	@property
	def name(self):
		return self._name

	@property
	def ftype(self):
		return self._ftype

	@property
	def col_loc(self):
		return self._col_loc

	@property
	def default(self):
		return self._default

	def format_data(self, data):
		fmt = "{:"+self._fformat+"}"
		return fmt.format(data)

	def get_data(self, line):
		start, end = self._col_loc

		try:
			raw_data = line[start:end]
		except IndexError:
			raise IndexError("Bad indices {start}, {end} to extract data")

		stripped_data = raw_data.strip()

		if stripped_data:
			try:
				data = self._ftype(raw_data.strip())
			except ValueError:
				raise ValueError(f"The '{self.name}' field cannot convert value '{raw_data}' of type {self._ftype}")
		else:
			if self._mandatory:
				raise ValueError(f"The '{self.name}' field is mandatory and a value must be present")

			# TODO: Tell to the user that the default value is used
			data = self._default

		return data


class AtomLine:
	def __init__(self, *atom_fields):
		self._atom_fields = {atom_field.name: atom_field for atom_field in atom_fields}

	def __getitem__(self, key):
		return self._atom_fields[key]

	def __iter__(self):
		return iter(self._atom_fields.keys())

	@property
	def length(self):
		max_col = 0
		for atom_field in self._atom_fields.values():
			if atom_field._col_loc[1] > max_col:
				max_col = atom_field._col_loc[1]
		return max_col

	def fields(self):
		return self._atom_fields.values()


class MolFileHandler:
	_ATOM_FIELDS = {}

	def __init__(self, filepath):
		self._filepath = filepath
		self._lines = []
		self._atom_line_indices = []
		self._current_line = None
		self._is_parsed = False

		self._data = {atom_field_name: [] for atom_field_name in self._ATOM_FIELDS}

	def is_atom_line(self):
		raise NotImplementedError

	def parse(self):
		raise NotImplementedError

	def parse_line(self, line):
		raise NotImplementedError

	def rewrite(self, filepath):
		new_lines = copy(self._lines)

		# Update atom lines
		for i, atom_line_index in enumerate(self._atom_line_indices):
			line_data = [self._data[name][i] for name in self._data]
			new_line = self.format_line(*line_data)
			new_lines[atom_line_index] = 	new_line+'\n'

		with open(filepath, 'w') as f:
			for line in new_lines:
				f.write(line)

	@classmethod
	def format_line(cls, *data):
		# TODO: Maybe use a dict instead of a list
		line_chars = np.array([' '] * cls._ATOM_FIELDS.length)
		for atom_field, data in zip(cls._ATOM_FIELDS.fields(), data):
			start, end = atom_field._col_loc
			line_chars[start:end] = [x for x in atom_field.format_data(data)]

		return ''.join(line_chars)


class PDB(MolFileHandler):
	_ATOM_FIELDS = AtomLine(
		AtomField("record",     str,   ( 0,  6), " <6",  True ),
		AtomField("serial",     int,   ( 6, 11), " >5",  True ),
		AtomField("name",       str,   (12, 16), " <4",  True ),
		AtomField("altloc",     str,   (16, 17), " >1",  False),
		AtomField("resname",    str,   (17, 20), " >3",  True ),
		AtomField("chainid",    str,   (21, 22), " >1",  False ),
		AtomField("resid",      int,   (22, 26), " >4",  True ),
		AtomField("icode",      str,   (26, 27), " >1",  False),
		AtomField("x",          float, (30, 38), "8.3f", True ),
		AtomField("y",          float, (38, 46), "8.3f", True ),
		AtomField("z",          float, (46, 54), "8.3f", True ),
		AtomField("occupancy",  float, (54, 60), "6.2f", True ),
		AtomField("tempfactor", float, (60, 66), "6.2f", True ),
		AtomField("element",    str,   (76, 78), " >2",  True ),
		AtomField("chargeval",  int,   (78, 79), " >1",  False),
		AtomField("chargesign", str,   (79, 80), " >1",  False),
	)

	def __init__(self, filepath):
		super().__init__(filepath)

	# TODO: Automatically set attributes based on self._data keys.
	@property
	def positions(self):
		return np.vstack([self._data["x"], self._data["y"], self._data["z"]]).T

	@property
	def elements(self):
		return self._data["element"]

	def set_positions(self, positions):
		self._data["x"] = positions[:,0]
		self._data["y"] = positions[:,1]
		self._data["z"] = positions[:,2]

	def is_atom_line(self):
		record_field = self._ATOM_FIELDS["record"]
		record_data = record_field.get_data(self._current_line)
		if record_data in ("ATOM", "HETATM"):
			return True
		else:
			return False

	def parse_line(self):
		for atom_field_name in self._ATOM_FIELDS:
			atom_field = self._ATOM_FIELDS[atom_field_name]
			atom_field_data = atom_field.get_data(self._current_line)
			self._data[atom_field_name].append(atom_field_data)

	def parse(self):
		with open(self._filepath) as f:
			for i, line in enumerate(f):
				self._current_line = line
				self._lines.append(line)

				if self.is_atom_line():
					self._atom_line_indices.append(i)
					self.parse_line()

		for atom_field_name in self._data:
			self._data[atom_field_name] = np.array(self._data[atom_field_name])

		self._atom_line_indices = np.array(self._atom_line_indices)
		self._is_parsed = True

	@classmethod
	def save_population(cls, filepath, population, indices=None, fitness=None, cluster_labels=None):
		record = "HETATM"
		name = "DU"
		altloc = ''
		chainid = 'A'
		icode = ''
		temp = 0.0
		element = "Du"
		charge_val = ''
		charge_sign = ''

		if indices is not None:
			serials = indices + 1
		else:
			serials = np.arange(len(population)).astype(int) + 1

		if fitness is not None:
			if len(population) != len(fitness):
				raise ValueError("The length of population and fitness must be the same")

			occs = np.array(fitness)
			if (occs > 100.0).any():
				raise ValueError("The maximum value for B-factor is 100")
		else:
			occs = [100.0] * len(population)

		if cluster_labels is not None:
			if len(population) != len(cluster_labels):
				raise ValueError("The length of population and cluster_labels must be the same")

			resnames = [f"CL{label+1}" for label in cluster_labels]
			resids = [label+1 for label in cluster_labels]
		else:
			resnames = ["POP"] * len(population)
			resids = [1] * len(population)

		with open(filepath, 'a') as f:
			f.write("MODEL\n")
			for i, items in enumerate(zip(population, serials, resnames, resids, occs), 1):
				individual, serial, resname, resid, occ = items
				x, y, z = individual
				f.write(cls.format_line(record, serial, name, altloc, resname, chainid, resid, icode, x, y, z, occ, temp, element, charge_val, charge_sign)+'\n')
			f.write("ENDMDL\n")


class SDF(MolFileHandler):
	_ATOM_FIELDS = AtomLine(
		AtomField("x",            float, ( 0,  10), "10.4f", True ),
		AtomField("y",            float, (10,  20), "10.4f", True ),
		AtomField("z",            float, (20,  30), "10.4f", True ),
		AtomField("element",      str,   (31,  34), " <3",   True ),
		AtomField("massdiff",     int,   (34,  36), " >2",   True ),
		AtomField("charge",       int,   (36,  39), " >3",   True ),
		AtomField("stereoparity", int,   (39,  42), " >3",   True ),
		AtomField("hcount",       int,   (42,  45), " >3",   True ),
		AtomField("stereocare",   int,   (45,  48), " >3",   True ),
		AtomField("valency",      int,   (48,  51), " >3",   True ),
		AtomField("designator",   int,   (51,  54), " >3",   False),
		AtomField("reactype",     int,   (54,  57), " >3",   False),
		AtomField("reacnum",      int,   (57,  60), " >3",   False),
		AtomField("mapping",      int,   (60,  63), " >3",   False),
		AtomField("invretflag",   int,   (63,  66), " >3",   False),
		AtomField("changeflag",   int,   (66,  69), " >3",   False),
	)

	def __init__(self, filepath):
		super().__init__(filepath)

	# TODO: Automatically set attributes based on self._data keys.
	@property
	def positions(self):
		return np.vstack([self._data["x"], self._data["y"], self._data["z"]]).T

	@property
	def elements(self):
		return self._data["element"]

	def set_positions(self, positions):
		self._data["x"] = positions[:,0]
		self._data["y"] = positions[:,1]
		self._data["z"] = positions[:,2]

	def is_atom_line(self):
		return self._mol_line_index > 3 and self._mol_line_index <= self._atom_count + 3 

	def parse_line(self):
		for atom_field_name in self._ATOM_FIELDS:
			atom_field = self._ATOM_FIELDS[atom_field_name]
			atom_field_data = atom_field.get_data(self._current_line)
			self._data[atom_field_name].append(atom_field_data)

	def parse(self):
		self._mol_line_index = 0

		with open(self._filepath) as f:
			for i, line in enumerate(f):
				self._current_line = line
				self._lines.append(line)

				# Check if the line corresponds to counts
				# NOTE: not optimal and quite ugly! But it works.
				if self._mol_line_index == 3:
					self._atom_count = int(line[:3])

				if self.is_atom_line():
					self._atom_line_indices.append(i)
					self.parse_line()

				if line == "$$$$\n":
					self._mol_line_index = 0
				else:
					self._mol_line_index += 1

		for atom_field_name in self._data:
			self._data[atom_field_name] = np.array(self._data[atom_field_name])

		self._atom_line_indices = np.array(self._atom_line_indices)
		self._is_parsed = True