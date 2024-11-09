def MMFF94_torsion(v1, v2, v3, a):
	return 0.5 * (v1 * (1 + np.cos(a)) + v2 * (1 - np.cos(2*a)) + v3 * (1 + np.cos(3*a)))


def MMFF94_vdw(Rij, e, d):
	return e * ((1.07 * Rij) / (d + 0.07 * Rij))**7 * ((1.12 * Rij**7) / (d**7 + 0.12 * Rij**7) - 2)


def MMFF94_elec(q1, q2, d, e=1):
	return 332.0716 * q1 * q2 / (e * (d + 0.05))


def MMFF4_1_4_distance(dij, djk, dkl, aijk, ajkl, tijkl):
	x = -dij * np.cos(aijk) + djk - dkl * np.cos(ajkl)
	y = dij * np.sin(aijk) - dkl * np.sin(ajkl) * np.cos(tijkl)
	z = dkl * np.sin(ajkl) * np.sin(tijkl)
	return np.sqrt(x**2 + y**2 + z**2)

def dihedral_scoring(rdconf):
	if not rdconf.HasOwningMol():
		
	