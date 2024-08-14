import random

import numpy as np

class Welz:
	@staticmethod
	def get_bounding_ball(S, epsilon=1e-7):
		"""
		Computes the smallest bounding ball of a set of points
		Parameters
		----------
		S : (M, N) ndarray, where 1 <= M <= N + 1
			The input points
		epsilon : float
			Tolerance used when testing if a set of point belongs to the same sphere.
			Default is 1e-7
		Returns
		-------
		C, r2 : ((2) ndarray, float)
			The center and the squared radius of the circumsphere
		"""

		# Iterative implementation of Welzl's algorithm, see
		# "Smallest enclosing disks (balls and ellipsoids)" Emo Welzl 1991
		def get_circumsphere(S, tol=1e-7):
			"""
			Computes the circumsphere of a set of points
			Parameters
			----------
			S : (M, N) ndarray, where 1 <= M <= N + 1
				The input points
			Returns
			-------
			C, r2 : ((2) ndarray, float)
				The center and the squared radius of the circumsphere
			"""

			# Check if some points are identical
			cells = (S / tol).astype(int) # Tolerance
			indices = np.unique(cells, axis=0, return_index=True)[1]
			S = S[indices]

			U = S[1:] - S[0]
			B = np.sqrt(np.sum(U ** 2, axis=1))
			U /= B[:, None]
			B /= 2
			try:
				C = np.dot(np.linalg.solve(np.inner(U, U), B), U)
			except np.linalg.LinAlgError:
				breakpoint()
				raise ValueError("Cannot solve with identical points")

			r2 = np.sum(C ** 2)
			C += S[0]
			return C, r2

		def circle_contains(D, p):
			c, r2 = D
			return np.sum((p - c) ** 2) <= r2

		def get_boundary(R):
			if len(R) == 0:
				return np.zeros(S.shape[1]), 0.

			if len(R) <= S.shape[1] + 1:
				return get_circumsphere(S[R])

			c, r2 = get_circumsphere(S[R[:S.shape[1] + 1]])
			if np.all(np.fabs(np.sum((S[R] - c) ** 2, axis=1) - r2) < epsilon):
				return c, r2

		class Node(object):
			def __init__(self, P, R):
				self.P = P
				self.R = R
				self.D = None
				self.pivot = None
				self.left = None
				self.right = None

		def traverse(node):
			stack = [node]
			while len(stack) > 0:
				node = stack.pop()

				if len(node.P) == 0 or len(node.R) >= S.shape[1] + 1:
					node.D = get_boundary(node.R)
				elif node.left is None:
					node.pivot = np.random.choice(node.P)
					node.left = Node(list(set(node.P) - set([node.pivot])), node.R)
					stack.extend((node, node.left))
				elif node.right is None:
					if circle_contains(node.left.D, S[node.pivot]):
						node.D = node.left.D
					else:
						node.right = Node(node.left.P, node.R + [node.pivot])
						stack.extend((node, node.right))
				else:
					node.D = node.right.D
					node.left, node.right = None, None

		S = S.astype(float, copy=False)
		root = Node(range(S.shape[0]), [])
		traverse(root)

		return root.D[0], np.sqrt(root.D[1])