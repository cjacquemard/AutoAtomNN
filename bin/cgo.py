import itertools
import pathlib
from typing import Sequence, Union


"""
Library dedicated to Compiled Graphics Objects (CGO) of PyMOL software 
"""


class CGOFormatter:
    """Export 3D lines to Compiled Graphics Objects (CGO) for PyMOL
    Parameters
    ----------
    filepath : Union[str, pathlib.Path]
        The filepath of the CGO file.
    name : str
        Name of the CGO object.
    rgb:
        Color of lines in RGB format from 0 to 1.
    """
    def __init__(self, filepath: Union[str, pathlib.Path] = None, name=None, rgb=None, pwidth=0.5):
        self.filepath = filepath
        self.name = name
        self.rgb = rgb
        self.pwidth = pwidth


def export_points_to_cgo(points: Sequence[Sequence[float]], cgo_formatter: CGOFormatter) -> None:
    """Export 3D points to Compiled Graphics Objects (CGO) for PyMOL
    Parameters
    ----------
    points : Sequence[Sequence[float, float, float]]
        Nx3 array of starting point coordinates of lines.
    cgo_formatter : CGOFormatter

    Returns
    -------
    None
    """
    with open(cgo_formatter.filepath, 'w') as f:
        f.write('from pymol.cgo import *\n')
        obj = ['BEGIN,LINES,COLOR,{:.2f},{:.2f},{:.2f}'.format(*cgo_formatter.rgb)]
        delta = cgo_formatter.pwidth / 2
        for x, y, z in points:
            obj.append('VERTEX,{},{},{}'.format(x - delta, y, z))
            obj.append('VERTEX,{},{},{}'.format(x + delta, y, z))

            obj.append('VERTEX,{},{},{}'.format(x, y - delta, z))
            obj.append('VERTEX,{},{},{}'.format(x, y + delta, z))

            obj.append('VERTEX,{},{},{}'.format(x, y, z - delta))
            obj.append('VERTEX,{},{},{}'.format(x, y, z + delta))

        obj.append('END')
        f.write('obj = [{}]\n'.format(','.join(obj)))

        f.write(f'cmd.load_cgo(obj, "{cgo_formatter.name}")\n')


def export_lines_to_cgo(starts: Sequence[Sequence[float]],
                        ends: Sequence[Sequence[float]],
                        cgo_formatter: CGOFormatter) -> None:
    """Export 3D lines to Compiled Graphics Objects (CGO) for PyMOL
    Parameters
    ----------
    starts : Sequence[Sequence[float, float, float]]
        Nx3 array of starting point coordinates of lines.
    ends : Sequence[Sequence[float, float, float]]
        Nx3 array of end point coordinates of lines.
    cgo_formatter : CGOFormatter

    Returns
    -------
    None
    """
    with open(cgo_formatter.filepath, 'w') as f:
        f.write('from pymol.cgo import *\n')
        obj = ['BEGIN,LINES,COLOR,{:.2f},{:.2f},{:.2f}'.format(*cgo_formatter.rgb)]
        for start, end in zip(starts, ends):
            obj.append('VERTEX,{},{},{}'.format(*start))
            obj.append('VERTEX,{},{},{}'.format(*end))

        obj.append('END')
        f.write('obj = [{}]\n'.format(','.join(obj)))

        f.write(f'cmd.load_cgo(obj, "{cgo_formatter.name}")\n')


def export_trimesh_to_cgo(mesh, cgo_formatter: CGOFormatter) -> None:
    import trimesh
    """Export trimesh object to Compiled Graphics Objects (CGO) for PyMOL
    Parameters
    ----------
    mesh : trimesh.Trimesh
    cgo_formatter : CGOFormatter

    Returns
    -------
    None
    """
    with open(cgo_formatter.filepath, 'w') as f:
        f.write('from pymol.cgo import *\n')
        obj = ['BEGIN,LINES,COLOR,{:.2f},{:.2f},{:.2f}'.format(*cgo_formatter.rgb)]
        for face in mesh.faces:
            for pair in itertools.combinations(face, 2):
                for vertex_id in pair:
                    coord = mesh.vertices[vertex_id]
                    obj.append('VERTEX,{},{},{}'.format(*coord))
        obj.append('END')
        f.write('obj = [{}]\n'.format(','.join(obj)))

        f.write(f'cmd.load_cgo(obj, "{cgo_formatter.name}")\n')


def export_box_to_cgo(min_bbox: Sequence[float], max_bbox: Sequence[float], cgo_formatter: CGOFormatter) -> None:
    """Export box to Compiled Graphics Objects (CGO) for PyMOL
    Parameters
    ----------
    min_bbox : Sequence[float, float, float]
        3D points of the lower corner of the box
    max_bbox : Sequence[float, float, float]
        3D points of the upper corner of the box
    cgo_formatter : CGOFormatter

    Returns
    -------
    None
    """
    with open(cgo_formatter.filepath, 'w') as f:
        f.write('from pymol.cgo import *\n')
        obj = [
            'BEGIN,LINES,COLOR,{:.2f},{:.2f},{:.2f}'.format(*cgo_formatter.rgb),

            'VERTEX,{},{},{}'.format(*min_bbox),
            'VERTEX,{},{},{}'.format(max_bbox[0], min_bbox[1], min_bbox[2]),

            'VERTEX,{},{},{}'.format(max_bbox[0], min_bbox[1], min_bbox[2]),
            'VERTEX,{},{},{}'.format(max_bbox[0], max_bbox[1], min_bbox[2]),

            'VERTEX,{},{},{}'.format(max_bbox[0], min_bbox[1], min_bbox[2]),
            'VERTEX,{},{},{}'.format(max_bbox[0], min_bbox[1], max_bbox[2]),

            'VERTEX,{},{},{}'.format(*min_bbox),
            'VERTEX,{},{},{}'.format(min_bbox[0], max_bbox[1], min_bbox[2]),

            'VERTEX,{},{},{}'.format(min_bbox[0], max_bbox[1], min_bbox[2]),
            'VERTEX,{},{},{}'.format(max_bbox[0], max_bbox[1], min_bbox[2]),

            'VERTEX,{},{},{}'.format(min_bbox[0], max_bbox[1], min_bbox[2]),
            'VERTEX,{},{},{}'.format(min_bbox[0], max_bbox[1], max_bbox[2]),

            'VERTEX,{},{},{}'.format(*min_bbox),
            'VERTEX,{},{},{}'.format(min_bbox[0], min_bbox[1], max_bbox[2]),

            'VERTEX,{},{},{}'.format(min_bbox[0], min_bbox[1], max_bbox[2]),
            'VERTEX,{},{},{}'.format(max_bbox[0], min_bbox[1], max_bbox[2]),

            'VERTEX,{},{},{}'.format(min_bbox[0], min_bbox[1], max_bbox[2]),
            'VERTEX,{},{},{}'.format(min_bbox[0], max_bbox[1], max_bbox[2]),

            'VERTEX,{},{},{}'.format(min_bbox[0], max_bbox[1], max_bbox[2]),
            'VERTEX,{},{},{}'.format(*max_bbox),

            'VERTEX,{},{},{}'.format(max_bbox[0], min_bbox[1], max_bbox[2]),
            'VERTEX,{},{},{}'.format(*max_bbox),

            'VERTEX,{},{},{}'.format(max_bbox[0], max_bbox[1], min_bbox[2]),
            'VERTEX,{},{},{}'.format(*max_bbox),

            'END',
        ]
        f.write('obj = [{}]\n'.format(','.join(obj)))

        f.write(f'cmd.load_cgo(obj, "{cgo_formatter.name}")\n')
