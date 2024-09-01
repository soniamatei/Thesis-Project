import IPython.core.magic as Imagic
import isort


@Imagic.register_cell_magic
def isortify(_, cell: str):
    """
    Use isort package on imports in jupyter notebooks.
    :param cell: the code in the cell
    """
    sorted_cell = isort.code(code=cell)
    print(sorted_cell)


def load_ipython_extension(ipython):
    """
    Called when the extension is loaded
    :param ipython: the current InteractiveShell
    """
    ipython.register_magic_function(func=isortify, magic_kind="cell")
