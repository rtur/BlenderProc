from glob import glob
from random import choice, choices

from src.main.Provider import Provider
from src.utility.Utility import Utility


class Path(Provider):
    """ Samples a path to one of the files in folder defined by a path.

        Example 1: return a path to a random .obj file in the defined folder.

        {
          "provider": "sampler.Path",
          "path": "/home/path/to/folder/*.obj"
        }

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "path", "A path to a folder containing files. Type: string."
        "sample_num", "Number of paths to return. If 1 return type "
                      "is a string, otherwise a list."
    """

    def __init__(self, config):
        Provider.__init__(self, config)

    def run(self):
        """ Samples a path to an object.

        :return: A path to object. Type: string.
        """
        # get path to folder
        path = Utility.resolve_path(self.config.get_string("path"))

        # get list of paths
        paths = glob(path)

        sample_num = self.config.get_int("sample_num", 1)
        if sample_num == 1:
            # chose a random one
            chosen_path = choice(paths)
            return chosen_path
        elif sample_num > 1:
            chosen_paths = choices(paths, k=sample_num)
            return chosen_paths
        else:
            raise RuntimeError("sample_num has to be greater than zero.")
