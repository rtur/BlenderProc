import os

import bpy
import numpy as np
import lz4.frame as lz4
import cv2
import pickle

from src.writer.WriterInterface import WriterInterface
from src.writer.FileLock import FileLock


class PickleWriter(WriterInterface):
    """ For each key frame merges all registered output files into one lz4 compressed pickle file

    **Configuration**:

    .. csv-table::
        :header: "Parameter", "Description"

        "append_to_existing_output", "If true, the names of the output pickle files will be chosen in a way such that "
                                    "there are no collisions with already existing pickle files in the output directory. "
                                    "Type: bool. Default: False"
        "delete_temporary_files_afterwards", "True, if all temporary files should be deleted after merging. "
                                             "Type: bool. Default value: True."
       "stereo_separate_keys", "If true, stereo images are saved as two separate images *_0 and *_1. Type: bool. "
                                "Default: False (stereo images are combined into one np.array (2, ...))."
        "avoid_rendering", "If true, exit. Type: bool. Default: False."
    """

    def __init__(self, config):
        WriterInterface.__init__(self, config)
        self._avoid_rendering = config.get_bool("avoid_rendering", False)
        self._ext = ".pickle.lz4"

        self._lockpath = \
            os.path.join(self._determine_output_dir(False), "lockfile")

    def run(self):
        with FileLock(self._lockpath):
            if self._avoid_rendering:
                print("Avoid rendering is on, no output produced!")
                return

            if self.config.get_bool("append_to_existing_output", False):
                frame_offset = 0
                # Look for pickle file with highest index
                for path in os.listdir(self._determine_output_dir(False)):
                    if path.endswith(self._ext):
                        index = path[:-len(self._ext)]
                        if index.isdigit():
                            frame_offset = max(frame_offset, int(index) + 1)
            else:
                frame_offset = 0

            # Go through all frames
            for frame in range(bpy.context.scene.frame_start, bpy.context.scene.frame_end):

                pickle_path = os.path.join(self._determine_output_dir(False),
                                           f"{frame+frame_offset:06d}{self._ext}")
                scene_dct = {}
                if 'output' not in bpy.context.scene:
                    print("No output was designed in prior models!")
                    return
                # Go through all the output types
                print("Merging data for frame " + str(frame) + " into " + pickle_path)

                for output_type in bpy.context.scene["output"]:
                    use_stereo = output_type["stereo"]
                    # Build path (path attribute is format string)
                    file_path = output_type["path"]
                    if '%' in file_path:
                        file_path = file_path % frame

                    if use_stereo:
                        path_l, path_r = self._get_stereo_path_pair(file_path)

                        img_l, new_key, new_version = self._load_and_postprocess(path_l, output_type["key"],
                                                                                   output_type["version"])
                        img_r, new_key, new_version = self._load_and_postprocess(path_r, output_type["key"],
                                                                                   output_type["version"])

                        if self.config.get_bool("stereo_separate_keys", False):
                            scene_dct[new_key + "_0"] = img_l
                            scene_dct[new_key + "_1"] = img_r
                        else:
                            data = np.array([img_l, img_r])
                            scene_dct[new_key] = data

                    else:
                        data, new_key, new_version = \
                                self._load_and_postprocess(
                                        file_path, output_type["key"],
                                        output_type["version"])

                        scene_dct[new_key] = data

                    scene_dct[new_key + "_version"] = new_version

                for k in list(scene_dct.keys()):
                    if k == "distance":
                        scene_dct[k] = scene_dct[k].astype(np.float16)
                    elif k in ["colors", "segmap"]:
                        scene_dct[k] = cv2.imencode('.png', scene_dct[k])[1]

                with open(pickle_path, "wb") as f:
                    data = pickle.dumps(scene_dct)
                    data = lz4.compress(
                            data,
                            compression_level=lz4.COMPRESSIONLEVEL_MINHC)
                    f.write(data)

    def _get_stereo_path_pair(self, file_path):
        """
        Returns stereoscopic file path pair for a given "normal" image file path.
        :param file_path: The file path of a single image. Type: string.
        :return: The pair of file paths corresponding to the stereo images,
        """
        path_split = file_path.split(".")
        path_l = "{}_L.{}".format(path_split[0], path_split[1])
        path_r = "{}_R.{}".format(path_split[0], path_split[1])

        return path_l, path_r
