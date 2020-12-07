
from sys import version_info
if version_info.major == 2:
    raise Exception("This script only works with python3.x!")

import os
import csv
from urllib.request import urlretrieve, build_opener, install_opener
from urllib.error import HTTPError
import subprocess


if __name__ == "__main__":
    # setting the default header, else the server does not allow the download
    opener = build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    install_opener(opener)

    # set the download directory relative to this one
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cc_texture_dir = os.path.join(current_dir, "..", "resources", "cctextures")

    os.makedirs(cc_texture_dir, exist_ok=True)

    # download the csv file, which contains all the download links
    csv_url = "https://cc0textures.com/api/v1/downloads_csv"
    csv_file_path = os.path.join(cc_texture_dir, "full_info.csv")
    urlretrieve(csv_url, csv_file_path)

    # extract the download links with the asset name
    data = {}
    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        for line in csv_reader:
            if line["Filetype"] == "zip" and line["DownloadAttribute"] == "2K-JPG":
                data[line["AssetID"]] = line["PrettyDownloadLink"]

    excluding_list = [
        "Fingerprints", "ChristmasTreeOrnament", "ColdCuts", "PaymentCard",
        "Leaf", "LeafSet", "sign", "roadlines", "manhole", "backdrop",
        "foliage", "TreeEnd", "TreeStump", "3DBread", "3DApple", "FlowerSet",
        "FoodSteps", "PineNeedles", "Grate", "PavingEdge", "Painting",
        "RockBrush", "WrinklesBrush", "Sticker", "3DRock"]

    # download each asset and create a folder for it (unpacking + deleting the zip included)
    for index, (asset, link) in enumerate(data.items()):
        do_not_use = False
        for exclude_element in excluding_list:
            if asset.lower().startswith(exclude_element.lower()):
                do_not_use = True
                break
        if do_not_use:
            continue
        print("Download asset: {} of {}/{}".format(asset, index, len(data)))
        current_folder = os.path.join(cc_texture_dir, asset)
        if os.path.exists(current_folder):
            print("\tSkipping, it already exists")
            continue
        else:
            os.makedirs(current_folder)
        current_file_path = os.path.join(current_folder, "{}.zip".format(asset))

        try:
            urlretrieve(link, current_file_path)
            subprocess.call(["unzip {} -d {}> /dev/null".format(current_file_path, current_folder)], shell=True)
            os.remove(current_file_path)
            downed_files = os.listdir(current_folder)
            if "opacity" in "".join(downed_files).lower():
                print(f"Removing {asset} assets because they have an opacity map.")
                for f in downed_files:
                    os.remove(os.path.join(current_folder, f))
                # we leave the folder empty so that the
                # asset won't be downloaded again
        except HTTPError as err:
            os.removedirs(current_folder)
            print(f"\tFailed due to: {err}")

    print("Done downloading textures, saved in {}".format(cc_texture_dir))




