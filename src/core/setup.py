import os
import pathlib

import orekit
from orekit.pyhelpers import setup_orekit_curdir


def init_orekit(data_zip: str):
    """Initialise the JVM and load Orekit data from *data_zip*."""
    vm = orekit.initVM()
    zip_path = pathlib.Path(data_zip).resolve()
    os.chdir(zip_path.parent)
    setup_orekit_curdir(zip_path.name)
    return vm
