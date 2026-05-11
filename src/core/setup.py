import os
import pathlib
import sys

import orekit
from orekit.pyhelpers import setup_orekit_curdir


def _jvm_dll_exists(path: pathlib.Path) -> bool:
    return (path / "bin" / "server" / "jvm.dll").exists() or \
           (path / "bin" / "server" / "libjvm.so").exists()


def _ensure_java_home():
    """Set JAVA_HOME to the conda-env JDK when it is not already configured.

    Searches the current Python prefix first, then all sibling conda envs,
    so the function works whether the kernel is the orbit env or base Anaconda.
    """
    if os.environ.get("JAVA_HOME"):
        return

    search_roots = [pathlib.Path(sys.prefix)]

    # Also search sibling envs (handles notebook running in base Anaconda kernel)
    envs_dir = pathlib.Path(sys.prefix).parent
    if envs_dir.name == "envs":
        # sys.prefix is already inside envs/ — add the base too
        search_roots.append(envs_dir.parent)
    else:
        # sys.prefix is the base; add all envs/*/
        envs_path = pathlib.Path(sys.prefix) / "envs"
        if envs_path.is_dir():
            search_roots.extend(sorted(envs_path.iterdir()))

    for root in search_roots:
        for suffix in (
            pathlib.Path("Library") / "lib" / "jvm",  # conda-forge openjdk Windows
            pathlib.Path("Library") / "jvm",
            pathlib.Path("jre"),
        ):
            candidate = root / suffix
            if _jvm_dll_exists(candidate):
                os.environ["JAVA_HOME"] = str(candidate)
                return


def init_orekit(data_zip: str):
    """Initialise the JVM and load Orekit data from *data_zip*."""
    _ensure_java_home()
    vm = orekit.initVM()
    zip_path = pathlib.Path(data_zip).resolve()
    os.chdir(zip_path.parent)
    setup_orekit_curdir(zip_path.name)
    return vm
