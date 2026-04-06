"""Inspired by https://github.com/Farama-Foundation/Gymnasium/blob/main/setup.py"""

import pathlib

from setuptools import find_packages, setup

CWD = pathlib.Path(__file__).absolute().parent


def get_version():
    """Gets the sheeprl version."""
    path = CWD / "sheeprl" / "sheeprl" / "__init__.py"
    content = path.read_text()

    for line in content.splitlines():
        if line.startswith("__version__"):
            return line.strip().split()[-1].strip().strip('"')
    raise RuntimeError("Bad version data in __init__.py")


upstream_packages = find_packages(where="sheeprl", include=["sheeprl", "sheeprl.*", "hydra_plugins", "hydra_plugins.*"])
local_packages = find_packages(where=".", include=["autoattack", "autoattack.*", "agent", "agent.*"])

setup(
    name="sheeprl",
    version=get_version(),
    packages=upstream_packages + local_packages,
    package_dir={
        "": ".",
        "sheeprl": "sheeprl/sheeprl",
        "hydra_plugins": "sheeprl/hydra_plugins",
    },
    py_modules=["main"],
)
