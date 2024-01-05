from setuptools import setup
from setuptools import find_packages

from openseg import __version__


NAME = 'openseg'
VERSION = __version__


def package_name(name):
    if 'git+' in name:
        return 'openback'
    else:
        return name


def _requires_from_file(filename):
    return [package_name(name) for name in open(filename).read().splitlines()]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('./requirements.txt'),
)