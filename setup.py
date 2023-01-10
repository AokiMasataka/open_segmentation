import io
import os
from setuptools import setup, find_packages
from openseg import __version__


NAME = 'openseg'
DESCRIPTION = 'sementic segmentation lib'
URL = 'git@github.com:AokiMasataka/open_segment.git'
EMAIL = None
AUTHOR = None
REQUIRES_PYTHON = '>=3.8.0'
VERSION = __version__

INSTALL_REQUIRES = ['torch', 'torchvision', 'tqdm', 'openbacks']
EXTRAS_REQUIRE = {}
PACKAGES = ['openseg']

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3 :: Only',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Visualization',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Topic :: Multimedia :: Graphics',
    'Framework :: Matplotlib',
]


here = os.path.abspath(os.path.dirname(__file__))
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


def take_package_name(name):
    if name.startswith('-e'):
        return name[name.find('=')+1:name.rfind('-')]
    else:
        return name.strip()

def load_requires_from_file(filepath):
    with open(filepath) as fp:
        return [take_package_name(pkg_name) for pkg_name in fp.readlines()]


def load_links_from_file(filepath):
    res = []
    with open(filepath) as fp:
        for pkg_name in fp.readlines():
            if pkg_name.startswith("-e"):
                res.append(pkg_name.split(" ")[1])
    return res


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=True,
    license='MIT',
    classifiers=CLASSIFIERS
)