import io
import os
from setuptools import find_packages, setup, Command


NAME = 'open_seg'
DESCRIPTION = 'sementic segmentation lib'
URL = 'git@github.com:AokiMasataka/open_segment.git'
EMAIL = None
AUTHOR = None
REQUIRES_PYTHON = '>=3.9.0'
VERSION = None

REQUIRED = ['torch', 'torchvision', 'timm']
EXTRAS = {}


here = os.path.abspath(os.path.dirname(__file__))

try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


about = {}
if not VERSION:
    project_slug = NAME.lower().replace('-', '_').replace(' ', '_')
    with open(os.path.join(here, project_slug, 'open_seg/__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=['open_seg', 'tools']),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
)