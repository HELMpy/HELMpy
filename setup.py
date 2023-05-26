import datetime

from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

with open('requirements.txt', 'r') as req_file:
    requirements = req_file.readlines()

TIMESTAMP = str(datetime.datetime.now().replace(microsecond=0).isoformat()).\
    replace('-', '').replace('T', '').replace(':', '')

setup(
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    name='helmpy',
    # version='0.1.0' + 'rc' + TIMESTAMP,
    version='0.1.1',
    url='https://github.com/HELMpy/HELMpy',
    description='HELMpy is an open source package of power flow solvers, including the Holomorphic Embedding Load Flow Method (HELM).',
    packages=find_packages(
        where='.',
        exclude=[
            '*.tests', '*.tests.*', 'tests.*', 'tests',
            '*.test', '*.test.*', 'test.*', 'test',
        ],
    ),
    package_dir={'': '.'},
    install_requires=requirements,
    zip_safe=False,
)
