from setuptools import setup, find_packages

with open('README.md', 'r') as readme_file:
    long_description = readme_file.read()

setup(
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    name='helmpy',
    version='0.1.0rc0',
    url='https://github.com/HELMpy/HELMpy',
    author='Tulio Molina, Juan José Ortega',
    author_email='tuliojose8@gmail.com, Juan José Ortega juanjoseop10@gmail.com',
    description='HELMpy is an open source package of power flow solvers, including the Holomorphic Embedding Load Flow Method (HELM).',
    packages=find_packages(
        where='.',
        exclude=[
            '*.tests', '*.tests.*', 'tests.*', 'tests',
            '*.test', '*.test.*', 'test.*', 'test',
        ],
    ),
    package_dir={'': '.'},
    install_requires=[
    ],
    zip_safe=False,
)
