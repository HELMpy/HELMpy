# HELMpy

HELMpy is an open source package of power flow solvers.

This package contains the Holomorphic Embedding Load flow Method (HELM) and
the Newton-Raphson (NR) algorithm.
The intention of HELMpy is to support research, especially on the HELM,
and to contribute with the development of open source code
related to this subject.
The developed code is properly commented and organized
so it would be easy to understand and modify.

## Repository structure

- data: sample data of large-sized, complex practical grids for testing purposes. Already computed results can also be found
- helmm: matlab files for downloading and parsing to `.xlsx` matpower grids
- helmpy: scripts with core functionality
- test: scripts for running tests using sample grids from `data` directory

## Compatibility

This package is compatible with Python 3.6 and 3.7.

## History

This package was developed by Tulio Molina and Juan José Ortega
as a part of their thesis research
to obtain the degree of Electrical Engineer
at Universidad de los Andes (ULA) in Mérida, Venezuela.

## HELMpy Guide

Please refer to `HELMpy user's guide.pdf`.

## License - AGPLv3

    HELMpy, open source package of power flow solvers developed on Python 3
    Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
