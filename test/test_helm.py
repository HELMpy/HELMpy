"""
HELMpy, open source package of power flow solvers.

Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""
import random

import pytest
import numpy

from helmpy.core.helm_ds_m1_pv1 import helm_ds_m1_pv1
from helmpy.core.helm_ds_m1_pv2 import helm_ds_m1_pv2
from helmpy.core.helm_ds_m2_pv1 import helm_ds_m2_pv1
from helmpy.core.helm_ds_m2_pv2 import helm_ds_m2_pv2
from helmpy.core.helm_pv1 import helm_pv1
from helmpy.core.helm_pv2 import helm_pv2
from helmpy.core.nr import nr
from helmpy.core.nr_ds import nr_ds
from helmpy.util.root_path import ROOT_PATH


@pytest.mark.parametrize(
    'file_name, ', [
        ROOT_PATH / 'data' / 'case' / 'case118.xlsx',
        ROOT_PATH / 'data' / 'case' / 'case2869pegase.xlsx',
        ROOT_PATH / 'data' / 'case' / 'case1354pegase.xlsx',
        ROOT_PATH / 'data' / 'case' / 'case9.xlsx',
    ],
    ids=['case118', 'case2869pegase', 'case1354pegase', 'case9']
)
@pytest.mark.parametrize(
    'function, ', [
        nr,
        nr_ds,
        helm_pv1,
        helm_pv2,
        helm_ds_m1_pv1,
        helm_ds_m1_pv2,
        helm_ds_m2_pv1,
        helm_ds_m2_pv2,
    ],
)
def test_helm_py(function, file_name):
    """
    Test the HELMpy package

    :return: None
    """
    # Seed for deterministic results
    random.seed(42)
    numpy.random.mtrand.seed(42)

    result = function(str(file_name), Mismatch=1e-8, Scale=1.02, Print_Details=True)
    print(result)

    return

    # TODO Make assertions


if __name__ == '__main__':
    test_helm_py()
