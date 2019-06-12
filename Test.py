
"""
HELMpy, open source package of power flow solvers developed on Python 3 
Copyright (C) 2019 Tulio Molina tuliojose8@gmail.com and Juan José Ortega juanjoseop10@gmail.com

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program. If not, see <https://www.gnu.org/licenses/>.
"""


# Test of helmpy package

import helmpy

#Files
xls118 = 'case118.xlsx'
xls2869 = 'case2869pegase.xlsx'
xls1354 = 'case1354pegase.xlsx'
xls9 = 'case9.xlsx'


a=HELMpy.nr(xls118,Mismatch=1e-8,Scale=1.02,Print_Details=True)


b=HELMpy.helm_PV1(xls118,Mismatch=1e-8,Scale=1.02,Print_Details=True)


c=HELMpy.helm_PV2(xls118,Mismatch=1e-8,Scale=1.02,Print_Details=True)
