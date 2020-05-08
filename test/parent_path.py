import sys
from pathlib import Path

def append_root_path():
	file = Path(__file__).resolve()
	root = file.parents[1]
	if(str(root) in sys.path):
		pass
	else:
		sys.path.append(str(root))

append_root_path()