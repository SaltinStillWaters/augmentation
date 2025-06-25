from text_utils import *
from utils import *

import os


uniques = set()

base_dir = 'augmented/semantic/'

for x, filename in enumerate(os.listdir(base_dir)):
    print(x, end=', ')
    filepath = os.path.join(base_dir, filename)
    uniques |= check_alignment(filepath)
    
print(uniques)