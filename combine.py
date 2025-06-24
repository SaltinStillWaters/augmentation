import os
import itertools
from text_utils import *

root_dir = 'augmented/'

# Get list of subdirectory paths
subdirs = sorted([
    os.path.join(root_dir, d)
    for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d))
])

# Collect file lists for each subdirectory
all_files = []
for subdir in subdirs:
    files = sorted([
        os.path.join(subdir, f)
        for f in os.listdir(subdir)
        if os.path.isfile(os.path.join(subdir, f))
    ])
    all_files.append(files)

print("📁 Files from each subdirectory:")
for i, files in enumerate(all_files):
    print(f"  - {os.path.basename(subdirs[i])}: {len(files)} files")

# Generate all combinations including subsets (i.e., optional subdirs)
combinations = []
num_subdirs = len(all_files)

for r in range(1, num_subdirs + 1):  # skip r=0 (empty combination)
    for subdir_indices in itertools.combinations(range(num_subdirs), r):
        # Select only the file lists for this combination of subdirs
        selected_file_lists = [all_files[i] for i in subdir_indices]
        # Cartesian product of selected subdir file options
        for combo in itertools.product(*selected_file_lists):
            combinations.append(combo)

print(f"\n✅ Total combinations (subdirs optional): {len(combinations)}")
# for c in combinations[:]:  # preview a few
#     print("  →", c)
    
filtered = []
for combo in combinations:
    yes = False
    for x in combo:
        if '50' in x:
            if not yes:
                yes = True
            else:
                break
    else:
        filtered.append(combo)

# for x in filtered:
#     print(x)
    
print(len(filtered))
# with open('out.txt', 'w', encoding='utf-8') as f:
#     for x in filtered:
#         f.write(str(x))
#         f.write('\n')

for a, x in enumerate(filtered):
    temp = [z for z in [read_jsonl(y) for y in x]]
    print(temp)
    save_jsonl(f'out/{a}.jsonl', temp)