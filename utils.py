import numpy as np

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    
def read_tsplib(filename):
    """
    Reads a TSPLIB file and returns the list of node coordinates.
    """
    node_coords = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    start = False
    for line in lines:
        if line.strip() == 'NODE_COORD_SECTION':
            start = True
            continue
        if start:
            if line.strip() == 'EOF':
                break
            parts = line.strip().split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                node_coords.append((x, y))

    # Normalize coordinates to [0, 1]
    coords_array = np.array(node_coords)
    min_coords = coords_array.min(axis=0)
    max_coords = coords_array.max(axis=0)
    normalized_coords = (coords_array - min_coords) / (max_coords - min_coords + 1e-8)
    return normalized_coords.tolist()

def read_solutions(filename):
    """
    Reads the solutions file and returns a dictionary of best known tour lengths.
    """
    solutions = {}
    with open(filename, 'r') as f:
        for line in f:
            if ':' in line:
                name, length = line.strip().split(':')
                name = name.strip()
                try:
                    length = float(length.strip())
                    solutions[name] = length
                except: 
                    pass
    return solutions

import os

if not os.path.exists('tours'):
    os.makedirs('tours')