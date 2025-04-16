import argparse
from PIL import Image
import scipy
import numpy as np
import hashlib

def i2t(image_path: str):
    image = Image.open(image_path)
    image = image.convert("RGB")

    masks = sam2postprocess(sam2.automatic_mask_generator.generate(image))
    depth = depth_estimator.predict(image)
    
    three_d = [
        [(x, y, depth[x, y]) for x in range(image.width) for y in range(image.height) if mask[x, y]]
        for mask in masks
    ]

    hulls = list(map(scipy.spatial.ConvexHull, three_d))
    n = len(hulls)

    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            _dist = max(scipy.spatial.distance.cdist(hulls[i].points, hulls[j].points).min(), 1e-6)
            dist[i, j] = _dist
            dist[j, i] = _dist

    mst = scipy.sparse.csgraph.minimum_spanning_tree(dist)
    mst = np.array(mst.toarray())

    def _hash(x: list[int]):
        return hashlib.md5(str(sorted(x)).encode())
    

    
    

    _volume = {}
    def volume(subtree):
        hs = _hash(subtree)
        if hs not in _volume:
            _volume[hs] = scipy.spatial.ConvexHull(np.concatenate([hulls[i].points for i in subtree])).volume
        return _volume[hs]

    def dfs(node, vis):
        ret = []
        vis[node] = True
        subtree = [node]
        for i, connected in enumerate(mst[node]):
            if connected and not vis[i]:
                tree, trees = dfs(i, vis)
                subtree.extend(tree)
                ret.extend(trees)
        ret.append(subtree)
        ret.append([x for x in range(n) if x not in subtree])
        return subtree, ret

    def recurse(subtree):
        if len(subtree) == 1:
            return subtree
        volumn_whole = volume(subtree)
        for x in subtree:
            if volume([x]) >= volumn_whole - 1e-4:
                return (subtree, [x], recurse([y for y in subtree if y != x]))
        
        _, subtrees = dfs(subtree[0], [x not in subtree for x in range(n)])
        min_sum = float('inf')
        win_subtrees = None
        for i in range(0, len(subtrees), 2):
            left = recurse(subtrees[i])
            right = recurse(subtrees[i + 1])
            v_sum = volume(left) + volume(right)
            if v_sum < min_sum:
                min_sum = v_sum
                win_subtrees = (left, right)
        return (subtree, recurse(win_subtrees[0]), recurse(win_subtrees[1]))
    
    parsed = recurse(list(range(n)))