import argparse
from PIL import Image
import scipy
import numpy as np
import hashlib
import openai
from io import BytesIO
import base64
import re
def i2t(image_path: str):
    image = Image.open(image_path)
    image = image.convert('RGB')

    masks = sam2postprocess(sam2.automatic_mask_generator.generate(image))
    depth = depth_estimator.predict(image)
    
    three_d = [
        [(x, y, depth[x, y]) for x, y in np.argwhere(mask)]
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

    def merged_mask(subtree):
        mask = np.zeros((image.width, image.height), dtype=np.uint8)
        for i in subtree:
            mask |= masks[i]
        return mask
    
    
def mask_caption(image, mask, bbox):
    masked = image.copy()
    masked.putalpha(mask * 255)

    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    width = (bbox[2] - bbox[0]) / 2
    height = (bbox[3] - bbox[1]) / 2

    cropping_cascade = []
    while True:
        bbox = [max(0, center[0] - width), max(0, center[1] - height), min(image.width, center[0] + width), min(image.height, center[1] + height)]
        cropping_cascade.append(image.crop(bbox).thumbnail((min(7 * width, 512), min(7 * height, 512))))
        if bbox == [0, 0, image.width, image.height]:
            break
        width *= 2
        height *= 2

    def image_to_base64(image):
        buffered = BytesIO()
        image.save(buffered, format='PNG')
        return f'data:image/png;base64,{base64.b64encode(buffered.getvalue()).decode()}'
    
    response = openai.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant that captions masks.'},
            {'role': 'user', 'content': [{'type': 'text', 'text': (
                'please caption the masked area. notice, only the 1st image is the object you should describe. the other images are just for context, and should not be described.'
                r'if you need time to reason, place your final caption in curly braces like {your_caption}.'
            )},
            *[{'type': 'image_url', 'image_url': {'url': image_to_base64(cropped)}} for cropped in cropping_cascade]]}
        ]
    ).choices[0].message.content

    return re.search(r'{.*}', response).group(0) or response
    
    


