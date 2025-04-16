import argparse
from PIL import Image
import scipy
import numpy as np
import hashlib
import openai
from io import BytesIO
import base64
import json
from gradio_client import Client, handle_file
import tempfile
import os


def depth_estimator(image):
    client = Client("http://localhost:7860/")
    if isinstance(image, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name
        try:
            result = client.predict(
                    input_image=handle_file(image_path),
                    api_name="/predict"
            )
        finally:
            os.remove(image_path)  # Clean up the temporary file
    else: # Assume it's already a path or URL if not a PIL Image
         result = client.predict(
                    input_image=handle_file(image),
                    api_name="/predict"
            )
    return result

def sam2(image):
    client = Client("http://localhost:7861/")
    if isinstance(image, Image.Image):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name)
            image_path = temp_file.name
        try:
            result = client.predict(
                    input_image=handle_file(image_path),
                    api_name="/predict"
            )
        finally:
            os.remove(image_path) # Clean up the temporary file
    else: # Assume it's already a path or URL if not a PIL Image
        result = client.predict(
                input_image=handle_file(image),
                api_name="/predict"
        )
    return result

def i2t(image):
    print(type(image))
    if isinstance(image, bytes):
        image = Image.open(BytesIO(image))
    image = image.convert('RGB')

    masks = sam2(image)
    depth = depth_estimator(image)
    
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
        if hs in _volume:
            return _volume[hs]
        hull = scipy.spatial.ConvexHull(np.concatenate([hulls[i].points for i in subtree]))
        ret = hull.volume, hull.points
        _volume[hs] = ret
        return ret

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
        volumn_whole, _ = volume(subtree)
        for x in subtree:
            volumn_x, _ = volume([x])
            if volumn_x >= volumn_whole - 1e-4:
                return (subtree, [x], recurse([y for y in subtree if y != x]))
        
        _, subtrees = dfs(subtree[0], [x not in subtree for x in range(n)])
        min_sum = float('inf')
        win_subtrees = None
        win_hulls = None
        for i in range(0, len(subtrees), 2):
            v1, h1 = volume(subtrees[i])
            v2, h2 = volume(subtrees[i + 1])
            if v1 + v2 < min_sum:
                min_sum = v1 + v2
                win_subtrees = (subtrees[i], subtrees[i + 1])
                win_hulls = (h1, h2)
        
        center1 = np.average(win_hulls[0], axis=0)
        center2 = np.average(win_hulls[1], axis=0)
        dir = center2 - center1
        return (subtree, recurse(win_subtrees[0]), recurse(win_subtrees[1]), dir)
    
    parsed = recurse(list(range(n)))

    def bbox(mask):
        arg = np.argwhere(mask)
        return [arg[:, 0].min(), arg[:, 1].min(), arg[:, 0].max(), arg[:, 1].max()]

    def merged_mask(subtree):
        mask = np.zeros((image.width, image.height), dtype=np.uint8)
        for i in subtree:
            mask |= masks[i]
        return mask
    
    def construct(semantic_tree):
        caption = mask_caption(image, merged_mask(semantic_tree[0]))
        if len(semantic_tree) == 1:
            return caption
        return {'caption': caption, 'child1': construct(semantic_tree[1]), 'child2': construct(semantic_tree[2]), 'vector': semantic_tree[3]}
    
    import os
    import datetime
    dirname = f'i2t_results'
    os.makedirs(dirname, exist_ok=True)
    with open(f'{dirname}/{hashlib.md5(str(image).encode()).hexdigest()}.json', 'w') as f:
        json.dump(parsed, f)
    for i, mask in enumerate(masks):
        mask.save(f'{dirname}/mask_{i}.png')
    return construct(parsed)

def extract_brace(x: str):
    import re
    match = re.search(r"{(.*?)}", x)
    if not match:
        return None
    return match.group(1)

def test_extract_brace():
    assert extract_brace(r'{hello}') == 'hello'
    assert extract_brace(r'{hello} world') == 'hello'
    assert extract_brace(r'hello {world}') == 'world'
    assert extract_brace(r'hello {world} {hello}') == 'world'
    assert extract_brace(r'hello') == None

def mask_caption(image, mask):
    masked = image.copy()
    masked.putalpha(mask * 255)

    bbox = bbox(mask)
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

    return extract_brace(response) or response
    
    
def vqa(image, question):
    semantic_tree = i2t(image)
    return openai.chat.completions.create(
        model='deepseek-reasoner',
        messages=[
            {'role': 'system', 'content': 'You are a agent that can understand image based on its semantic tree, and answer vqa questions.'},
            {'role': 'user', 'content': (f'''
this is the semantic tree of an image. 
each subtree represent a part of the image, and its 2 children subparts of it, forming a hierarchical understanding of the image. 
each subtree has a caption, two children, and a 3d vector pointing from center of child1 to center of child2, to give you an understanding of the spatial relationship between the two.
subtree caption is a caption of the whole subtree, so it may duplicate the caption of its children.
{json.dumps(semantic_tree)}

now, answer the question based on the semantic tree.
question: {question})
'''
r'put your final answer in curly braces like {your_answer}.')
}
        ]
    ).choices[0].message.content

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--question', type=str, required=True)
    args = parser.parse_args()
    print(vqa(args.image, args.question))
