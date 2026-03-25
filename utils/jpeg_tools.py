from PIL import Image
import numpy as np
from collections import OrderedDict

STD_LUMINANCE_QTABLE = np.array([
    16,11,10,16,24,40,51,61,
    12,12,14,19,26,58,60,55,
    14,13,16,24,40,57,69,56,
    14,17,22,29,51,87,80,62,
    18,22,37,56,68,109,103,77,
    24,35,55,64,81,104,113,92,
    49,64,78,87,103,121,120,101,
    72,92,95,98,112,100,103,99
])

quant_table_cache = {}

def scale_quant_table(base_table, quality):
    if quality in quant_table_cache:
        return quant_table_cache[quality]

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2
    scaled = np.floor((base_table * scale + 50) / 100)
    scaled = np.clip(scaled, 1, 255)
    result = scaled.astype(int)

    quant_table_cache[quality] = result
    return result

def extract_qtables(img):
    # img = Image.open(image_path)
    qtables = img.quantization
    return qtables

class LRUCache:
    def __init__(self, maxsize=100):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.maxsize:
            self.cache.popitem(last=False)

quality_estimation_cache = LRUCache(maxsize=256)

def estimate_quality(img):
    qtables = extract_qtables(img)
    if 0 not in qtables:
        return None
    
    img_qtable = np.array(qtables[0])
    qtable_key = img_qtable.tobytes()

    cached = quality_estimation_cache.get(qtable_key)
    if cached is not None:
        return cached

    min_error = float('inf')
    best_quality = None
    for q in range(1, 101):
        std_qtable = scale_quant_table(STD_LUMINANCE_QTABLE, q)
        error = np.sum((img_qtable - std_qtable) ** 2)
        if error < min_error:
            min_error = error
            best_quality = q

    quality_estimation_cache.put(qtable_key, best_quality)
    return best_quality