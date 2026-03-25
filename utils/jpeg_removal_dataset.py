from PIL import Image, ImageOps
import random
import io, os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC


def resize_with_random_scale(image: Image.Image, p: float = 0.5, scale_range=(0.25, 2.0)) -> Image.Image:
    if random.random() > p:
        return image 

    w, h = image.size
    scale = random.uniform(*scale_range)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return image.resize((new_w, new_h), resample=Image.BICUBIC)

def pad_crop_image(image, target_width, target_height):
    src_width, src_height = image.size
    pad_left = pad_right = pad_top = pad_bottom = 0
    crop_left = crop_top = 0

    if src_width < target_width:
        pad_left = random.randint(0, target_width - src_width)
        pad_right = target_width - src_width - pad_left
    else:
        crop_left = random.randint(0, src_width - target_width)

    if src_height < target_height:
        pad_top = random.randint(0, target_height - src_height)
        pad_bottom = target_height - src_height - pad_top
    else:
        crop_top = random.randint(0, src_height - target_height)

    if src_width > target_width or src_height > target_height:
        left = crop_left if src_width > target_width else 0
        top = crop_top if src_height > target_height else 0
        right = left + target_width
        bottom = top + target_height
        image = image.crop((left, top, right, bottom))
    if src_width < target_width or src_height < target_height:
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0)
        )
    return image

def get_start_positions(image_length, block_size):
    if image_length <= block_size:
        return [0]
    num_blocks = image_length // block_size + (image_length % block_size > 0)
    if num_blocks == 1:
        return [0]
    total_stride = (image_length - block_size) / (num_blocks - 1)
    positions = [int(round(i * total_stride)) for i in range(num_blocks - 1)]
    positions.append(image_length - block_size)
    return positions

def crop_image_to_tiles(image, block_size):
    width, height = image.size
    x_positions = get_start_positions(width, block_size)
    y_positions = get_start_positions(height, block_size)
    tiles = []
    for y in y_positions:
        for x in x_positions:
            tiles.append(image.crop((x, y, x + block_size, y + block_size)))
    return tiles

def random_crop(image: Image.Image, size: int = 224, is_pad_black: bool = False) -> Image.Image:
    w, h = image.size
    crop_size = min(w, h, size)
    if w == crop_size:
        left = 0
    else:
        left = random.randint(0, w - crop_size)
    if h == crop_size:
        top = 0
    else:
        top = random.randint(0, h - crop_size)

    cropped = image.crop((left, top, left + crop_size, top + crop_size))
    if is_pad_black and crop_size < size:
        padded_img = Image.new("RGB", (size, size), (0, 0, 0))
        offset = ((size - crop_size) // 2, (size - crop_size) // 2)
        padded_img.paste(cropped, offset)
        return padded_img
    else:
        return cropped

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def clip_transform(n_px):
    return transforms.Compose([
        transforms.Resize(n_px, interpolation=BICUBIC),
        transforms.CenterCrop(n_px),
        _convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

class JpegAugment():
    def __init__(self, quality=None, p=1, seed=42):
        self.local_random = random.Random(seed)
        self.quality = quality
        self.p = p
    
    def __call__(self, img):
        if self.quality is None:
            return img
        if self.local_random.random() < self.p:
           
            buffer = io.BytesIO()
            if self.quality == -1:
                img.save(buffer, format='JPEG', quality=self.local_random.randint(30, 100))
            else:
                img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img

def find_files_with_extensions(directory, extensions, max_num=None):
    matching_files = []
    check_num = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(tuple(extensions)):
                matching_files.append(os.path.join(root, file))
                check_num += 1
                if max_num is not None:
                    if check_num >= max_num:
                        return matching_files
    return matching_files

# from my_utils.network_fbcnn import FBCNN as net
from utils import FBCNN_image as fbcnn_util

def tensor2pil(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))  # CHW -> HWC
    img = (img * 255.0).round().astype(np.uint8)
    return Image.fromarray(img)

class JpegRemovalDataset():
    def __init__(self, real_dirs:list, fake_dirs:list, tile_size=224, set_size=None, shuffle_seed=42, balance=True, 
                 jpeg_augment=None, jpeg_p=0.1, resize_augment=None, resize_p=0.5, default_train_tiles=4, is_train=False,
                 fbcnn_model=None, device=None,
                 ):
        self.set_size = set_size
        self.shuffle_seed = shuffle_seed
        self.samples = []
        self.real_samples = []
        self.fake_samples = []
        self.tile_size=tile_size
        self.transform = clip_transform(tile_size)
        self.default_train_tiles = default_train_tiles
        self.is_train = is_train

        self.fbcnn = fbcnn_model
        self.device = device
        
        self.jpeg_augment=jpeg_augment
        self.jpeg_p = jpeg_p
        self.JpegEnhance = JpegAugment(quality=jpeg_augment, p=jpeg_p, seed=shuffle_seed)
        self.resize_augment = resize_augment
        self.resize_p = resize_p

        for p in real_dirs:
            find_paths_real = find_files_with_extensions(p, ['.jpg', '.png', '.JPEG', '.PNG', '.jpeg', '.JPEG'])
            self.real_samples += [(f, 0) for f in find_paths_real]
        for p in fake_dirs:
            find_paths_fake = find_files_with_extensions(p, ['.jpg', '.png', '.JPEG', '.PNG', '.jpeg', '.JPEG'])
            self.fake_samples += [(f, 1) for f in find_paths_fake]
        if balance:
            target_len = min(len(self.real_samples), len(self.fake_samples))
            self.real_samples = self.real_samples[:target_len]
            self.fake_samples = self.fake_samples[:target_len]
        self.samples = self.real_samples + self.fake_samples
        # if self.set_size is not None and len(self.samples) < self.set_size:
        #     n_repeats = math.ceil(self.set_size / len(self.samples))
        #     self.samples = (self.samples * n_repeats)[:self.set_size]
        if self.set_size is not None and len(self.samples) > self.set_size:
            local_random = random.Random(self.shuffle_seed)
            self.samples = local_random.sample(self.samples, self.set_size)
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupted image: {path}")
            return self.__getitem__((idx + 1) % len(self.samples))
        
        if self.resize_augment is not None:
            if self.resize_augment == -1:
                img = resize_with_random_scale(img, p=self.resize_p)
            else:
                img = resize_with_random_scale(img, p=self.resize_p, scale_range=(self.resize_augment, self.resize_augment))
        if self.jpeg_augment is not None:
            img = self.JpegEnhance(img)

        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img_tensor = fbcnn_util.uint2tensor4(img_cv).to(self.device)
        with torch.no_grad():
            img_E, QF = self.fbcnn(img_tensor)
        QF = float((1 - QF).cpu().numpy())

        img_dejpeg = tensor2pil(img_E)

        tiles = []
        for i in range(self.default_train_tiles):  # 2
            tiles.append(random_crop(img, self.tile_size, is_pad_black=False))
        img = self.transform(img)
        img_list_o = [img]
        for tile in tiles:
            img_list_o.append(self.transform(tile))

        tiles = []
        for i in range(self.default_train_tiles):  # 2
            tiles.append(random_crop(img_dejpeg, self.tile_size, is_pad_black=False))
        img_dejpeg = self.transform(img_dejpeg)
        img_list = [img_dejpeg]
        for tile in tiles:
            img_list.append(self.transform(tile))
        return img_list_o, img_list, QF, label

class ManualDataLoader:
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._create_epoch_indices()

    def _create_epoch_indices(self):
        self.indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(self.indices)
        self.batches = [
            self.indices[i:i + self.batch_size]
            for i in range(0, len(self.indices), self.batch_size)
        ]

    def new_epoch(self):
        self.epoch += 1
        self._create_epoch_indices()

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch_indices in self.batches:
            batch = [self.dataset[i] for i in batch_indices]

            batch_o_list = [sample[0] for sample in batch]        # list of [1+n] tensors
            batch_dejpeg_list = [sample[1] for sample in batch]   # list of [1+n] tensors
            qf_list = [sample[2] for sample in batch]
            labels = [sample[3] for sample in batch]

            # list[list[tensor]] -> tensor: [B, 1+n, C, H, W]
            def stack_image_lists(image_lists):
                return torch.stack([
                    torch.stack(img_set, dim=0)  # [1+n, C, H, W]
                    for img_set in image_lists
                ], dim=0)  # -> [B, 1+n, C, H, W]

            batch_o_tensor = stack_image_lists(batch_o_list)
            batch_dejpeg_tensor = stack_image_lists(batch_dejpeg_list)

            # [B, 1]
            yield batch_o_tensor, batch_dejpeg_tensor, torch.tensor(qf_list, dtype=torch.float32).unsqueeze(1), torch.tensor(labels)

if __name__ == '__main__':
    from dataset_source import get_Genimage_path
    train_reals, train_fakes = get_Genimage_path()

    dataset = JpegRemovalDataset(train_reals[0], train_fakes[0])
    loader = ManualDataLoader(dataset, 4, shuffle=True)
    for batch_o, batch_dejpeg, qf_list, labels in loader:
        print(batch_o.shape, batch_dejpeg.shape, qf_list, labels)
    loader.new_epoch()
    pass

