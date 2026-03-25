from PIL import Image, ImageOps, ImageFilter
import random
import io, os
import math
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
    # 0
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
    # Padding
    if src_width < target_width or src_height < target_height:
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(0, 0, 0)
        )
    return image

def get_start_positions(image_length, block_size, bias=0):
    if image_length <= block_size:
        return [0+bias]
    num_blocks = image_length // block_size + (image_length % block_size > 0)
    if num_blocks == 1:
        return [0+bias]
    total_stride = (image_length - block_size) / (num_blocks - 1)
    positions = [int(round(i * total_stride) + bias) for i in range(num_blocks - 1)]
    positions.append(image_length - block_size + bias)
    return positions

def crop_image_to_tiles(image, block_size):
    """
    crop to block_size x block_size
    """
    width, height = image.size
    x_positions = get_start_positions(width, block_size)
    y_positions = get_start_positions(height, block_size)
    tiles = []
    for y in y_positions:
        for x in x_positions:
            tiles.append(image.crop((x, y, x + block_size, y + block_size)))
    return tiles

def crop_image_to_tiles_new(image, block_size):
    width, height = image.size
    x_positions = get_start_positions(width, block_size)
    y_positions = get_start_positions(height, block_size)
    tiles = []
    for y in y_positions:
        for x in x_positions:
            tiles.append(image.crop((x, y, x + block_size, y + block_size)))
    if width > block_size and height > block_size:
        leave_space = min(block_size, width-block_size)
        x_positions = get_start_positions(width-leave_space, block_size, leave_space//2)
        leave_space = min(block_size, height-block_size)
        y_positions = get_start_positions(height-leave_space, block_size, leave_space//2)
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


def clip_transform_allresize(n_px, blur_radius=None, blur_prob=0.5):
    if blur_radius is None:
        return transforms.Compose([
            transforms.Resize((n_px, n_px), interpolation=BICUBIC),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((n_px, n_px), interpolation=BICUBIC),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=blur_radius)], p=blur_prob  # (0.1, 3)
            ),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

def clip_transform(n_px, blur_radius=None, blur_prob=0.5):
    if blur_radius is None:
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            _convert_image_to_rgb,
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(n_px, interpolation=BICUBIC),
            transforms.CenterCrop(n_px),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=blur_radius)], p=blur_prob  # (0.1, 3)
            ),
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
                if isinstance(self.quality, tuple):
                    img.save(buffer, format='JPEG', quality=self.local_random.randint(*self.quality))
                else:
                    img.save(buffer, format='JPEG', quality=self.quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        return img

def webp_compress(img, quality=80, method=4):
    buffer = io.BytesIO() 
    img.save(buffer, format='WEBP', quality=quality, method=method)
    buffer.seek(0)
    return Image.open(buffer).convert('RGB')

def find_files_with_extensions(directory, extensions, max_num=None):
    # find path
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

class TiledRFDataset(Dataset):
    """ Dataset -> (img, label)"""
    def __init__(self, real_dirs:list, fake_dirs:list, tile_size=224, set_size=None, shuffle_seed=42, balance=True, 
                 jpeg_augment=None, jpeg_p=0.1,
                 webp_augment=None, webp_p=0.1,
                 resize_augment=None, resize_p=0.5,
                 blur_augment=None, blur_p=0.5,
                 default_train_tiles=4, is_train=False):
        self.set_size = set_size
        self.shuffle_seed = shuffle_seed
        self.samples = []
        self.real_samples = []
        self.fake_samples = []
        self.tile_size=tile_size
        self.transform = clip_transform(tile_size, blur_augment, blur_p)
        self.default_train_tiles = default_train_tiles
        self.is_train = is_train
        
        self.jpeg_augment=jpeg_augment
        self.jpeg_p = jpeg_p
        self.JpegEnhance = JpegAugment(quality=jpeg_augment, p=jpeg_p, seed=shuffle_seed)
        self.webp_augment = webp_augment
        self.webp_p = webp_p
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
    
    def _get_train_item(self, idx):
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
                if isinstance(self.resize_augment, tuple):
                    img = resize_with_random_scale(img, p=self.resize_p, scale_range=self.resize_augment)
                else:
                    img = resize_with_random_scale(img, p=self.resize_p, scale_range=(self.resize_augment, self.resize_augment))
        if self.jpeg_augment is not None:
            img = self.JpegEnhance(img)
        
        # crops_one_side = 2
        # img = pad_crop_image(img, crops_one_side*self.tile_size, crops_one_side*self.tile_size)
        # tiles = crop_image_to_tiles(img, self.tile_size)[:crops_one_side**2]
        tiles = []
        for i in range(self.default_train_tiles):  # 10
            tiles.append(random_crop(img, self.tile_size, is_pad_black=False))
        img_resize = self.transform(img)
        img_list = [img_resize]
        for tile in tiles:
            img_list.append(self.transform(tile))
        return img_list, img, label

    def _get_test_item(self, idx):
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
                if isinstance(self.resize_augment, tuple):
                    img = resize_with_random_scale(img, p=self.resize_p, scale_range=self.resize_augment)
                else:
                    img = resize_with_random_scale(img, p=self.resize_p, scale_range=(self.resize_augment, self.resize_augment))
        if self.jpeg_augment is not None:
            img = self.JpegEnhance(img)
        if self.webp_augment is not None:
            if random.random() < self.webp_p:
                if isinstance(self.webp_augment, tuple):
                    img = webp_compress(img, quality=random.randint(*self.webp_augment))
                else:
                    img = webp_compress(img, quality=self.webp_augment)
        # testtime batchsize=1
        tiles = crop_image_to_tiles(img, self.tile_size)
        img_resize = self.transform(img)
        img_list = [img_resize]
        for tile in tiles:
            img_list.append(self.transform(tile))
        return img_list, img, label, path

    def __getitem__(self, idx):
        if self.is_train:
            return self._get_train_item(idx)
        else:
            return self._get_test_item(idx)
        

def collate_img_list_to_tensor_train(batch):
    """
    batch = list of (img_list, label)
    - img_list: List of N Tensor[B, 3, H, W]
    return：
    - images: Tensor[B, N, 3, H, W]
    - labels: Tensor[B]
    """
    img_lists, raw_imgs, labels = zip(*batch)  # len = B
    # img_lists: list of length B, each is List[N x (3,H,W)]
    stacked = torch.stack([torch.stack(imgs) for imgs in img_lists])  # (B, N, 3, H, W)
    raw_imgs = list(raw_imgs)
    labels = torch.tensor(labels)
    return stacked, raw_imgs, labels

def collate_img_list_to_tensor_test(batch):
    """
    输入：batch = list of (img_list, label)
    - img_list: List of N Tensor[B, 3, H, W]
    return：
    - images: Tensor[B, N, 3, H, W]
    - labels: Tensor[B]
    """
    img_lists, raw_imgs, labels, paths = zip(*batch)  # len = B
    # img_lists: list of length B, each is List[N x (3,H,W)]
    stacked = torch.stack([torch.stack(imgs) for imgs in img_lists])  # (B, N, 3, H, W)
    raw_imgs = list(raw_imgs)
    labels = torch.tensor(labels)
    return stacked, raw_imgs, labels, paths


if __name__ == '__main__':
    block_size = 224
    width = 512
    leave_space = min(block_size, width-block_size)
    x_positions = get_start_positions(width-leave_space, block_size, leave_space//2)
    print(leave_space, x_positions)

