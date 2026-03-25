from PIL import Image, ImageOps, ImageFilter
import random
import io, os
import numpy as np
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

def get_start_positions(image_length, block_size):
    """Compute start pos"""
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
    """
    crop to multiple block_size x block_size block
    """
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
    
def random_aligned_crop(img, mask, size):
    w, h = img.size
    th, tw = size, size
    
    if w < tw or h < th:
        img_cropped = img.resize((tw, th), Image.LANCZOS)
        if mask:
            mask_cropped = mask.resize((tw, th), Image.NEAREST)
        else:
            mask_cropped = None
        return img_cropped, mask_cropped

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)
    
    img_cropped = img.crop((x1, y1, x1 + tw, y1 + th))
    if mask:
        mask_cropped = mask.crop((x1, y1, x1 + tw, y1 + th))
    else:
        mask_cropped = None
        
    return img_cropped, mask_cropped

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
                # print("!"*50, self.quality, type(self.quality))
                if isinstance(self.quality, tuple) or isinstance(self.quality, list):
                    img.save(buffer, format='JPEG', quality=self.local_random.randint(*self.quality))
                else:
                    # print("*"*50, self.quality)
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

class TiledRFDataset_paired(Dataset):
    def __init__(self, real_dirs:list, fake_dirs:list, tile_size=224, set_size=None, shuffle_seed=42,
                 jpeg_augment=None, jpeg_p=0.1,
                 webp_augment=None, webp_p=0.1,
                 resize_augment=None, resize_p=0.5,
                 blur_augment=None, blur_p=0.5,
                 default_train_tiles=4, is_train=False,
                 replace_p=0.1,
                 replace_ratio=(0.2, 0.98),
                 replace_grid=8,
                 vit_patch_size=14,):
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
        self.replace_p = replace_p
        self.replace_ratio = replace_ratio
        self.replace_grid = replace_grid

        self.vit_patch_size = vit_patch_size
        self.vit_patch_count = tile_size // vit_patch_size

        self.pairs = []
        if self.set_size is not None:
            self.pair_size = self.set_size // 2
        else:
            self.pair_size = None
        # extensions=['.jpg', '.png', '.JPEG', '.PNG', '.jpeg', '.JPEG']
        if len(real_dirs) != len(fake_dirs):
            raise ValueError("Real and Fake folder numbers must be matched.")
        for real_path, fake_path in zip(real_dirs, fake_dirs):
            real_files = set(os.listdir(real_path))
            fake_files = set(os.listdir(fake_path))
            # print(real_path, len(real_files), fake_path, len(fake_files))
            real_map = {os.path.splitext(f)[0]: f for f in real_files}
            fake_map = {os.path.splitext(f)[0]: f for f in fake_files}

            common_keys = real_map.keys() & fake_map.keys()
            # keep extension name
            real_common = [real_map[k] for k in common_keys]
            fake_common = [fake_map[k] for k in common_keys]
            if len(real_common) != 0:
                for r_file, f_file in zip(real_common, fake_common):
                    self.pairs.append((
                        os.path.join(real_path, r_file),
                        os.path.join(fake_path, f_file)
                    ))
            else:
                r_imgs = find_files_with_extensions(real_path, ['.jpg', '.png', '.JPEG', '.PNG', '.jpeg', '.JPEG'])
                f_imgs = find_files_with_extensions(fake_path, ['.jpg', '.png', '.JPEG', '.PNG', '.jpeg', '.JPEG'])
                for r_file, f_file in zip(r_imgs, f_imgs):
                    self.pairs.append((
                        os.path.join(real_path, r_file),
                        os.path.join(fake_path, f_file)
                    ))
        if self.pair_size is not None and len(self.pairs) > self.pair_size:
            # local random
            local_random = random.Random(self.shuffle_seed)
            self.pairs = local_random.sample(self.pairs, self.pair_size)

    def __len__(self):
        return len(self.pairs)

    def generate_soft_patch_labels(self, mask_224: Image.Image) -> torch.Tensor:
        """get 16x16 soft label"""
        if mask_224 is None:
            return None # 稍后处理

        # 输出为 float 类型
        patch_labels = torch.zeros((self.vit_patch_count, self.vit_patch_count), dtype=torch.float32)
        patch_size = self.tile_size // self.vit_patch_count

        mask_array = np.array(mask_224)

        for i in range(self.vit_patch_count):
            for j in range(self.vit_patch_count):
                patch = mask_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                label_value = np.mean(patch) / 255.0
                patch_labels[i, j] = label_value
        
        return patch_labels.flatten() # (256,)

    def __getitem__(self, idx):
        real_path, fake_path = self.pairs[idx]
        try:
            real_img = Image.open(real_path).convert('RGB')
            fake_img = Image.open(fake_path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupted image: {real_path} {fake_path}")
            return self.__getitem__((idx + 1) % len(self.pairs))

        augmented_mask = None
        if random.random() < self.replace_p:
            #  self.resize_p 
            if self.resize_augment is not None:
                augmented_fake_img = resize_with_random_scale(fake_img, p=self.resize_p, scale_range=self.resize_augment)
            else:
                augmented_fake_img = fake_img
            resized_real_img = real_img.resize(augmented_fake_img.size, Image.LANCZOS)

            #  mask
            w, h = augmented_fake_img.size
            grid_size = self.replace_grid
            grid_w, grid_h = max(1, w // grid_size), max(1, h // grid_size)
            num_replace = int(grid_w * grid_h * random.uniform(*self.replace_ratio))
            replace_indices = np.random.choice(grid_w * grid_h, num_replace, replace=False)
            grid_mask = np.ones(grid_w * grid_h, dtype=bool)
            grid_mask[replace_indices] = False
            grid_mask = grid_mask.reshape((grid_h, grid_w))
            augmented_mask = Image.fromarray(grid_mask).resize((w, h), Image.NEAREST).point(lambda p: 255 if p else 0, 'L')
            
            fake_img = Image.composite(resized_real_img, augmented_fake_img, augmented_mask)

        if self.jpeg_augment is not None:
            real_img = self.JpegEnhance(real_img)
            fake_img = self.JpegEnhance(fake_img)
        
        # crop to tiles
        real_tiles, fake_tiles = [], []
        real_patch_labels, fake_patch_labels = [], []


        real_img_resized = real_img.resize((self.tile_size, self.tile_size), Image.LANCZOS)
        real_patch_labels.append(torch.zeros(self.vit_patch_count**2, dtype=torch.float32))

        fake_img_resized = fake_img.resize((self.tile_size, self.tile_size), Image.LANCZOS)
        if augmented_mask:
            mask_resized = augmented_mask.resize((self.tile_size, self.tile_size), Image.BILINEAR)
            fake_patch_labels.append(self.generate_soft_patch_labels(mask_resized))
        else:
            fake_patch_labels.append(torch.ones(self.vit_patch_count**2, dtype=torch.float32))
            
        for _ in range(self.default_train_tiles):
            # Real crop 
            real_tile_img, _ = random_aligned_crop(real_img, None, self.tile_size)
            real_tiles.append(real_tile_img)
            real_patch_labels.append(torch.zeros(self.vit_patch_count**2, dtype=torch.float32))
            
            # Fake crop 
            fake_tile_img, mask_crop = random_aligned_crop(fake_img, augmented_mask, self.tile_size)
            fake_tiles.append(fake_tile_img)
            if mask_crop:
                fake_patch_labels.append(self.generate_soft_patch_labels(mask_crop))
            else:
                fake_patch_labels.append(torch.ones(self.vit_patch_count**2, dtype=torch.float32))

        # transform
        img_list_real = [self.transform(real_img_resized)] + [self.transform(t) for t in real_tiles]
        img_list_fake = [self.transform(fake_img_resized)] + [self.transform(t) for t in fake_tiles]
        
        real_patch_labels_tensor = torch.stack(real_patch_labels)
        fake_patch_labels_tensor = torch.stack(fake_patch_labels)

        return (img_list_real, real_img, 0, real_patch_labels_tensor), \
               (img_list_fake, fake_img, 1, fake_patch_labels_tensor)

def collate_img_list_to_tensor_paried_train(batch):
    real_lists, fake_lists = [], []
    real_raws, fake_raws = [], []
    real_labels, fake_labels = [], []
    real_patch_labels_list, fake_patch_labels_list = [], []

    for (real_item, fake_item) in batch:
        real_imgs, real_raw, real_label, real_patch_labels = real_item
        fake_imgs, fake_raw, fake_label, fake_patch_labels = fake_item
        
        real_lists.append(torch.stack(real_imgs))
        fake_lists.append(torch.stack(fake_imgs))
        real_raws.append(real_raw)
        fake_raws.append(fake_raw)
        real_labels.append(real_label)
        fake_labels.append(fake_label)
        real_patch_labels_list.append(real_patch_labels)
        fake_patch_labels_list.append(fake_patch_labels)

    all_images = real_lists + fake_lists
    all_raws = real_raws + fake_raws
    all_labels = real_labels + fake_labels
    all_patch_labels = real_patch_labels_list + fake_patch_labels_list

    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)
    patch_labels_tensor = torch.stack(all_patch_labels)

    return images_tensor, all_raws, labels_tensor, patch_labels_tensor


class TiledRFDataset_Augmented_Paired(Dataset):
    def __init__(self, real_dirs:list, fake_dirs:list, tile_size=224, set_size=None, shuffle_seed=42,
                 resize_augment=None, resize_p=0.5,
                 blur_augment=None, blur_p=0.5,
                 default_train_tiles=4,
                 augment_p=0.5,
                 replace_ratio=0.3,
                 vit_patch_size=14):
        
        self.tile_size = tile_size
        self.transform = clip_transform(tile_size, blur_augment, blur_p)
        self.default_train_tiles = default_train_tiles
        
        
        self.resize_augment = resize_augment
        self.resize_p = resize_p 
        self.augment_p = augment_p
        self.replace_ratio = replace_ratio
        self.vit_patch_count = tile_size // vit_patch_size


        self.pairs = []
        if set_size is not None:
            self.pair_size = set_size // 2
        else:
            self.pair_size = None

        if len(real_dirs) != len(fake_dirs):
            raise ValueError("Real and Fake folder numbers must be matched.")
        for real_path, fake_path in zip(real_dirs, fake_dirs):
            real_files, fake_files = set(os.listdir(real_path)), set(os.listdir(fake_path))
            real_map, fake_map = {os.path.splitext(f)[0]: f for f in real_files}, {os.path.splitext(f)[0]: f for f in fake_files}
            common_keys = real_map.keys() & fake_map.keys()
            for k in common_keys:
                self.pairs.append((os.path.join(real_path, real_map[k]), os.path.join(fake_path, fake_map[k])))
        
        if self.pair_size is not None and len(self.pairs) > self.pair_size:
            random.Random(shuffle_seed).shuffle(self.pairs)
            self.pairs = self.pairs[:self.pair_size]

    def __len__(self):
        return len(self.pairs)

    def generate_soft_patch_labels(self, mask_224: Image.Image) -> torch.Tensor:
        if mask_224 is None:
            return None 

        # float
        patch_labels = torch.zeros((self.vit_patch_count, self.vit_patch_count), dtype=torch.float32)
        patch_size = self.tile_size // self.vit_patch_count

        mask_array = np.array(mask_224)

        for i in range(self.vit_patch_count):
            for j in range(self.vit_patch_count):
                patch = mask_array[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
                # mask  0=real, 255=fake
                label_value = np.mean(patch) / 255.0
                patch_labels[i, j] = label_value
        
        return patch_labels.flatten() # (256,)

    def __getitem__(self, idx):
        real_path, fake_path = self.pairs[idx]
        try:
            real_img = Image.open(real_path).convert('RGB')
            fake_img = Image.open(fake_path).convert('RGB')
        except Exception:
            return self.__getitem__((idx + 1) % len(self.pairs))

        augmented_mask = None
        if random.random() < self.augment_p:
            augmented_fake_img = resize_with_random_scale(fake_img, p=self.resize_p)
            resized_real_img = real_img.resize(augmented_fake_img.size, Image.LANCZOS)
            
            w, h = augmented_fake_img.size
            grid_size = 32
            grid_w, grid_h = max(1, w // grid_size), max(1, h // grid_size)
            num_replace = int(grid_w * grid_h * self.replace_ratio)
            replace_indices = np.random.choice(grid_w * grid_h, num_replace, replace=False)
            grid_mask = np.ones(grid_w * grid_h, dtype=bool)
            grid_mask[replace_indices] = False
            grid_mask = grid_mask.reshape((grid_h, grid_w))
            augmented_mask = Image.fromarray(grid_mask).resize((w, h), Image.NEAREST).point(lambda p: 255 if p else 0, 'L')
            
            fake_img = Image.composite(resized_real_img, augmented_fake_img, augmented_mask)

        real_tiles, fake_tiles = [], []
        real_patch_labels, fake_patch_labels = [], []

        real_img_resized = real_img.resize((self.tile_size, self.tile_size), Image.LANCZOS)
        real_patch_labels.append(torch.zeros(self.vit_patch_count**2, dtype=torch.float32))

        fake_img_resized = fake_img.resize((self.tile_size, self.tile_size), Image.LANCZOS)
        if augmented_mask:
            mask_resized = augmented_mask.resize((self.tile_size, self.tile_size), Image.NEAREST)
            fake_patch_labels.append(self.generate_soft_patch_labels(mask_resized))
        else:
            fake_patch_labels.append(torch.ones(self.vit_patch_count**2, dtype=torch.float32))
            
        for _ in range(self.default_train_tiles):
            # Real crop 
            real_tile_img, _ = random_aligned_crop(real_img, None, self.tile_size)
            real_tiles.append(real_tile_img)
            real_patch_labels.append(torch.zeros(self.vit_patch_count**2, dtype=torch.float32))
            
            # Fake crop 
            fake_tile_img, mask_crop = random_aligned_crop(fake_img, augmented_mask, self.tile_size)
            fake_tiles.append(fake_tile_img)
            if mask_crop:
                fake_patch_labels.append(self.generate_soft_patch_labels(mask_crop))
            else:
                fake_patch_labels.append(torch.ones(self.vit_patch_count**2, dtype=torch.float32))

        # transform
        img_list_real = [self.transform(real_img_resized)] + [self.transform(t) for t in real_tiles]
        img_list_fake = [self.transform(fake_img_resized)] + [self.transform(t) for t in fake_tiles]
        
        real_patch_labels_tensor = torch.stack(real_patch_labels)
        fake_patch_labels_tensor = torch.stack(fake_patch_labels)

        return (img_list_real, real_img, 0, real_patch_labels_tensor), \
               (img_list_fake, fake_img, 1, fake_patch_labels_tensor)

def collate_img_list_to_tensor_paried_train(batch):
    real_lists, fake_lists = [], []
    real_raws, fake_raws = [], []
    real_labels, fake_labels = [], []
    real_patch_labels_list, fake_patch_labels_list = [], []

    for (real_item, fake_item) in batch:
        real_imgs, real_raw, real_label, real_patch_labels = real_item
        fake_imgs, fake_raw, fake_label, fake_patch_labels = fake_item
        
        real_lists.append(torch.stack(real_imgs))
        fake_lists.append(torch.stack(fake_imgs))
        real_raws.append(real_raw)
        fake_raws.append(fake_raw)
        real_labels.append(real_label)
        fake_labels.append(fake_label)
        real_patch_labels_list.append(real_patch_labels)
        fake_patch_labels_list.append(fake_patch_labels)

    all_images = real_lists + fake_lists
    all_raws = real_raws + fake_raws
    all_labels = real_labels + fake_labels
    all_patch_labels = real_patch_labels_list + fake_patch_labels_list

    images_tensor = torch.stack(all_images)
    labels_tensor = torch.tensor(all_labels)
    patch_labels_tensor = torch.stack(all_patch_labels)

    return images_tensor, all_raws, labels_tensor, patch_labels_tensor

if __name__ == '__main__':
    pass

