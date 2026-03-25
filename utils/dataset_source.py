import os
from os.path import join as osp

def get_AIGCDetect_path():
    dataset_path = "/path/to/AIGC_Dataset/AIGCDetect/test/"
    ordered_types = ['progan', 'stylegan', 'biggan', 'cyclegan', 'stargan', 'gaugan', 'stylegan2', 'whichfaceisreal', 'ADM', 'Glide', 'Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'VQDM', 'wukong', 'DALLE2']
    real_dir = []
    fake_dir = []
    for type in ordered_types:
        sub_folders = os.listdir(osp(dataset_path, type))
        if '0_real' in sub_folders and '1_fake' in sub_folders:
            real_dir.append([osp(dataset_path, type, '0_real')])
            fake_dir.append([osp(dataset_path, type, '1_fake')])
        else:
            temp_real = []
            temp_fake = []
            for sub_folder in sub_folders:
                temp_real.append(osp(dataset_path, type, sub_folder, '0_real'))
                temp_fake.append(osp(dataset_path, type, sub_folder, '1_fake'))
            real_dir.append(temp_real)
            fake_dir.append(temp_fake)
    return real_dir, fake_dir

def get_Genimage_path():
    dataset_path = "/path/to/AIGC_Dataset/GenImage/mix/eval/"
    ordered_types = ['Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'ADM', 'glide', 'wukong', 'VQDM', 'BigGAN']
    real_dir = []
    fake_dir = []
    for type in ordered_types:
        real_dir.append([osp(dataset_path, type, '0_real')])
        fake_dir.append([osp(dataset_path, type, '1_fake')])
    return real_dir, fake_dir

def get_Genimage_train_path():
    dataset_path = "/path/to/AIGC_Dataset/GenImage/mix/train/"
    ordered_types = ['Midjourney', 'stable_diffusion_v_1_4', 'stable_diffusion_v_1_5', 'ADM', 'glide', 'wukong', 'VQDM', 'BigGAN']
    real_dir = []
    fake_dir = []
    for type in ordered_types:
        real_dir.append(osp(dataset_path, type, '0_real'))
        fake_dir.append(osp(dataset_path, type, '1_fake'))
    return [real_dir], [fake_dir]

def get_Genimage_SD14_train():
    real_dir = [["/path/to/AIGC_Dataset/My_ImageNet/0_real"]]
    fake_dir = [["/path/to/AIGC_Dataset/GenImage/mix/train/stable_diffusion_v_1_4/1_fake/"]]
    return real_dir, fake_dir

def get_Genimage_SD14_pure_train():
    real_dir = [["/path/to/AIGC_Dataset/GenImage/mix/train/stable_diffusion_v_1_4/0_real/"]]
    fake_dir = [["/path/to/AIGC_Dataset/GenImage/mix/train/stable_diffusion_v_1_4/1_fake/"]]
    return real_dir, fake_dir

def get_Genimage_SD14_pure_test():
    real_dir = [["/path/to/AIGC_Dataset/GenImage/mix/eval/stable_diffusion_v_1_4/0_real/"]]
    fake_dir = [["/path/to/AIGC_Dataset/GenImage/mix/eval/stable_diffusion_v_1_4/1_fake/"]]
    return real_dir, fake_dir

def get_Genimage_SD14_pure_train_rec():
    real_dir = [["/path/to/AIGC_Dataset/GenImage/mix/train/stable_diffusion_v_1_4/0_real/"]]
    fake_dir = [["/path/to/AIGC_Dataset/GenImage/mix/train/stable_diffusion_v_1_4/my_sd14_rec/"]]
    return real_dir, fake_dir

def get_aeroblade_path(all_togeter=False):
    dataset_path = "/path/to/AIGC_Dataset/aeroblade/data/raw/"
    use_real  = "/path/to/AIGC_Dataset/aeroblade/data/raw/real/"
    use_fake = "/path/to/AIGC_Dataset/aeroblade/data/raw/generated/"
    real_dir = []
    fake_dir = []
    if not all_togeter:
        for type in os.listdir(use_fake):
            real_dir.append([use_real])
            fake_dir.append([osp(use_fake, type)])
    else:
        real_dir.append([use_real])
        temp_fake = []
        for type in os.listdir(use_fake):
            temp_fake.append(osp(use_fake, type))
        fake_dir.append(temp_fake)
    return real_dir, fake_dir

def get_Chameleon_path():
    return [["/path/to/AIGC_Dataset/Chameleon/test/0_real/"]], [["/path/to/AIGC_Dataset/Chameleon/test/1_fake/"]]

def get_progan_path(is_train=True):
    if is_train:
        dataset_path = "/path/to/AIGC_Dataset/CNNSpot2020/progan_train/"
    else:
        dataset_path = "/path/to/AIGC_Dataset/CNNSpot2020/progan/"
    real_dir = []
    fake_dir = []
    temp_real = []
    temp_fake = []
    for type in os.listdir(dataset_path):
        temp_real.append(osp(dataset_path, type, '0_real'))
        temp_fake.append(osp(dataset_path, type, '1_fake'))
    real_dir.append(temp_real)
    fake_dir.append(temp_fake)
    return real_dir, fake_dir

def get_hires():
    dataset_path = "/path/to/AIGC_Dataset/z_dataset/HiRes_50K"
    ordered_types = ['W_0900', 'W_1200', 'W_1500', 'W_2000', 'W_2500', 'W_3000', 'W_5000', 'W>5000']
    real_dir = []
    fake_dir = []
    for type in ordered_types:
        real_dir.append([osp(dataset_path, type, '0_real')])
        fake_dir.append([osp(dataset_path, type, '1_fake')])
    return real_dir, fake_dir

def get_DRCT_train_1_4_path():
    real_dir = [["/path/to/AIGC_Dataset/DRCT/MSCOCO/train2017/"]]
    fake_dir = [['/path/to/AIGC_Dataset/DRCT/fake_rec_images/stable-diffusion-v1-4/train2017/']]
    return real_dir, fake_dir

def get_DRCT_train_2_1_path():
    real_dir = [["/path/to/AIGC_Dataset/DRCT/MSCOCO/train2017/"]]
    fake_dir = [['/path/to/AIGC_Dataset/DRCT/fake_rec_images/stable-diffusion-2-1/train2017/']]
    return real_dir, fake_dir

def get_DRCT_test_path():
    dataset_fake_path = "/path/to/AIGC_Dataset/DRCT/images/"
    dataset_real_path = "/path/to/AIGC_Dataset/DRCT/MSCOCO/"
    ordered_types = ['ldm-text2im-large-256', 
                     'stable-diffusion-v1-4', 'stable-diffusion-v1-5', 'stable-diffusion-2-1', 'stable-diffusion-xl-base-1.0', 'stable-diffusion-xl-refiner-1.0', 
                     'sd-turbo', 'sdxl-turbo',
                     'lcm-lora-sdv1-5', 'lcm-lora-sdxl',
                     'sd-controlnet-canny', 'sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0',
                     'stable-diffusion-inpainting', 'stable-diffusion-2-inpainting', 'stable-diffusion-xl-1.0-inpainting-0.1']
    real_dir = []
    fake_dir = []
    for type in ordered_types:
        real_dir.append(["/path/to/AIGC_Dataset/DRCT/MSCOCO/val2017"])
        fake_dir.append([osp(dataset_fake_path, type, 'val2017')])
    return real_dir, fake_dir

dataset_info = {
    'imagenet_genimage-sd14': get_Genimage_SD14_train(),
    'progan_train': get_progan_path(is_train=True),
    'progan_test': get_progan_path(is_train=False),
    'genimage': get_Genimage_path(),
    'genimage_train': get_Genimage_train_path(),
    'genimage_sd14_pure': get_Genimage_SD14_pure_train(),
    'genimage_sd14_pure_test':get_Genimage_SD14_pure_test(),
    'genimage_sd14_pure_myrec': get_Genimage_SD14_pure_train_rec(),
    'AIGCDetect': get_AIGCDetect_path(),
    'Chameleon': get_Chameleon_path(),
    'aeroblade': get_aeroblade_path(),
    'DRCT_train_1_4': get_DRCT_train_1_4_path(),
    'DRCT_train_2_1': get_DRCT_train_2_1_path(),
    'DRCT_test': get_DRCT_test_path(),
    'HiRes_50K': get_hires(),
}