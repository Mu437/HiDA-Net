import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

MODEL_ID_MAP = {
    'SD-1-1':"CompVis/stable-diffusion-v1-1",
    'SD-1-4':"CompVis/stable-diffusion-v1-4",
    'SD-2':"stabilityai/stable-diffusion-2-base",
    'KD-2-1':"kandinsky-community/kandinsky-2-1",
}

def process_images_with_diffusion(input_dir,
                                  output_dir,
                                  model_id_short='SD-1-4',
                                  num_inference_steps=50,
                                  start_noise_step=-1,
                                  batch_size=16,
                                  max_images=None,
                                  device=None):
    import torch
    from diffusers import DiffusionPipeline
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
    from torchvision import transforms

    model_id = MODEL_ID_MAP[model_id_short]
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)
    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16,
                                            variant="fp16" if "kandinsky" in model_id else None,
                                             local_files_only=True, # 强制本地
                                            )
    pipe.to(device)
    print("Model Load Done!")
    if hasattr(pipe, "vae"):
        ae = pipe.vae
        if hasattr(pipe, "upcast_vae"):
            pipe.upcast_vae()
    elif hasattr(pipe, "movq"):
        ae = pipe.movq
    else:
        raise ValueError("Cannot find vae or movq in pipeline.")

    unet = pipe.unet
    scheduler = pipe.scheduler
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    preprocess = transforms.Compose([
        transforms.Resize(512),               
        transforms.CenterCrop((512, 512)),   
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    image_files = sorted(
        [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.JPEG'))]
    )
    if max_images is not None:
        image_files = image_files[:max_images]
    existing_files = set(os.listdir(output_dir))
    image_files = [f for f in image_files if f not in existing_files]

    image_paths = [os.path.join(input_dir, f) for f in image_files]

    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_files = image_files[i:i + batch_size]
        batch_paths = image_paths[i:i + batch_size]

        images = [Image.open(p).convert("RGB") for p in batch_paths]
        image_tensors = torch.stack([preprocess(img) for img in images]).to(device).half()

        with torch.no_grad():
            latents = ae.encode(image_tensors).latent_dist.sample() * 0.18215

        if start_noise_step is None:
            with torch.no_grad():
                decoded = ae.decode(latents / 0.18215).sample
            decoded = (decoded.clamp(-1, 1) + 1) / 2
            decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
            decoded = (decoded * 255).astype(np.uint8)

            for img_np, fname in zip(decoded, batch_files):
                # Image.fromarray(img_np).save(os.path.join(output_dir, fname))
                base_name = os.path.splitext(fname)[0]
                new_fname = base_name + ".png"
                save_path = os.path.join(output_dir, new_fname)
                Image.fromarray(img_np).save(save_path)
            continue

        scheduler.set_timesteps(num_inference_steps)
        timestep = scheduler.timesteps[start_noise_step]
        noise = torch.randn_like(latents)
        noisy_latents = scheduler.add_noise(latents, noise, timestep)

        text_input = tokenizer([""] * len(batch_files), return_tensors="pt", padding=True).input_ids.to(device)
        encoder_hidden_states = text_encoder(text_input)[0]

        latents = noisy_latents
        for t in scheduler.timesteps[start_noise_step:]:
            with torch.no_grad():
                if "kandinsky" in model_id:
                    noise_pred = unet(latent_input=latents, timestep=t, encoder_hidden_states=encoder_hidden_states).sample
                else:
                    noise_pred = unet(latents, t, encoder_hidden_states=encoder_hidden_states).sample
                latents = scheduler.step(noise_pred, t, latents).prev_sample

        with torch.no_grad():
            decoded = ae.decode(latents / 0.18215).sample
        decoded = (decoded.clamp(-1, 1) + 1) / 2
        decoded = decoded.cpu().permute(0, 2, 3, 1).numpy()
        decoded = (decoded * 255).astype(np.uint8)

        for img_np, fname in zip(decoded, batch_files):
            Image.fromarray(img_np).save(os.path.join(output_dir, fname))

if __name__ == '__main__':
    process_images_with_diffusion(
        input_dir="/path/to/input",
        output_dir="/path/to/output",
        model_id_short='SD-1-4',
        max_images=200000,
        start_noise_step=None,
    )