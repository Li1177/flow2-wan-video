import os
import torch
from torch.hub import download_url_to_file
import comfy.utils
import folder_paths
from spandrel import ModelLoader

# Note: The .frame_interpolation module (IFNet, preprocess_frames, postprocess_frames, generic_frame_loop)
# is assumed to be available in the project directory or installed as a dependency.
from .frame_interpolation.rife_arch import IFNet
from .frame_interpolation.utils import preprocess_frames, postprocess_frames, generic_frame_loop

def clear_cuda_cache():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def download_github_model(repo_id, tag, file_name, saved_directory):
    download_url = f"https://github.com/{repo_id}/releases/download/{tag}/{file_name}"
    saved_directory = os.path.join(folder_paths.models_dir, saved_directory)
    saved_full_path = os.path.join(saved_directory, file_name)
    if os.path.exists(saved_full_path):
        print(f"exists model: {saved_full_path}")
        return saved_full_path
    else:
        os.makedirs(saved_directory, exist_ok=True)
    
    print(f"Downloading the {file_name} model. Please wait a moment...")
    download_url_to_file(download_url, saved_full_path, hash_prefix=None, progress=True)
    return saved_full_path

class WanVideoEnhancer_F2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "upscale_model": (["disabled"] + folder_paths.get_filename_list("upscale_models"), ),
                "interpolate_model": (("disabled", "rife47.pth", "rife48.pth", "rife49.pth"), ), 
                "upscale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.01}),
                "interpolate_frame": ("INT", {"default": 30, "min": 30, "max": 60, "step": 30}),
                "order": (("upscale_first", "interpolate_first"), ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("images", "framerate", )
    FUNCTION = "process"

    CATEGORY = "Flow2/Wan 2.1"

    def process(
            self,
            images,
            upscale_model,
            interpolate_model,
            upscale_factor,
            interpolate_frame,
            order,
        ):
        def upscale(images):
            return self.upscale(upscale_model, images, upscale_factor)

        def interpolate(images):
            return self.interpolate(interpolate_model, images, interpolate_frame)
        
        if order == "upscale_first":
            order = [upscale, interpolate]
        else:
            order = [interpolate, upscale]

        for k in order:
            images = k(images)

        framerate = interpolate_frame if interpolate_model != "disabled" else 16
        return (images, framerate, )

    def interpolate(self, model_name, images, framerate):
        if model_name == "disabled":
            print("No interpolation model specified. Skipping interpolation.")
            return images

        model_path = download_github_model("styler00dollar/VSGAN-tensorrt-docker", "models", model_name, "vfi_models")
        model = IFNet(arch_ver="4.7")
        sd = torch.load(model_path)
        model.load_state_dict(sd)
        del sd
        model.eval().to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        frames = preprocess_frames(images)
        
        print("interpolating...")
        clear_cache_after_n_frames = 10
        multiplier = int(framerate / 15)
        
        def return_middle_frame(frame_0, frame_1, timestep, model, scale_list, in_fast_mode, in_ensemble):
            return model(frame_0, frame_1, timestep, scale_list, in_fast_mode, in_ensemble)
        
        args = [model, [8, 4, 2, 1], True, True]
        images = postprocess_frames(
            generic_frame_loop(
                model_name.replace(".pth", ""),
                frames,
                clear_cache_after_n_frames,
                multiplier,
                return_middle_frame,
                *args,
                interpolation_states=None,
                dtype=torch.float32,
            )
        )
        clear_cuda_cache()
        return images

    def upscale(self, model_name, images, factor):
        if model_name == "disabled":
            print("No upscale model specified. Skipping upscale.")
            return images

        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        model = ModelLoader().load_from_state_dict(sd).eval()
        del sd

        if not isinstance(model, ImageModelDescriptor):
            del model
            raise Exception("Upscale model must be a single-image model.")
        
        print("upscaling...")
        scale = model.scale
        memory_required = comfy.model_management.module_size(model.model)
        memory_required += (512 * 512 * 3) * images.element_size() * max(scale, 1.0) * 384.0
        memory_required += images.nelement() * images.element_size()
        comfy.model_management.free_memory(memory_required, torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        model.to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
        in_img = images.movedim(-1, -3).to(torch.cuda.current_device() if torch.cuda.is_available() else "cpu")

        tile = 512
        overlap = 32
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=scale, pbar=pbar)
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
                
        del model
        clear_cuda_cache()
        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)
        scale_by = factor / scale
        samples = s.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, "lanczos", "disabled")
        s = s.movedim(1, -1)
        return s
