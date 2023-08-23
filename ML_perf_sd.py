from diffusers import StableDiffusionPipeline
import torch
import csv

model_id_list = [
"stabilityai/stable-diffusion-2-1-base",
]
steps = 100
batch_size = 1
height = 512
width = 512
dtype = torch.float16
safety_chk = False
warm_up = True

mode_list = [
        "eager",
        "compilation",
        "xformers",
        ]

mode_list = [
        "eager",
        "compilation",
        ]

v_torch = torch.__version__
v_cuda = torch._C._cuda_getCompiledVersion()
v_cudnn = torch.backends.cudnn.version()
v_device = torch.cuda.get_device_name(0)

def run_sd(k):
    with open('ML_perf_sd_' + str(v_torch) + '_' + str(v_cuda) + '_' + str(v_cudnn) + '.csv','w') as f:
        for m in mode_list:
            w = csv.writer(f)
            w.writerow(["torch", v_torch])
            w.writerow(["rocm/cuda", v_cuda])
            w.writerow(["cudnn", v_cudnn])
            w.writerow(["device", v_device])
            #pipe = StableDiffusionPipeline.from_pretrained(k, torch_dtype=dtype) # official repo
            pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-2-1-base", torch_dtype=dtype) # local directory
            pipe = pipe.to("cuda:0")
            if safety_chk == True:
                pipe.safety_checker = lambda images, clip_input: (images, False)
            if m == "eager":
                print("[INFO]eager_mode")
                data = "eager(itrs/sec), steps, image_size"
                w.writerow([data])
            elif m == "compilation":
                print("[INFO]compilation_mode")
                data = "triton_comp(itrs/sec), steps, image_size"
                pipe.unet = torch.compile(pipe.unet)
                w.writerow([data])
            elif m == "xformers":
                print("[INFO]xformers_mode")
                data = "xformers_mha(itrs/sec), steps, image_size"
                w.writerow([data])
                from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
                pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
                pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
            #with torch.inference_mode(): # conflicts with xformers
            prompt = "postapocalyptic steampunk city, exploration, cinematic, realistic, hyper detailed, photorealistic maximum detail, volumetric light, (((focus))), wide-angle, (((brightly lit))), (((vegetation))), lightning, vines, destruction, devastation, wartorn, ruins"
            neg_prompt = "(((blurry))), ((foggy)), (((dark))), ((monochrome)), sun, (((depth of field)))"

            if warm_up == True:
                image = pipe(prompt, negative_prompt = neg_prompt, guidance_scale=15.0, num_inference_steps = steps, num_images_per_prompt = batch_size, height=height, width=width).images
            image = pipe(prompt, negative_prompt = neg_prompt, guidance_scale=15.0, num_inference_steps = steps, num_images_per_prompt = batch_size, height=height, width=width).images
            k = k.replace("/", "_")
            image[0].save(k + "_" + str(steps) + "_" + str(dtype) + "_" + str(height) + ".jpg")
            data =  "{:.5f}, {}, {}".format(steps/pipe.latency_measure(), steps, width)
            w.writerow([data])
            print("[INFO]itr/sec", steps/pipe.latency_measure())

if __name__ == "__main__":
    for i, k in enumerate(model_id_list):
        run_sd(k)
