import time
import torch
import os
import gc
import subprocess
import platform as pf
import pkg_resources as pr
import csv
from diffusers import StableDiffusionPipeline
from transformers import pipeline, AutoModelWithLMHead, AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import ViTImageProcessor, ViTForImageClassification

torch.set_float32_matmul_precision('high')

from PIL import Image
import requests

os.environ["TOKENIZERS_PARALLELISM"] = "false"

e_flan_t5_large = False
e_bert_large_uncased_compile = False
e_vit_base_patch16_224_compile = False
e_stable_diffusion_v1_5_compile = False

def model_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

model_id_lists = [
"huggyllama/llama-7b",
"bert-large-uncased" ,
"google/flan-t5-large",
"facebook/opt-1.3b",
"bigscience/bloomz-1b1",
"EleutherAI/gpt-neo-1.3b",
"bigcode/tiny_starcoder",
"google/vit-base-patch16-224",
"runwayml/stable-diffusion-v1-5" ,
]

model_id_lists = [
"google/flan-t5-large",
]
#Mid Journey

hf_path_prefix = "https://huggingface.co/"
data_list = ["device", "rocm", "python", "torch", "library", "model_category", "model_id", "optimize", "data_type", "metric_latency(ms),bs=1,itr=10", "regress_input", "regress_output", "remarks"]
lib_list = ["diffusers", "transformers", "accelerate"]
opt_list = ["eager", "compile"]
dtype_list = [torch.float16, torch.float32, torch.bfloat16]
itr = 10

def model_prep(mid):
    try:
        if subprocess.run(["ls", mid], stdout = subprocess.DEVNULL).returncode == 2:
            subprocess.run(["git-lfs", "clone", hf_path_prefix+model_id])
    except:
        print("check out the HF model name")

def fill_mask(mid, opt, dty, itr):
    print(" ")
    print("current step: ", mid, opt, dty)
    print(" ")
    global model
    global tokenizer
    tc_mode = "default"
    device = "cuda"

    tokenizer = AutoTokenizer.from_pretrained(mid)
    model = AutoModelWithLMHead.from_pretrained(mid, torch_dtype=dty)
    if e_bert_large_uncased_compile == True and opt == "compile":
        return "SEG_FAULT", "SEG_FAULT", "SEG_FAULT"
    if opt == "eager":
        model.to(device)
    elif opt == "compile":
        model.to(device)
        #model = torch.compile(model, mode = tc_mode, dynamic=True)
        model = torch.compile(model, mode = tc_mode)
    elif opt == "deepspeed":
        return "TBD", "TBD", "TBD"
        import deepspeed
        ds_model = deepspeed.init_inference(
            model=model,      # Transformers models
            mp_size=1,        # Number of GPU
            dtype=dty, # dtype of the weights (fp16)
            # injection_policy={"BertLayer" : HFBertLayerPolicy}, # replace BertLayer with DS HFBertLayerPolicy
            replace_method="auto", # Lets DS autmatically identify the layer to replace
            replace_with_kernel_inject=True, # replace the model with the kernel injector
        )

    inp = "Paris is crowded with [MASK] now."
    input_ids = tokenizer.encode(inp, return_tensors="pt").to(device)
    model.eval()
    # warmup
    model_out = model(input_ids)
    # measure
    e_tot = 0
    for i in range(itr):
        s = time.perf_counter()
        model_out = model(input_ids)
        e_tot = e_tot + time.perf_counter() - s

    model_out["input_ids"] = input_ids.cpu()
    # create temp pipe instance for reusing postprocess
    pipe = pipeline('fill-mask', model='bert-large-uncased', device="cuda:0", torch_dtype=torch.float16)
    outp = pipe.postprocess(model_out, top_k=1)
    del model
    del tokenizer
    return "{:.0f}".format(e_tot/itr * 1000), inp, outp

def text_generation(mid, opt, dty, itr):
    print(" ")
    print("current step: ", mid, opt, dty)
    print(" ")
    global model
    global tokenizer
    tc_mode = "default"
    cache = True
    device = "cuda"

    if mid == "EleutherAI/gpt-neo-1.3b":
        model = GPTNeoForCausalLM.from_pretrained(mid, torch_dtype=dty).cuda(device)
        tokenizer = GPT2Tokenizer.from_pretrained(mid)
        if opt == "eager":
            model.to(device)
        elif opt == "compile":
            model.to(device)
            for i in range(model.config.num_hidden_layers):
                model.transformer.h[i].attn = torch.compile(model.transformer.h[i].attn, mode= tc_mode)
        elif opt == "deepspeed":
            return "TBD", "TBD", "TBD"
    elif mid == "google/flan-t5-large":
        model = T5ForConditionalGeneration.from_pretrained(mid, torch_dtype=dty).cuda(device)
        tokenizer = T5Tokenizer.from_pretrained(mid)
        if e_flan_t5_large == True:
            return "SEG_FAULT", "SEG_FAULT", "SEG_FAULT"
        if opt == "eager":
            model.to(device)
        elif opt == "compile":
            model.to(device)
            model = torch.compile(model, mode= tc_mode)
        elif opt == "deepspeed":
            return "TBD", "TBD", "TBD"
    elif mid == "bigscience/bloomz-1b1":
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dty).cuda(device)
        tokenizer = AutoTokenizer.from_pretrained(mid)
        if opt == "eager":
            model.to(device)
        elif opt == "compile":
            model.to(device)
            for i in range(model.config.num_hidden_layers):
                model.transformer.h[i].self_attention = torch.compile(model.transformer.h[i].self_attention, mode= tc_mode)
        elif opt == "deepspeed":
            return "TBD", "TBD", "TBD"

    elif mid == "huggyllama/llama-7b":
        # exception
        if dty == torch.float32:
            return "OOM", "OOM", "OOM"
        else:
            model = LlamaForCausalLM.from_pretrained(mid, torch_dtype=dty).cuda(device)
            tokenizer = AutoTokenizer.from_pretrained(mid)
            if opt == "eager":
                model.to(device)
            elif opt == "compile":
                model.to(device)
                for i in range(model.config.num_hidden_layers):
                    model.model.layers[i].self_attn = torch.compile(model.model.layers[i].self_attn, mode= tc_mode)
            elif opt == "deepspeed":
                return "TBD", "TBD", "TBD"
    elif mid == "facebook/opt-1.3b":
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dty).cuda(device)
        tokenizer = AutoTokenizer.from_pretrained(mid)
        if opt == "eager":
            model.to(device)
        elif opt == "compile":
            model.to(device)
            for i in range(model.config.num_hidden_layers):
                model.model.decoder.layers[i].self_attn = torch.compile(model.model.decoder.layers[i].self_attn, mode= tc_mode)
        elif opt == "deepspeed":
            return "TBD", "TBD", "TBD"
    else:
        model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dty).cuda(device)
        tokenizer = AutoTokenizer.from_pretrained(mid)

    if mid == "bigcode/tiny_starcoder":
        inp = "def print_hello_world():"
    elif mid == "google/flan-t5-large":
        inp = "translate English to German: How old are you?"
    else:
        inp = "Upon graduation, I will be able to analyze medieval Spanish poems using literary terms and cultural context, describe the electronegativity trends on the periodic table, and identify when to use logarithmic differentiation to simplify a derivative problem. Despite knowing how to execute these very particular tasks, I currently fail to understand how to change a tire, how to do my taxes efficiently, or how to obtain a good insurance policy. A factory-model school system that has been left essentially unchanged for nearly a century has been the driving force in my educational development. I have been conditioned to complete tasks quickly, efficiently, and with an advanced understanding. I measured my self-worth as my ability to outdo my peers academically, thinking my scores were the only aspect that defined me; and they were. I was getting everything right. Then, I ran for Student Government and failed. Rejection. I didn’t even make it past the first round of cuts. How could that be? I was statistically a smart kid with a good head on my shoulders, right? Surely someone had to have made a mistake. Little did I know, this was my first exposure to meaning beyond numbers. As I was rejected from StuGo for the second year in a row, I discovered I had been wrongfully measuring my life through numbers--my football statistics, my test scores, my age, my height (I’m short). I had the epiphany that oh wait, maybe it was my fault that I had never prioritized communication skills, or open-mindedness (qualities my fellow candidates possessed). Maybe it was me. That must be why I always had to be the one to approach people during my volunteer hours at the public library to offer help--no one ever asked me for it. I resolved to alter my mindset, taking a new approach to the way I lived. From now on I would emphasize qualitative experiences over quantitative skills. I had never been more uncomfortable. I forced myself to learn to be vulnerable by asking questions even if I was terrified of being wrong. My proficiency in using data evidence could not teach me how to communicate with young children at church, nor could my test scores show me how to be more open to criticism. The key to all of these skills, I was to discover, happened to be learning from those around me. Turns out, I couldn’t do everything by myself. The process of achieving this new mindset came through the cultivation of relationships. I became fascinated by the new perspectives each person " 

    batch_exp = 1
    v = 32 
    input_ids = tokenizer(inp, return_tensors="pt").input_ids.cuda(device)
    k = len(input_ids[0])
    print(k)
    for b in range(0, batch_exp):
        # warm up
        batch = 2 ** b
        gen_tokens = model.generate(
          input_ids,
          do_sample=False,
          min_length=k + v,
          max_length=k + v,
          use_cache=cache,
          pad_token_id=tokenizer.eos_token_id
        )

        e_tot = 0
        for i in range(itr):
            s = time.perf_counter()
            gen_tokens = model.generate(
              input_ids,
              do_sample=False,
              min_length=k + v,
              max_length=k + v,
              use_cache=cache,
              pad_token_id=tokenizer.eos_token_id
            )
            e_tot = e_tot + time.perf_counter() - s

    outp = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    del model
    del tokenizer
    return "{:.0f}".format(e_tot/itr * 1000), inp, outp

def image_classification(mid, opt, dty, itr):
    print(" ")
    print("current step: ", mid, opt, dty)
    print(" ")
    global model
    global processor
    tc_mode = "default"
    cache = True
    device = "cuda"

    if e_vit_base_patch16_224_compile and opt == "compile":
        return "SEG_FAULT", "SEG_FAULT", "SEG_FAULT"

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', torch_dtype=dty)
    if opt == "eager":
        model.to(device)
    elif opt == "compile":
        model.to(device)
        model = torch.compile(model, mode = tc_mode)
    elif opt == "deepspeed":
        return "TBD", "TBD", "TBD"

    inp = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(inp, stream=True).raw)
    inputs = processor(images=image, return_tensors="pt").to(device)

    model.eval()
    # warmup
    outputs = model(**inputs)
    e_tot = 0
    for i in range(itr):
        s = time.perf_counter()
        outputs = model(**inputs)
        e_tot = e_tot + time.perf_counter() - s

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    outp = "Predicted class:", model.config.id2label[predicted_class_idx]
    del model
    del processor
    return "{:.0f}".format(e_tot/itr * 1000), inp, outp

def text_to_image(mid, opt, dty, itr):
    print(" ")
    print("current step: ", mid, opt, dty)
    print(" ")
    global model
    global processor

    if e_stable_diffusion_v1_5_compile == True and opt == "compile":
        return "SEG_FAULT", "SEG_FAULT", "SEG_FAULT"

    tc_mode = "default"
    device = "cuda"
    steps = 25
    batch_size = 1
    height = 512
    width = 512

    pipe = StableDiffusionPipeline.from_pretrained(mid, torch_dtype=dty)

    if opt == "eager":
        pipe = pipe.to(device)
    elif opt == "compile":
        pipe = pipe.to(device)
        pipe.unet = torch.compile(pipe.unet, mode = tc_mode)
    elif opt == "deepspeed":
        return "TBD", "TBD", "TBD"

    inp = "interior design, open plan, kitchen and living room, modular furniture with cotton textiles, wooden floor, high ceiling, large steel windows viewing a city"
    #warm-up
    image = pipe(inp, num_inference_steps = steps, num_images_per_prompt = batch_size, height=height, width=width).images

    e_tot = 0
    for i in range(itr):
        s = time.perf_counter()
        image = pipe(inp, num_inference_steps = steps, num_images_per_prompt = batch_size, height=height, width=width).images
        e_tot = e_tot + time.perf_counter() - s

    mid = mid.replace("/", "_")
    image[0].save(mid+"_"+str(dty)+"_"+str(steps)+"_steps.jpg")
    del pipe
    return "{:.0f}".format(e_tot/itr * 1000), inp, mid+"_"+str(dty)+"_"+str(steps)+"_steps.jpg"

def get_data(mid, data_list, lib_list, opt, dty, itr):
    global latency, inp, outp
    data = {}
    for e in data_list:
        if e == "device":
            #data.update({e:torch.cuda.get_device_name(0)})
            data.update({e:"NVIDIA_GeForce_RTX_4090"})
        elif e == "rocm/cuda":
            data.update({e:torch._C._cuda_getCompiledVersion()})
        elif e == "python":
            data.update({e:pf.python_version()})
        elif e == "torch":
            data.update({e:pr.get_distribution("torch").version})
        elif e == "library":
            for i in lib_list:
                data.update({e+"_"+i:pr.get_distribution(i).version})
        elif e == "model_category":
            if mid == "bert-large-uncased":
                data.update({e:"fill_mask"})
                latency, inp, outp = fill_mask(mid, opt, dty, itr)
                rmk = "bs=1"
            elif mid == "bigcode/tiny_starcoder":
                data.update({e:"text_generation"})
                latency, inp, outp = "LICENSE", "LICENSE", "LICENSE"
                rmk = "Need Licensing"
            elif mid == "facebook/opt-1.3b" or mid == "bigscience/bloomz-1b1" or mid == "EleutherAI/gpt-neo-1.3b" or mid == "huggyllama/llama-7b":
                data.update({e:"text_generation"})
                latency, inp, outp = text_generation(mid, opt, dty, itr)
                rmk = "bs=1, in_length=512, out_length=32"
            elif mid == "google/flan-t5-large":
                data.update({e:"text_to_text_generation"})
                latency, inp, outp = text_generation(mid, opt, dty, itr)
                rmk = "bs=1"
            elif mid == "google/vit-base-patch16-224":
                data.update({e:"image_classification"})
                latency, inp, outp = image_classification(mid, opt, dty, itr)
                rmk = "bs=1"
            elif mid == "runwayml/stable-diffusion-v1-5":
                data.update({e:"text_to_image"})
                latency, inp, outp = text_to_image(mid, opt, dty, itr)
                rmk = "bs=1, steps=25, size=512x512"
        elif e == "model_id":
            data.update({e:mid})
        elif e == "optimize":
            data.update({e:opt})
        elif e == "data_type":
            data.update({e:dty})
        elif e == "metric_latency(ms),bs=1,itr=10":
            data.update({e:latency})
        elif e == "regress_input":
            data.update({e:inp})
        elif e == "regress_output":
            data.update({e:outp})
        elif e == "remarks":
            data.update({e:rmk})
    return data

tokenizer = None
processor = None
model = None
pipe = None

latency  = None
inp = None
outp = None

if __name__ == "__main__":
    with open('ML_regress.csv','w') as f:
        w = csv.writer(f)
        for i, mid in enumerate(model_id_lists):
            for j, opt in enumerate(opt_list):
                for k, dty in enumerate(dtype_list):
                    #model_prep(mid)
                    data = get_data(mid, data_list, lib_list, opt, dty, itr)
                    if i == 0 and j == 0 and k == 0:
                        w.writerow(data.keys())
                    w.writerow(data.values())
                    model_cleanup()
        f.close()
