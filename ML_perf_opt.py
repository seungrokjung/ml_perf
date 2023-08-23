import torch
import time
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaForCausalLM, LlamaTokenizer
import os
import csv
torch.set_float32_matmul_precision('high')

os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id_lists = [
#"facebook/opt-125m",
#"facebook/opt-350m",
"facebook/opt-1.3b",
#"facebook/opt-6.7b",
#"facebook/opt-13b",
#"facebook/opt-66b",
#"gpt/gpt2-xl",
#"EleutherAI/gpt-neo-1.3B",
#"EleutherAI/gpt-neox-20b",
]

dtype_list = [torch.float16, torch.float32, torch.bfloat16]
dtype_list = [torch.float16]

v_torch = torch.__version__
v_cuda = torch._C._cuda_getCompiledVersion()
v_cudnn = torch.backends.cudnn.version()
v_device = torch.cuda.get_device_name(0)

def get_data(mid, dty):
    global tokenizer
    global model
    d_type = torch.float16
    device = "cuda:0"
    cache = True
    warm_up = True

    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=dty).cuda(device)
    tokenizer = AutoTokenizer.from_pretrained(mid)
    model.to(device)
    model.eval()
    batch_exp = 7
    in_out_length = {128:8, 512:32}
    with open('ML_perf_opt_' + str(v_torch) + '_' + str(v_cuda) + '_' + str(v_cudnn) + '.csv','w') as f:
        w = csv.writer(f)
        w.writerow(["torch", v_torch])
        w.writerow(["rocm/cuda", v_cuda])
        w.writerow(["cudnn", v_cudnn])
        w.writerow(["device", v_device])
        init = "batch, input_length, output_length, total_latency(ms), input_latency(ms), output_latency(ms), 1-token_output_latency(ms)"
        w.writerow([init])
        for k, v in in_out_length.items():
            for b in range(0, batch_exp):
              if 0:
                  # exception 
                  if b == 6 and k == 512:
                      exception_data = "OOM, OOM, OOM, OOM, OOM, OOM, OOM"
                      w.writerow([exception_data])
                      continue
              batch = 2 ** b
              if warm_up == True:
                if  device == "cpu":
                    input_ids = torch.randint(20, 50000, (batch, k))
                else:
                    input_ids = torch.randint(20, 50000, (batch, k)).cuda(device)

                gen_tokens = model.generate(
                    input_ids,
                    do_sample=False,
                    min_length=k + 1,
                    max_length=k + 1,
                    use_cache=cache,
                    pad_token_id=tokenizer.eos_token_id
                )
              if  device == "cpu":
                input_ids = torch.randint(20, 50000, (batch, k))
              else:
                input_ids = torch.randint(20, 50000, (batch, k)).cuda(device)
              start = time.perf_counter()
              gen_tokens = model.generate(
                input_ids,
                do_sample=False,
                min_length=k + 1,
                max_length=k + 1,
                use_cache=cache,
                pad_token_id=tokenizer.eos_token_id
              )
              end = time.perf_counter() - start
              in_latency = end

              if warm_up == True:
                if  device == "cpu":
                    input_ids = torch.randint(20, 50000, (batch, k))
                else:
                    input_ids = torch.randint(20, 50000, (batch, k)).cuda(device)

                gen_tokens = model.generate(
                    input_ids,
                    do_sample=False,
                    min_length=k + v,
                    max_length=k + v,
                    use_cache=cache,
                    pad_token_id=tokenizer.eos_token_id
                )
              if  device == "cpu":
                input_ids = torch.randint(20, 50000, (batch, k))
              else:
                #input_ids = torch.randint(20, 50000, (batch, k)).cuda(device)
                prompt = "DeepSPeed is a machine learning framework" 
                inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda(device)
              start = time.perf_counter()
              gen_tokens = model.generate(
                input_ids,
                do_sample=False,
                min_length=k + v,
                max_length=k + v,
                use_cache=cache,
                pad_token_id=tokenizer.eos_token_id
              )
              end = time.perf_counter() - start
              tot_latency = end
              print(str(batch) + ", " +  str(k) + ", " + str(v) + ", {:.0f}".format(tot_latency * 1000) + ", {:.0f}".format(in_latency * 1000) + ", {:.0f}".format((tot_latency - in_latency) * 1000) + ", {:.0f}".format((tot_latency - in_latency)/v * 1000))
              data =  str(batch) + ", " +  str(k) + ", " + str(v) + ", {:.0f}".format(tot_latency * 1000) + ", {:.0f}".format(in_latency * 1000) + ", {:.0f}".format((tot_latency - in_latency) * 1000) + ", {:.0f}".format((tot_latency - in_latency)/v * 1000)
              w.writerow([data])
        f.close()

tokenizer = None
model = None

if __name__ == "__main__":
        for i, mid in enumerate(model_id_lists):
            for k, dty in enumerate(dtype_list):
                get_data(mid, dty)
            #del model
            #del tokenizer
