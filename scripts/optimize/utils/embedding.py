from sentence_transformers import SentenceTransformer
import torch    
import gc
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel,AutoModelForCausalLM)
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig
from accelerate.hooks import remove_hook_from_module
from tqdm.autonotebook import trange
import torch.nn.functional as F
LLM_DIM_DICT = {"ST": 384, "BERT": 768, "e5": 1024, "llama2_7b": 4096, "llama2_13b": 5120, "llama3_1_8b": 5120,"llama3_2_3b": 5120}


def mean_pooling(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-10)



def get_sentence_embedding(sentence,device='cpu'):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentence,device=device,convert_to_numpy=True)   
    return embeddings




def get_encoder(llm_name, cache_dir="cache_data/model", batch_size=1):
    return SentenceEncoder(llm_name, cache_dir, batch_size,)  











def get_available_devices():
    r"""Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids
class LLMModel(torch.nn.Module):
    """
    Large language model from transformers.
    If peft is ture, use lora with pre-defined parameter setting for efficient fine-tuning.
    quantization is set to 4bit and should be used in the most of the case to avoid OOM.
    """
    def __init__(self, llm_name, quantization=True, peft=True, cache_dir="cache_data/model", max_length=1000):
        super().__init__()
        assert llm_name in LLM_DIM_DICT.keys()
        self.llm_name = llm_name
        self.quantization = quantization

        self.indim = LLM_DIM_DICT[self.llm_name]
        self.cache_dir = cache_dir
        self.max_length = max_length
        model, self.tokenizer = self.get_llm_model()
        if peft:
            self.model = self.get_lora_perf(model)
        else:
            self.model = model
        self.tokenizer.padding_side = "right"
        self.tokenizer.truncation_side = 'right'

    def find_all_linear_names(self, model):
        """
        find all module for LoRA fine-tuning.
        """
        cls = bnb.nn.Linear4bit if self.quantization else torch.nn.Linear
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    def create_bnb_config(self):
        """
        quantization configuration.
        """
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            # bnb_4bit_use_double_quant=True,
            # bnb_4bit_quant_type="nf4",
            # bnb_4bit_compute_dtype=torch.bfloat16
        )

        return bnb_config

    def get_lora_perf(self, model):
        """
        LoRA configuration.
        """
        target_modules = self.find_all_linear_names(model)
        config = LoraConfig(
            target_modules=target_modules,
            r=16,  # dimension of the updated matrices
            lora_alpha=16,  # parameter for scaling
            lora_dropout=0.2,  # dropout probability for layers
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, config)

        return model

    def get_llm_model(self):
        if self.llm_name == "llama2_7b":
            model_name = "meta-llama/Llama-2-7b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer
        elif self.llm_name == "llama3_1_8b":
            model_name = "meta-llama/Llama-3.1-8B"
            ModelClass = AutoModelForCausalLM
            TokenizerClass = AutoTokenizer
            
        elif self.llm_name == "llama2_13b":
            model_name = "meta-llama/Llama-2-13b-hf"
            ModelClass = LlamaForCausalLM
            TokenizerClass = LlamaTokenizer
        elif self.llm_name == "llama3_2_3b":    
            model_name = "meta-llama/Llama-3.2-3B"
            ModelClass = LlamaForCausalLM
            TokenizerClass = AutoTokenizer
        elif self.llm_name == "e5":
            model_name = "intfloat/e5-large-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "BERT":
            model_name = "bert-base-uncased"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        elif self.llm_name == "ST":
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            ModelClass = AutoModel
            TokenizerClass = AutoTokenizer

        else:
            raise ValueError(f"Unknown language model: {self.llm_name}.")
        if self.quantization:
            bnb_config = self.create_bnb_config()
            model = ModelClass.from_pretrained(model_name,
                                               quantization_config=bnb_config,
                                               #attn_implementation="flash_attention_2",
                                               #torch_type=torch.bfloat16,
                                               cache_dir=self.cache_dir)
        else:
            model = ModelClass.from_pretrained(model_name, cache_dir=self.cache_dir)
        model = remove_hook_from_module(model, recurse=True)
        model.config.use_cache = False
        if model_name == "meta-llama/Llama-3.1-8B":
            tokenizer = TokenizerClass.from_pretrained(model_name,add_eos_token=True)
            
        tokenizer = TokenizerClass.from_pretrained(model_name, cache_dir=self.cache_dir, add_eos_token=True)
        if self.llm_name[:6] == "llama2":
            tokenizer.pad_token = tokenizer.bos_token
        return model, tokenizer


    def pooling(self, outputs, text_tokens=None):
        # if self.llm_name in ["BERT", "ST", "e5"]:
        return F.normalize(mean_pooling(outputs, text_tokens["attention_mask"]), p=2, dim=1)

        # else:
        #     return outputs[text_tokens["input_ids"] == 2] # llama2 EOS token

    def forward(self, text_tokens):
        outputs = self.model(input_ids=text_tokens["input_ids"],
                             attention_mask=text_tokens["attention_mask"],
                             output_hidden_states=True,
                             return_dict=True)["hidden_states"][-1]

        return self.pooling(outputs, text_tokens)

    def encode(self, text_tokens, pooling=False):

        with torch.no_grad():
            outputs = self.model(input_ids=text_tokens["input_ids"],
                                 attention_mask=text_tokens["attention_mask"],
                                 output_hidden_states=True,
                                 return_dict=True)["hidden_states"][-1]
            outputs = outputs.to(torch.float32)
            if pooling:
                outputs = self.pooling(outputs, text_tokens)

            return outputs, text_tokens["attention_mask"]


class SentenceEncoder:
    def __init__(self, llm_name, cache_dir="cache_data/model", batch_size=1, multi_gpu=False):
        self.llm_name = llm_name
        self.device, _ = get_available_devices()
        self.batch_size = batch_size
        self.multi_gpu = multi_gpu
        self.model = LLMModel(llm_name, quantization=True, peft=False, cache_dir=cache_dir)
        self.model.to(self.device)

    def encode(self, texts, to_tensor=True):
        all_embeddings = []
        with torch.no_grad():
            for start_index in trange(0, len(texts), self.batch_size, desc="Batches", disable=False, ):
                sentences_batch = texts[start_index: start_index + self.batch_size]
                if self.llm_name in ['llama3_1_8b','llama3_2_3b']:
                    text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", truncation=True,
                                           max_length=500).to(self.device)  
                else: 
                    text_tokens = self.model.tokenizer(sentences_batch, return_tensors="pt", padding="longest", truncation=True,
                                           max_length=500).to(self.device)
                embeddings, _ = self.model.encode(text_tokens, pooling=True)
                embeddings = embeddings.cpu()
                all_embeddings.append(embeddings)
        all_embeddings = torch.cat(all_embeddings, dim=0)
        if not to_tensor:
            all_embeddings = all_embeddings.numpy()

        return all_embeddings

    def flush_model(self):
        # delete llm from gpu to save GPU memory
        if self.model is not None:
            self.model = None
        gc.collect()
        torch.cuda.empty_cache()