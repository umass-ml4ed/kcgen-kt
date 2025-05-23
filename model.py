import torch
import torch.optim as optim
import os
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model


def create_lstm_model(configs, device, hid_dim):
    lstm = nn.LSTM(configs.lstm_inp_dim, hid_dim, num_layers=configs.num_layers)
    lstm.to(device)
    
    return lstm

def create_knowledge_linear(device, hid_dim):
    linear = nn.Sequential(
        nn.ReLU(),
        nn.Linear(64, hid_dim),  
    )

    linear = linear.to(device)

    return linear

def create_tokenizer(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.okt_model)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def create_model(configs, device, lstm_hid_dim):
    tokenizer = create_tokenizer(configs)

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
        )
        
    model = AutoModelForCausalLM.from_pretrained(
            configs.okt_model,
            quantization_config=bnb_config
        )
    
    lora_config = LoraConfig(
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            r=configs.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", ],
            inference_mode=False
        )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.to(device)

    lstm = create_lstm_model(configs, device, lstm_hid_dim)

    return lstm, model, tokenizer

def load_okt_model(configs, device, now, continue_train):
    tokenizer = create_tokenizer(configs)
    model = okt_model_init(configs, device, now, continue_train)

    lstm = create_lstm_model(configs, device)
    lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))

    return lstm, model, tokenizer

def okt_model_init(configs, device, now, continue_train, load_in_8bit=True):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        bnb_8bit_compute_dtype=torch.bfloat16 if continue_train else torch.float16
    )

    model_dir = os.path.join(configs.model_save_dir, now, 'model')
    peft_config = PeftConfig.from_pretrained(model_dir)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(_hf_model, model_dir, is_trainable=continue_train).to(device)

    for param in model.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()

    return model

def create_multitask_predictor(device):
    predictor = nn.Linear(4096, 1).to(device)

    torch.nn.init.xavier_uniform_(predictor.weight)
    return predictor

def load_model_eval(configs, now, device):
    if configs.save_model:
        model = okt_model_init(configs, device, now, False, load_in_8bit=True)
        tokenizer = create_tokenizer(configs)

        lstm_hid_dim = 18 if configs.baseline else 11
        if configs.transition:
            lstm = create_lstm_model(configs, device, 64)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))
            trans_linear = create_knowledge_linear(device, lstm_hid_dim)
            trans_linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'transition')))

        else:
            lstm = create_lstm_model(configs, device, lstm_hid_dim)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))
            trans_linear = None

        if configs.multitask:
            predictor = create_multitask_predictor(device)
            predictor.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'predictor')))
    
    else:
        lstm_dim = 18 if configs.baseline else 11
        if not configs.transition:
            lstm, model, tokenizer = create_model(configs, device, lstm_dim)
            trans_linear = None
        else:
            lstm, model, tokenizer = create_model(configs, device, 64)
            trans_linear = create_knowledge_linear(device, lstm_dim)

        if configs.multitask:
            predictor = create_multitask_predictor(device)
    
    return model, lstm, predictor, tokenizer, trans_linear