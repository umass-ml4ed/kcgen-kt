import torch
import torch.optim as optim
import os
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model
from pdb import set_trace

def create_lstm_model(configs, device, hid_dim):
    lstm = nn.LSTM(configs.lstm_inp_dim, hid_dim, num_layers=configs.num_layers)
    lstm.to(device)
    
    return lstm

def create_knowledge_linear(device, hid_dim, transition_dim):
    if transition_dim == 64:
        linear = nn.Sequential(
        nn.ReLU(),
        nn.Linear(transition_dim, hid_dim),
    )

    else:
        linear = nn.Sequential(
            nn.Linear(transition_dim, 64),
            nn.ReLU(),  
            nn.Linear(64, hid_dim),
        )

    linear = linear.to(device)

    return linear

def create_tokenizer(configs):
    tokenizer = AutoTokenizer.from_pretrained(configs.okt_model, use_fast=False)
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
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            inference_mode=False
        )
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.to(device)

    lstm = create_lstm_model(configs, device, lstm_hid_dim)

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

def create_multitask_predictor(device, multilayer=False):
    if multilayer:
        predictor = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),  
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        for layer in predictor:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    else:
        predictor = nn.Linear(4096, 1)
        torch.nn.init.xavier_uniform_(predictor.weight)

    predictor = predictor.to(device)

    return predictor

def load_model_eval(configs, now, device, no_kc):
    if configs.save_model:
        model = okt_model_init(configs, device, now, False, load_in_8bit=True)
        model.eval()

        tokenizer = create_tokenizer(configs)

        lstm_hid_dim = no_kc
        if configs.transition:
            lstm = create_lstm_model(configs, device, configs.transition_dim)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm'), weights_only=True))
            trans_linear = create_knowledge_linear(device, lstm_hid_dim, configs.transition_dim)
            trans_linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'transition'), weights_only=True))

        else:
            lstm = create_lstm_model(configs, device, lstm_hid_dim)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))
            trans_linear = None

        if configs.multitask:
            predictor = create_multitask_predictor(device, configs.predictor_multilayer)
            predictor.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'predictor'), weights_only=True))
    
    else:
        lstm_dim = no_kc
        if not configs.transition:
            lstm, model, tokenizer = create_model(configs, device, lstm_dim)
            trans_linear = None
        else:
            lstm, model, tokenizer = create_model(configs, device, configs.transition_dim)
            trans_linear = create_knowledge_linear(device, lstm_dim, configs.transition_dim)

        if configs.multitask:
            predictor = create_multitask_predictor(device)
    
    return model, lstm, predictor, tokenizer, trans_linear

def create_binary_predictor(device):
    predictor = nn.Linear(4096, 2)
    torch.nn.init.xavier_uniform_(predictor.weight)

    predictor = predictor.to(device)

    return predictor