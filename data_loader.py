import pandas as pd
import ast
import code_ast
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import log_loss, r2_score
from scipy.stats import ttest_ind
import torch
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace
from model import *
from trainer import *
import random
import math
import statistics
from sklearn.preprocessing import OneHotEncoder

# get unique problem statement
def get_unique_problems(df):
    questions = df['ProblemID'].unique()
    return questions

# get problem method declaration
def get_method_dec(df):
    method_dict = {}
    unique_prob_df = df.drop_duplicates(subset='ProblemID')
    for ind, row in unique_prob_df.iterrows():
        code_piece = row['Code'].split('\r')
        method_dict[row['prompt']] = code_piece[0]

    return method_dict

# Value of kc_dict has a format: [(initial_generated_kc, cluster_name, final_kc)]
# return a dictionary {problem: [final_kc1, final_kc2...]}
def get_problem_kc(kc_file="problem_kc.json"):
    res = {}
    kc_set= set()
    with open(kc_file, 'r') as f:
        kc_dict = json.load(f)
        for key, val in kc_dict.items():
            uniq_kc = set([i[-1] for i in val])
            res[key] = list(uniq_kc)

            for j in val:
                kc_set.add(j[-1])

    kc_set = list(kc_set)
    kc_dict_res = {kc_set[i]: i for i in range(len(kc_set))}

    
    return res, kc_dict_res


# get baseline kc
def extract_baseline_kc(file):
    df = pd.read_csv(file)
    df.fillna(0, inplace=True)

    kcs = list(df.columns)[4:]
    res = {row['Requirement']:df.columns[4:][row[4:] == 1].tolist() for _, row in df.iterrows()}

    kc_dict_res = {kcs[i]: i for i in range(len(kcs))}

    return res, kc_dict_res
    

def read_data(file, kc_problem_dict, configs):
    df = pd.read_pickle(file)
    # print(df.head())

    # Decide final score format
    if configs.label_type == 'binary':
        df['Score'] = np.where(df["Score_x"] == 1, 1, 0)

    else:
        df['Score'] = df['Score_x']

    df.drop(columns=['Score_x', 'Score_y'], inplace=True)

    # Map problem KCs to each data
    df['knowledge_component'] = df['prompt'].map(kc_problem_dict)

    # if configs.first_ast_convertible:
    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'],keep='first').reset_index(drop=True)

    prev_subject_id = 0
    subjectid_appendix = []
    timesteps = []
    
    for i in tqdm(range(len(df)), desc="splitting students' records ..."):
        if prev_subject_id != df.iloc[i].SubjectID:
            # when encountering a new student ID
            prev_subject_id = df.iloc[i].SubjectID
            accumulated = 0
            id_appendix = 1
        else:
            accumulated += 1
            if accumulated >= max_len:
                id_appendix += 1
                accumulated = 0
        timesteps.append(accumulated)
        subjectid_appendix.append(id_appendix)
    df['timestep'] = timesteps
    df['SubjectID_appendix'] = subjectid_appendix
    df['SubjectID'] = [df.iloc[i].SubjectID + '_{}'.format(df.iloc[i].SubjectID_appendix) for i in range(len(df))]

    students = df['SubjectID'].unique()

    train_stu, test_stu = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
    valid_stu, test_stu = train_test_split(test_stu, test_size=0.5, random_state=configs.seed)
    return train_stu, valid_stu, test_stu, df, students


def make_pytorch_dataset(students, dataset):
    lstm_student = []

    for student in students:
        subset = dataset[dataset['SubjectID'] == student]
        subset.loc[:, 'prompt-embedding'] = subset['prompt-embedding'].apply(lambda x: torch.tensor(x))
        data_dict = {
            'SubjectID': student,
            'ProblemID_seq': subset.ProblemID.tolist(),
            'Score': subset.Score.tolist(),
            'prompt-embedding': subset['prompt-embedding'].tolist(),
            'input': subset.input.tolist(),
            'KC': subset['knowledge_component'].tolist(),
            'next_prompt': subset.prompt.tolist(),
            'next_code': subset.Code.tolist(),
        }
        lstm_student.append(data_dict)

    return lstm_student

def make_dataloader(students, dataset, collate_fn, configs, shuffle=False):
    lstm_student = make_pytorch_dataset(students, dataset)
    data_loader = torch.utils.data.DataLoader(lstm_student, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size)
    return data_loader


def build_prompt_with_special_tokens(prompt, kcs):
    if ":" in prompt:
        prompt = prompt.replace(":", "")

    if "?" in prompt:
        prompt = prompt.replace("?", ".")

    assert "student written code" not in prompt

    prompt = "Question: " + prompt
    for i in range(len(kcs)):
        kc = kcs[i]
        kc_intro = f" KC {i+1}: {kc}."
        kc_level = f" The student's mastery level on {kc} is ?"
        prompt += kc_intro + kc_level

    prompt += " Student written code:"

    return prompt


def build_input_with_special_tokens(prompt, kcs, code, tokenizer):
    input = build_prompt_with_special_tokens(prompt, kcs) + " " + code.strip() + tokenizer.eos_token
    return input


class CollateForKC(object):
    def __init__(self, tokenizer, configs, device, kc_dict, eval=False):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" if eval else "right"

        self.configs = configs
        self.device = device
        self.delimiter_token_id = tokenizer.convert_tokens_to_ids("Ġwritten")
        self.level_token_id = tokenizer.convert_tokens_to_ids('Ġ?')
        self.kc_dict = kc_dict
        self.eval = eval


    def __call__(self, batch):
        scores = [b['Score'] for b in batch]
        max_len = max([len(i) for i in scores])
        padded_scores = [i + [-100] * (max_len - len(i)) for i in scores]
        padded_scores = torch.Tensor(padded_scores).t().to(self.device)  # shape: (T, B)

        question_seqs = [b['ProblemID_seq'] for b in batch]
        # question_seqs = [[self.question_no_map[i] for i in seqs] for seqs in question_seqs]
        padded_question_seqs = [i + [-100]*(max_len - len(i)) for i in question_seqs]
        padded_question_seqs = torch.tensor(padded_question_seqs) # shape: (B, T)

        inputs = [b['input'] for b in batch]
        padded_inputs = [i + [torch.zeros(i[0].shape[0])] * (max_len - len(i)) for i in inputs]
        padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float().to(self.device)  # shape: (T, B, D)

        kcs_name = [b['KC'] for b in batch]
        kcs = [[[self.kc_dict[elem] for elem in kc_i] for kc_i in sub_ls] for sub_ls in kcs_name]
        max_kc_inner_len = max([len(inner) for kc_i in kcs for inner in kc_i])
        max_kc_len = max(len(kc_i) for kc_i in kcs)


        def pad_inner(inner, max_len_inner):
            return inner + [-1] * (max_len_inner - len(inner))

        
        padded_kc = []
        for kc_i in kcs:
            padded_inner = [pad_inner(i, max_kc_inner_len) for i in kc_i]
            padded_inner += [[-1]*max_kc_inner_len] * (max_kc_len - len(kc_i))
            padded_kc.append(padded_inner)

        padded_kc = torch.tensor(padded_kc).float().to(self.device)

        padded_kc = padded_kc.transpose(0, 1)   # shape: (T, B, max_single_kc_len)


        codes = [b['next_code'] for b in batch]

        students = []
        for i in range(len(batch)):
            stu_name = batch[i]['SubjectID']
            student_ls = [stu_name] * len(codes[i])
            students.append(student_ls)

        padded_students = [i + [''] * (max_len - len(i)) for i in students]
        stacked_students = list(map(list, zip(*padded_students)))

        padded_codes = [i + [''] * (max_len - len(i)) for i in codes]
        stacked_codes = list(map(list, zip(*padded_codes)))

        prompts = [b['next_prompt'] for b in batch]
        padded_prompts = [i + [''] * (max_len - len(i)) for i in prompts]
        stacked_prompts = list(map(list, zip(*padded_prompts)))

        if self.eval:
            input_texts = [[build_prompt_with_special_tokens(prompt_i, kc_i) for prompt_i, kc_i in
                            zip(entry['next_prompt'], entry['KC'])] for entry in batch]
        else:
            input_texts = [[build_input_with_special_tokens(prompt_i, kc_i, code_i, self.tokenizer) for
                            prompt_i, kc_i, code_i in zip(entry['next_prompt'], entry['KC'], entry['next_code'])] for entry in batch]

        inputs_ids_ls, attention_mask_ls, labels_ls, prompt_id_lens_ls, level_loc_ls = [], [], [], [], []

        for input_sub in input_texts:
            inputs = self.tokenizer(input_sub, return_tensors='pt', padding=True, truncation=True)
            inputs_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)

            if not self.eval:
                inputs_ids[:, -1] = self.tokenizer.eos_token_id

            delimiter_indices = torch.where(inputs_ids == self.delimiter_token_id, 1, 0)
            prompt_id_lens = torch.argmax(delimiter_indices, dim=-1)
            prompt_id_lens = prompt_id_lens + 3


            labels = inputs_ids.detach().clone()
            labels = labels.masked_fill((attention_mask == 0), -100)
            range_tensor = torch.arange(inputs_ids.size(1), device=self.device).unsqueeze(0)
            range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1)
            mask_tensor = (range_tensor < prompt_id_lens.unsqueeze(-1))
            labels[mask_tensor] = -100

            inputs_ids_ls.append(inputs_ids)
            attention_mask_ls.append(attention_mask)
            labels_ls.append(labels)
            prompt_id_lens_ls.append(prompt_id_lens)

            # Finding mastery level inserting position
            col_indices = torch.arange(inputs_ids.shape[1]).unsqueeze(0)
            valid_mask = col_indices.to(self.device) < prompt_id_lens.unsqueeze(1)

            matches = (inputs_ids == self.level_token_id) & valid_mask
            level_sent_ind, level_token_ind = torch.nonzero(matches, as_tuple=True)

            mask = level_sent_ind != 0
            filtered_sent_ind = level_sent_ind[mask] - 1  # Subtract 1 from remaining indices
            filtered_token_ind = level_token_ind[mask]
            
            level_loc_ls.append((filtered_sent_ind, filtered_token_ind))

        max_length = max([sub.shape[1] for sub in inputs_ids_ls])

        padded_input_ids_ls = [torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]),
                               value=self.tokenizer.eos_token_id) for input_ids in inputs_ids_ls]
        padded_input_ids_ls = pad_sequence(padded_input_ids_ls, batch_first=True,
                              padding_value=self.tokenizer.eos_token_id)  # shape: (B, T, max_length)


        padded_attention_mask_ls = [torch.nn.functional.pad(attention_mask, (0, max_length - attention_mask.shape[1]),
                                    value=0) for attention_mask in attention_mask_ls]
        padded_attention_mask_ls = pad_sequence(padded_attention_mask_ls, batch_first=True, padding_value=0)
        padded_attention_mask_ls = torch.transpose(padded_attention_mask_ls, 0, 1) # shape: (T, B, max_length)

        padded_labels_ls = [torch.nn.functional.pad(labels, (0, max_length - labels.shape[1]), value=-100) for labels in
                            labels_ls]
        padded_labels_ls = pad_sequence(padded_labels_ls, batch_first=True, padding_value=-100)
        padded_labels_ls = torch.transpose(padded_labels_ls, 0, 1)  # shape: (T, B, max_length)

        padded_prompt_id_lens_ls = [torch.cat((i, torch.zeros(max_len - i.size(0)).to(self.device)), 0) for i in
                                    prompt_id_lens_ls]
        padded_prompt_id_lens_ls = torch.stack(padded_prompt_id_lens_ls).t()  # shape: (T, B)

        if self.eval:
            return padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, stacked_codes, stacked_prompts, padded_scores, stacked_students, padded_kc, level_loc_ls, padded_question_seqs

        return padded_scores, padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, padded_labels_ls, padded_prompt_id_lens_ls, padded_kc, level_loc_ls

