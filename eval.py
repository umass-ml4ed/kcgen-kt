import torch
import pickle
import nltk
from nltk import ngrams
import os
import abc
from tqdm import tqdm
from pdb import set_trace
import hydra
import json
from transformers import GenerationConfig
from model import *
import wandb
from sklearn.metrics import f1_score, roc_auc_score, cohen_kappa_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import statistics

from utils import set_random_seed
from trainer import *
from data_loader import *
from evaluator.CodeBLEU import calc_code_bleu
from huggingface_hub import login
from utils import aggregate_metrics
import warnings


def evaluate(configs, now, test_loader, tokenizer, device, kc_no_dict, lang):
    results = {}
    lstm = None
    predictor = None


    model, lstm, predictor, tokenizer, trans_linear = load_model_eval(configs, now, device, len(kc_no_dict))

    tokenizer.padding_side = 'left'

    # Set model to eval mode
    model.eval()
    lstm.eval()

    if configs.transition:
        trans_linear.eval()

    if configs.multitask:
        predictor.eval()

    generated_code_total, gt_code_total, prompt_total, pred_score_total, gt_score_total, student_total = [], [], [], [], [], []

    for idx, batch in enumerate(tqdm(test_loader, desc="inference", leave=False)):
        if configs.multitask:
            generated_code_ls, gt_code_ls, prompt_ls, pred_score_ls, gt_score_ls, student_ls = generate_code_student(batch, tokenizer, model, lstm, configs, device, predict_linear=predictor, trans_linear=trans_linear)
            pred_score_total.append(pred_score_ls)
            gt_score_total.append(gt_score_ls)
        else:
            generated_code_ls, gt_code_ls, prompt_ls, student_ls = generate_code_student(batch, tokenizer, model, lstm, configs, device, predict_linear=predictor, trans_linear=trans_linear)
        
        generated_code_total.append(generated_code_ls)
        gt_code_total.append(gt_code_ls)
        prompt_total.append(prompt_ls)
        student_total.append(student_ls)

    generated_codes = [gen_code_i for code_ls in generated_code_total for gen_code_i in code_ls]
    gt_codes = [gt_code_i for code_ls in gt_code_total for gt_code_i in code_ls]
    prompts = [prompt_i for prompt_ls in prompt_total for prompt_i in prompt_ls]
    students = [student_i for student_ls in student_total for student_i in student_ls]

    if configs.multitask:
        pred_scores = [pred_score_i for pred_subset in pred_score_total for pred_score_i in pred_subset]
        pred_labels = [1 if value > 0.5 else 0 for value in pred_scores]
        gt_scores = [gt_score_i for gt_subset in gt_score_total for gt_score_i in gt_subset]
        

        pred_res = sum([pred == label for pred, label in zip(pred_labels, gt_scores)])
        acc = pred_res / len(gt_scores)
        results['Acc'] = acc

        f1 = f1_score(gt_scores, pred_labels)
        results['F1'] = f1

        auc = roc_auc_score(gt_scores, pred_scores)
        results['AUC'] = auc

    
    codebleu_score, detailed_codebleu_score = compute_code_bleu(gt_codes, generated_codes, lang)
    results['codebleu'] = codebleu_score
    results['detailed_codebleu'] = detailed_codebleu_score
    
    ## compute diversity
    metrics = {'dist_1': Distinct_N(1), 
               'dist_2': Distinct_N(2), 
               'dist_3': Distinct_N(3),
    }
    for i, (name, metric) in enumerate(metrics.items()):
        metric_result = metric.compute_metric(generated_codes)
        results[name] = metric_result

    print(f"results: {results}")

    ## save results
    results['generated_codes'] = generated_codes
    results['ground_truth_codes'] = gt_codes
    results['prompts'] = prompts
    results['students'] = students
   
    
    if configs.save_model:
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.txt'), 'w') as f:
            json.dump(results, f, indent=2)

    return results


def generate_code_student(batch, tokenizer, model, lstm, configs, device, predict_linear=None, trans_linear=None):
    gen_code_ls, gt_code_ls, prompt_ls, gt_score_ls, pred_score_ls, pred_label_ls, student_ls = [], [], [], [], [], [], []
    padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, padded_codes, padded_prompts, padded_scores, padded_students, padded_kc, level_loc_ls, _ = batch[0][:-1], batch[1][:, 1:], batch[2][1:], batch[3][1:], batch[4][1:], batch[5][1:], batch[6][1:], batch[7][1:], batch[8], batch[9][:, 1:]

    input_wte, _ = update_input_weight(padded_input_ids_ls, padded_inputs, padded_kc, level_loc_ls, device, model, lstm, tokenizer, configs.transition, trans_linear, configs.kc_loss_method)

    input_wte = input_wte.to(device=device, dtype=model.dtype)
    padded_attention_mask_ls = padded_attention_mask_ls.to(device=device, dtype=model.dtype)

    T, B, max_length, D = input_wte.shape
    input_wte = input_wte.reshape((T * B), max_length, D)
    padded_attention_mask_ls = padded_attention_mask_ls.reshape((T * B), -1)
    padded_scores = torch.unsqueeze(padded_scores, -1).reshape((T * B), -1)

    flattened_codes = [code_i for subcode in padded_codes for code_i in subcode]
    flattened_prompt = [prompt_i for subprompt in padded_prompts for prompt_i in subprompt]
    flattened_students = [student_i for substudent in padded_students for student_i in substudent]

    input_wte_subset = torch.split(input_wte, 1)
    attention_mask_subset = torch.split(padded_attention_mask_ls, 1)

    if configs.multitask:
        scores_subset = torch.split(padded_scores, 1)

    config = GenerationConfig(max_new_tokens=configs.max_new_tokens, do_sample=False)

    for i in range(len(input_wte_subset)):
        ground_truth_code = flattened_codes[i]
        prompt = flattened_prompt[i]
        student = flattened_students[i]
        if ground_truth_code:
            input_wte_i = input_wte_subset[i]
            attention_i = attention_mask_subset[i]
            
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            outputs = model.generate(inputs_embeds=input_wte_i, max_new_tokens=configs.max_new_tokens, do_sample=False, generation_config=config, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators, attention_mask=attention_i)
            
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # generated_code = ''

            gen_code_ls.append(generated_code.strip())
            gt_code_ls.append(ground_truth_code.strip())
            prompt_ls.append(prompt)
            student_ls.append(student)

            if configs.multitask:
                predicted_score = predict_score_question_only(input_wte_i, attention_i, model, predict_linear)

                pred_score_ls.append(predicted_score.item())
                gt_score_ls.append(scores_subset[i][0][0].cpu().item())


    if configs.multitask:
        return gen_code_ls, gt_code_ls, prompt_ls, pred_score_ls, gt_score_ls, student_ls
    
    return gen_code_ls, gt_code_ls, prompt_ls, student_ls


def predict_score_question_only(generator_input_wte, attention_mask, model, predict_linear):
    eps = 1e-8

    with torch.no_grad():
        output = model(inputs_embeds=generator_input_wte, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = output['hidden_states'][-1]

        attention_sub_expand = torch.unsqueeze(attention_mask, -1)
        hidden_states_valid = hidden_states * attention_sub_expand
        pooled_out = hidden_states_valid.sum(dim=1)
        valid_cnt = attention_sub_expand.sum(dim=1)
        pooled_out = pooled_out / (valid_cnt + eps)

        logits = predict_linear(pooled_out)
        score = torch.sigmoid(logits)

        return score[0][0].cpu()



def compute_code_bleu(ground_truth_codes, generated_codes, lang='java'):
    params='0.25,0.25,0.25,0.25'
    codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    
    return codebleu_score, detailed_codebleu_score


class Metric():
    """
    Defines a text quality metric.
    """
    def get_name(self):
        return self.name


    @abc.abstractmethod
    def compute_metric(self, texts):
        pass


class Distinct_N(Metric):

    def __init__(self, n):
        """
        Distinct n-grams metrics. This is a sequence-level diversity metric.
        See https://www.aclweb.org/anthology/N16-1014 for more details.

        Args:
            n (int): n-grams 
        """
        self.n = n
        self.name = f'Distinct_{n}'


    def compute_metric(self, texts):
        return self._distinct_ngrams(texts, self.n)


    def _distinct_ngrams(self, texts, n):
        total = 0.0
        for t in texts:
            try:
                tokens = nltk.tokenize.word_tokenize(t)
                n_distinct = len(set(ngrams(tokens, n)))
                total += n_distinct/ len(tokens)
            except Exception as e:
                print(f"Exception in computing Distinct_N metric: {e}")
                continue

        return total / len(texts)


@hydra.main(version_base=None, config_path=".", config_name="configs_kc")
def main(configs):
    warnings.filterwarnings("ignore")

    # Make reproducible
    set_random_seed(configs.seed)

    # now = datetime.now().strftime("%Y%m%d_%H%M%S")

    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if configs.use_cuda: assert device.type == 'cuda', 'No GPU found'

    # if configs.log_wandb:
    #     wandb.login(key=configs.wandb_key, verify=True)
    #     wandb.init(project=configs.wandb_project, id="tkqrhhbv", resume="must")

    tokenizer = create_tokenizer(configs)

    if configs.baseline:
        kc_problem_dict, kc_no_dict = extract_baseline_kc('data/prompt_concept.csv')
    else:
        kc_problem_dict, kc_no_dict = get_problem_kc(configs.kc_path)   # Fine-grained kcs (more than 20)
    

    _, _, test_stu, df, _ = read_data('data/dataset_time.pkl', kc_problem_dict, configs)

    collate_fn = CollateForKC(tokenizer, configs, device, kc_no_dict, eval=True)
    test_loader = make_dataloader(test_stu, df, collate_fn, configs)

    print('start eval func:')
    lang = 'java' # when running on CodeWorkout, else set lang to 'python' 

    res = evaluate(configs, now, test_loader, tokenizer, device, kc_no_dict, lang)

    # result = {'codeBLEU': res['codebleu']}
    # result['Acc'] = res['Acc']
    # result['AUC'] = res['AUC']
    # result['F1'] = res['F1']

    # print(result)
    # if configs.log_wandb:
    #     wandb.log(result)
    #     wandb.finish()



if __name__ == "__main__":
    #torch.set_printoptions(profile="full")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    main()