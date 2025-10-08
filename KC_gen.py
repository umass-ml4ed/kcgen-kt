import os
import json
import openai
from openai import OpenAI
import code_ast
from data_loader import *
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
import ast
from collections import defaultdict
import re
from collections import Counter
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from collections import Counter, defaultdict

# Generate 2 sample solutions for each question
def get_sample_solution(prompt, model='gpt-4o-mini', temperature=1, max_tokens=400):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    try:
        user_prompt = "Problem: " + prompt[0] + " Method name and parameters: " + prompt[1]
        
        system_content = """You are an experienced computer science teacher and education expert. You are given a Java programming problem with the method name and parameter. Your job is to generate two sample solution code to solve the problem. Please follow these instructions carefully when generating the solutions:
- Each solution must be different in approach.
- Do not include comments in the solutions.
- Return the solutions in a JSON object using the template: {"code 1": code_1, "code 2": code_2}.
"""
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt}
            ],
            temperature = temperature,
            max_tokens = max_tokens,
            n=1
        )

        reply = response.choices[0].message.content.strip()
        reply_json = json.loads(reply)

        return reply_json
    except Exception as e:
        print(e)
        return None


def get_incorrect_exmaple(df, random_draw=True, k=1):
    # Find incorrect student submission
    incorrect_df = df[df['Score_x'] < 1]

    if random_draw or k == 1:
        inc_dict = incorrect_df.groupby('prompt', group_keys=False).apply(lambda x: x.sample(n=k, random_state=42)['Code'].tolist()).to_dict()

    else:
        code_groups = incorrect_df.groupby('prompt')

        inc_dict = {}
        cnt = 1
        for name, group in code_groups:
            code_ls = group['Code'].tolist()
            selected_sol = find_solution_by_cluster("microsoft/graphcodebert-base", code_ls, k)

            inc_dict[name] = selected_sol
            print(f'Solution group {cnt} sampled')
            cnt += 1

    return inc_dict


# Save sample solutions for each problem
def save_problem_solutions(unique_problems, model='gpt-4o-mini', temperature=1, max_tokens=400):
    solution_dict = {}
    cnt = 0
    for problem, method_sig in unique_problems.items():
        solutions_i = get_sample_solution([problem, method_sig], model=model, temperature=temperature, max_tokens=max_tokens)
        solution_list = list(solutions_i.values())
        if len(solution_list) != 2 or solution_list[0] == solution_list[1]:
            print('Invalid format response: ', problem)
            cnt += 1
        else:
            solution_dict[problem] = solution_list

    print('Invalid question cnt:', cnt)
    with open('solution.pkl', 'wb') as f:
        pickle.dump(solution_dict, f)



# Few shot KC generation for each problem
def get_KCs_few_shot(problem, language, code_ls, exam_dict, model='gpt-4o', temperature=0):
    try:
        # Using code in the original pipeline
        user_prompt = "Now analyze the following problem using their solution code. \n# Problem:\n" + problem + "\n\n" + "\n".join([f"## Sample solution {i+1}:\n{sol}" for i, sol in enumerate(code_ls)]) + "\n\nFollow the instructions in system message. First, carefully examine the solutions and identify the important elements and patterns. Then, explicitly reason about what underlying knowledge components are required based on these solution codes. Finally, take the examples as reference and summarize your analysis clearly into generalizable and concise knowledge components."


        exam_prompt = ""
        for idx, (problem, kc_list) in enumerate(exam_dict.items(), 1):
            kc_json = {f"KC {i+1}": kc for i, kc in enumerate(kc_list)}
            kc_str = ",\n  ".join([f'"{k}": "{v}"' for k, v in kc_json.items()])
            example_block = f"""# Example {idx}:
## Problem:
{problem}

## Expected KCs:
{{
  {kc_str}
}}

"""
            exam_prompt += example_block
        
        exam_prompt += user_prompt


        system_content = f"""You are an experienced computer science teacher and education expert. You are given a {language} programming problem along with {len(code_ls)} sample solutions. Your task is to identify generalizable knowledge components (skills or concepts) necessary to solve such problems.

A knowledge component (KC) is a single, reusable unit of programming understanding, such as a language construct, pattern, or skill, that contributes to solving a programming problem and can be learned or mastered independently.

Please follow these steps:
1. Analyze each solution carefully, noting critical constructs.
2. Reflect step by step on how each solution maps to distinct programming KCs that are independent and reusable.
3. For each KC, generate a concise name and provide a one-sentence reasoning explaining why this KC is necessary based on the provided solutions. Use the provided examples as reference for the appropriate level of detail. Make sure KCs are generalizable and applicable to a wide range of similar programming problems without referencing problem-specific details.
4. Ensure each KC is atomic and not bundled with others.
 

Your final response must strictly follow this JSON template:
{{
    "KC 1": {{"reasoning": "Reasoning for this KC (exactly 1 sentence)", "name": "Knowledge component name"}},
    "KC 2": {{"reasoning": "Reasoning for this KC (exactly 1 sentence)", "name": "Another specific knowledge component name"}},
    ...
}}"""

        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": exam_prompt}
        ],
            temperature = temperature,
            n=1
        )

        reply = response.choices[0].message.content.strip()
        reply_ls = json.loads(reply)

        kc_dict = reply_ls.values()
        kcs = [val['name'] for val in kc_dict]

        for kc_i in kcs:
            if 'KC' in kc_i:
                set_trace()
                print('Invalid kc generated', kcs)


        return reply_ls

    except Exception as e:
        print(e)
        return None


def get_KCs_few_shot_with_inc(problem, correct_ls, incor_ls, exam_dict, model='gpt-4o', temperature=0):
    try:
        user_prompt = "Now analyze the following problem using both correct and incorrect student solutions.\nProblem:\n" + problem + "\n\n" + "\n".join([f"Correct solution {i+1}:\n{sol}" for i, sol in enumerate(correct_ls)]) + '\n' + "\n".join([f"Incorrect solution {i+1}:\n{sol}" for i, sol in enumerate(incor_ls)]) + "\nLet's think step by step. First, carefully examine the solutions to identify critical constructs and recurring mistakes. Then, explicitly reason about what programming skills or knowledge components are demonstrated in the correct solutions, and what conceptual misunderstandings or missing knowledge are evident in the incorrect ones. Finally, take the examples as reference and use these insights to produce a set of high-granularity, generalizable knowledge components applicable to similar programming tasks."


        exam_prompt = ""
        for idx, (problem, kc_list) in enumerate(exam_dict.items(), 1):
            kc_json = {f"KC {i+1}": kc for i, kc in enumerate(kc_list)}
            kc_str = ",\n  ".join([f'"{k}": "{v}"' for k, v in kc_json.items()])
            example_block = f"""Example {idx}:
Problem:
{problem}

Expected Output:
{{
  {kc_str}
}}

"""
            exam_prompt += example_block
        
        exam_prompt += user_prompt

        system_content = f"""You are an experienced computer science teacher and education expert. You are given a Java programming problem along with {len(correct_ls)} correct solutions and {len(incor_ls)} incorrect student submissions. Your task is to identify and articulate generalizable knowledge components (skills or concepts) necessary to solve such problems, including those revealed by common mistakes.

Please follow these steps:
1. Analyze all provided solutions, both correct and incorrect, paying close attention to critical constructs, logical patterns, and common errors.
2. Reflect step-by-step on how each solution demonstrates distinct programming knowledge components that are reusable, and how incorrect submissions reflect misunderstandings or missing conceptual knowledge.
3. Generate atomic knowledge components based on both the successful approaches and the observed mistakes. Each KC should represent a single concept, avoiding bundling.
4. For each knowledge component, provide a one-sentence reasoning that explains why this component is necessary or how its absence leads to the observed error.
5. When identifying skills from incorrect submissions, consider what conceptual gap or misunderstanding the error reveals.
6. The granularity of knowledge components need to be high, take the knowledge component in the provided examples as reference.

Your final response must strictly follow this JSON template:
{{
    "KC 1": {{"name": "Knowledge component name", "reasoning": "Reasoning for this KC (exactly 1 sentence)"}},
    "KC 2": {{"name": "Another specific knowledge component name", "reasoning": "Reasoning for this KC (exactly 1 sentence)"}},
    ...
}}

Make sure knowledge components are generalizable and applicable to a wide range of similar programming problems without referencing problem-specific details."""

        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": exam_prompt}
        ],
            temperature = temperature,
            n=1
        )

        reply = response.choices[0].message.content.strip()
        reply_ls = json.loads(reply)

        kc_dict = reply_ls.values()
        kcs = [val['name'] for val in kc_dict]

        for kc_i in kcs:
            if 'KC' in kc_i:
                set_trace()
                print('Invalid kc generated', kcs)


        return reply_ls

    except Exception as e:
        print(e)
        return None


# Get the final KC set by union of all unique questions
def get_KC_set(solution_dict, dataset, incorrect_dict, example_dict, model='gpt-4o', temperature=0):
    problem_kc_dict = {}

    if dataset == 'CodeWorkout':
        sel_1 = 'Write a function in Java that implements the following logic: Given a string str and a non-empty word, return a version of the original string where all chars have been replaced by pluses (+), except for appearances of the word which are preserved unchanged.'
        sel_2 = 'Write a function in Java that implements the following logic: You are driving a little too fast, and a police officer stops you. Write code to compute the result, encoded as an int value: 0=no ticket, 1=small ticket, or 2=big ticket. If speed is 60 or less, the result is 0. If speed is between 61 and 80 inclusive, the result is 1. If speed is 81 or more, the result is 2. Unless it is your birthday--on that day, your speed can be 5 higher in all cases.'
        language = 'Java'
    else:
        sel_1 = "Write a Python program that calculates only, but DOES NOT print, the average SAT score across all schools, and outputs the name of each school that is below the average. This work makes use of the sat.csv dataset, which describes the average SAT performance for students attending 350+ schools."
        sel_2 = "You have been asked to write a program that analyzes the results acquired from several flight tests executed last week. Write a program that gets from the user the number flight tests performed, and gets the result from each flight (which you may assume are whole numbers) from the user. Your program must print out: The average result for the flight tests. The count of flight tests that scored below the average."
        language = 'Python'

    selected = [sel_1, sel_2]

    sel_exam_dict = {sel: example_dict[sel] for sel in selected}


    cnt = 1
    for problem, solutions in solution_dict.items():
        n_sol = len(solutions)
        process_i = f'Generating KCs for Q{cnt}'
        print(process_i)
        cnt += 1

        # KCs with selected P-KC mapping few shot by both correct and incorrect codes
        if incorrect_dict:
            KCs = get_KCs_few_shot_with_inc(problem, solutions, incorrect_dict[problem], sel_exam_dict, model, temperature)
        
        else:
            # KCs with selected P-KC mapping few shot by code
            KCs = get_KCs_few_shot(problem, language, solutions, sel_exam_dict, model, temperature)
            
        kc_dict = KCs.values()
        kcs = [val['name'] for val in kc_dict]
        problem_kc_dict[problem] = kcs

    
    kc_set = problem_kc_dict.values()
    kc_ls = [val for value in kc_set for val in value]
    print('Total KCs across problems:', len(kc_ls))
    kc_ls = set(kc_ls)
    print('Unique KCs across problems:', len(kc_ls))

    check = [len(i) for i in problem_kc_dict.values()]
    print('Average Kcs per question:', sum(check) / len(check))
    set_trace()

    check = sorted(list(kc_ls))
    for item in check:
        print(item)


    if incorrect_dict:
        file_name = f'problem_kc_{n_sol}_{len(incorrect_dict)}.json'
    else:
        file_name = f'problem_kc_{n_sol}_{dataset}.json'

    with open(file_name, "w") as res: 
        json.dump(problem_kc_dict, res)

    return problem_kc_dict


def check_solution_format(file):
    with open(file, 'rb') as f:
        solution = pickle.load(f)
        cnt = 0
        for key, val in solution.items():
            for elem in val:
                if elem[-1] != '}':
                    print(val)
                    cnt += 1

        if cnt > 0:
            print("Invalid solution exists")


def KC_cluster_linkage(KC_list, encoder_name, n_cluster):
    encoder = SentenceTransformer(encoder_name)
    KC_emb = encoder.encode(KC_list, convert_to_numpy=True)

    distance_matrix = pdist(KC_emb, metric='cosine')
    Z = linkage(distance_matrix, method='average')

    labels = fcluster(Z, t=n_cluster, criterion='maxclust')

    cluster_to_kcs = defaultdict(list)
    for kc_text, cluster_id in zip(KC_list, labels):
        cluster_to_kcs[cluster_id].append(kc_text)

    return cluster_to_kcs


def KC_cluster_summarize(kc_list, model='gpt-4o', temperature=0, n=1):
    try:
        user_prompt = "The knowledge components list is:\n" + json.dumps(kc_list) + "\n\nNow follow the instructions in system message. First, examine the list carefully to understand their shared meaning. Second, explicitly reason about the fundamental skill or concept underlying these knowledge components. Third, based on the reasoning, either select one KC that best represents the group if they share the same concept or summarize your analysis into one clear and concise phrase that accurately captures the essence of this cluster."

        system_content = """You are an experienced computer science teacher and education expert. You will be provided with a list of knowledge components (KCs) that may vary in wording but sometimes refer to the same or related underlying concepts or skills.

The KCs will be given in the format: ["KC 1 name", "KC 2 name", ..., "KC k name"]

Your task is to:
1. Carefully examine all the KCs in the list to ensure none are overlooked.
2. Reason explicitly whether the KCs collectively refer to the same underlying concept or skill, or if they are related but represent distinct or complementary aspects of a broader theme.
3. Based on your reasoning:
    - If the KCs refer to the same concept or skill, select one KC from the list that best represents the group — choose the one that is most clearly worded, generalizable, and inclusive of the others.
    - If the KCs are related but too distinct to be represented by a single KC, create a brief and meaningful summary name that captures the broader theme shared by the KCs. The summary name must not contain the word “and”.

Return your output strictly in the following JSON format:
{
  "reasoning": "...",        // Exactly one sentence explaining your reasoning
  "representative_kc": "..." , // Selected KC if applicable, otherwise null
  "summary_name": "..." ,       // Summary name if representative KC not chosen, otherwise null
}
"""
    
        response = openai.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_content},
                {'role': 'user', 'content': user_prompt},
            ],
            temperature=temperature,
            n=n
        )

        reply = response.choices[0].message.content.strip()
        kc = json.loads(reply)

        representative = kc['representative_kc'] if kc['representative_kc'] is not None else kc['summary_name']

        return representative.lower()

    except Exception as e:
        print(e)
        return ''


def KC_summarize(kc_dict, model='gpt-4o', temperature=0, n=1):
    kc_summarized, map_dict = {}, {}
    cnt = 0

    track_same_dict = defaultdict(list)

    for key, val in kc_dict.items():
        if len(val) > 1:
            kc_name = KC_cluster_summarize(val, model, temperature, n)
        else:
            kc_name = val[0].lower()

        if kc_name:
            kc_summarized[key] = kc_name

            for kc_i in val:
                map_dict[kc_i] = kc_name

            track_same_dict[kc_name].append(val)
            
            print(f'Processed: {cnt}')

        else:
            print(f'Invalid KC summarization: {cnt}')
        
        cnt += 1

    kc_cnt = sum([len(val) for val in track_same_dict.values()])
    print('KC Count:', kc_cnt)
    print('Unique KC Count:', len(track_same_dict))

    set_trace()
    for idx, (key, val) in enumerate(track_same_dict.items()):
        print(f"{idx}: {key}")
        for item in val:
            print(item)
            print('-------------------------')

    print('Done')


    return kc_summarized, map_dict



def find_solution_by_cluster(model_path, code_ls, k, file_path):
    # # Uncomment this part to save the code embedding
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model = AutoModel.from_pretrained(model_path)

    # model.eval()
    # inputs = tokenizer(code_ls, return_tensors="pt", padding=True, truncation=True)

    
    # with torch.no_grad():
    #     outputs = model(**inputs)

    # embeddings = outputs.last_hidden_state

    # # # Use CLS token for embedding
    # # cls_embedding = embeddings[:, 0, :]

    # # Use mean pooling for embedding
    # attention_mask = inputs["attention_mask"]
    # mask_expand = torch.unsqueeze(attention_mask, -1)
    # hidden_states_question = embeddings * mask_expand
    # sum_embeddings = torch.sum(hidden_states_question, dim=1)
    # sum_mask = torch.clamp(mask_expand.sum(1), min=1e-9)
    # mean_embeddings = sum_embeddings / sum_mask

    # # Convert to numpy for clustering
    # embeddings_np = mean_embeddings.numpy()
    # embeddings_norm = normalize(embeddings_np)

    file_name = f"code_embedding/embeddings_problem_{file_path}.npy"
    # np.save(file_name, embeddings_norm)
    embeddings_norm = np.load(file_name)

    clusterer = AgglomerativeClustering(
        n_clusters=k, 
        metric='cosine',
        linkage='average',
    )


    labels = clusterer.fit_predict(embeddings_norm)
    # print("Cluster Labels:", labels)

    code_dict = pd.DataFrame({'code': code_ls, 'label': labels})
    groups = code_dict.groupby('label')

    selected_sol = []

    for name, group in groups:
        code_i = group['code'].tolist()
        selected = random.sample(code_i, 1)
        selected_sol.append(selected[0])

    return selected_sol


def student_solution_construct(df, k, random=True):
    ## randomly sample
    if random:
        solution_dict = df.groupby('prompt', group_keys=False).apply(lambda x: x.sample(n=k, random_state=42)['Code'].tolist()).to_dict()

    # cluster correct codes by semantic and structure, try to find different solutions in this way
    else:
        code_groups = df.groupby('prompt')

        problem_id_dict = df.set_index('prompt')['ProblemID'].to_dict()

        solution_dict = {}
        cnt = 1
        for name, group in code_groups:
            code_ls = group['Code'].tolist()
            # print(f'No. {cnt} Problem {problem_id_dict[name]} started sampling')
            selected_sol = find_solution_by_cluster("microsoft/graphcodebert-base", code_ls, k, problem_id_dict[name])

            solution_dict[name] = selected_sol
            
            cnt += 1
    
    with open('sample_sol_dict.json', 'w') as f: 
        json.dump(solution_dict, f)

    return solution_dict



def extract_method_nodes(root_node):
    for child in root_node.children:
        if child.type == "class_declaration":
            for class_child in child.children:
                if class_child.type == "class_body":
                    return [
                        item for item in class_child.children 
                        if item.type == "method_declaration"
                    ]
    return []

def linkage_cluster_mapping(problem_kc_dict, map_dict):
    final_dict = {}

    for key, val in problem_kc_dict.items():
        new_mapping = set([map_dict[kc_i] for kc_i in val])
        final_dict[key] = list(new_mapping)
    
    return final_dict


def kc_mapping_pipeline(df, dataset, model='gpt-4o', temperature=0, n_correct=5, n_cluster=None, random_draw=True, n_incor=0, random_inc=True):
    incor_dict = None
    if n_incor > 0:
        incor_dict = get_incorrect_exmaple(df, random_draw=random_inc, k=n_incor)

    if dataset == 'CodeWorkout':
        kc_problem_dict_basleine, kc_no_dict_baseline = extract_baseline_kc('data/prompt_concept.csv')

        nat_lan_ls = ['If and else statement', 'Nested if statement', 'While loop', 'For loop', 'Nested for loop', 'Basic arithmetic operations', 'Modulus operation', 'Logical operators', 'Numerical comparisons', 'Boolean operations', 'String formatting', 'String concatenation', 'String indexing', 'String length', 'String equality comparison', 'Character equality comparison', 'Array indexing', 'Function definition']

        kc_baseline = list(kc_no_dict_baseline.keys())
        kc_dict_baseline = {kc_baseline[i]: nat_lan_ls[i] for i in range(len(kc_baseline))}

        kc_baseline_natural = {}
        for key, val in kc_problem_dict_basleine.items():
            nat_ls = [kc_dict_baseline[it] for it in val]
            kc_baseline_natural[key] = nat_ls

    else:
        kc_problem_dict_basleine, kc_no_dict_baseline = extract_falcon_baseline_kc('data/problems_falcon_4.csv')

        kc_baseline_natural = {
            "Write a Python program that calculates only, but DOES NOT print, the average SAT score across all schools, and outputs the name of each school that is below the average. This work makes use of the sat.csv dataset, which describes the average SAT performance for students attending 350+ schools.": ["Output generation", "Variable assignment", "Conditional statement", "Function invocation", "Iterating over elements", "math calculation", "File reading", "List usage"],
            "You have been asked to write a program that analyzes the results acquired from several flight tests executed last week. Write a program that gets from the user the number flight tests performed, and gets the result from each flight (which you may assume are whole numbers) from the user. Your program must print out: The average result for the flight tests. The count of flight tests that scored below the average." : ["Type conversion", "Output generation", "Variable assignment", "Conditional statement", "Counting loop", "Iterating over elements", "Math calculation", "List usage"]
        }


    ## Generating solutions by LLM
    # if gen_sol:
    #     unique_problems = get_method_dec(df)
    #     save_problem_solutions(unique_problems)

    ## Selecting solutions from correct student submission
    if dataset == 'CodeWorkout':
        correct_sub = df[df['Score_x'] == 1.0]
    else:
        correct_sub = df[df['Score'] == 1]

    if n_correct == 1:
        sampled_solution = student_solution_construct(correct_sub, n_correct, random=True)
    else:
        sampled_solution = student_solution_construct(correct_sub, n_correct, random=random_draw)


    # # Uncomment to load the generated solution dictionary, otherwise, choose correct solution from student submission
    # with open('solution.pkl', 'rb') as sol:
    #     unique_problems = pickle.load(sol)
    #     set_trace()
    #     get_KC_set(unique_problems, incor_dict, kc_baseline_natural, model=model, temperature=temperature)


    kc_problem_dict = get_KC_set(sampled_solution, dataset, incor_dict, kc_baseline_natural, model=model, temperature=temperature)

    ## Process to cluster the KCs and summarize into different abstraction level
    if n_cluster:
        file_name = f'problem_kc_{n_correct}_{dataset}.json'

        with open(file_name, 'r') as f:
            kc_problem_dict = json.load(f)
            kc_set = kc_problem_dict.values()
            kc_ls = set([val for value in kc_set for val in value])
            kc_ls = list(kc_ls)

            # scipy linkage for hierarchical clustering
            kc_dict = KC_cluster_linkage(kc_ls, 'all-MiniLM-L6-v2', n_cluster)

            kc_summarized, map_dict = KC_summarize(kc_dict, model=model, temperature=temperature)
    
            kc_problem_dict = linkage_cluster_mapping(kc_problem_dict, map_dict)

            # file name info order: No. of correct sub, No. of clusters/kcs
            file_name = f'problem_kc_{n_correct}_{n_cluster}_{dataset}.json'
            print(file_name)

            with open(file_name, "w") as res:
                json.dump(kc_problem_dict, res)
            

    kc_set = kc_problem_dict.values()
    kc_set = set([val for value in kc_set for val in value])

    if n_cluster:
        kc_total_map = [len(val) for val in kc_problem_dict.values()]
        avg_kc = sum(kc_total_map) / len(kc_problem_dict)
        print('Aerage KCs per problem after clustering:', avg_kc)
        print('')

    for final_kc in sorted(kc_set):
        print(final_kc)

    print('Done')


def get_incorrect_df(df, file, fine_grained=True):
    df = df.drop_duplicates(subset=['SubjectID', 'ProblemID'],keep='first').reset_index(drop=True)

    incorrect_df = df[df['Score_x'] != 1.0]

    if fine_grained:
        problem_kc_map, _ = get_problem_kc_updated(file)
    else:
        problem_kc_map, _ = extract_baseline_kc('data/prompt_concept.csv')

    incorrect_df['knowledge_component'] = incorrect_df['prompt'].map(problem_kc_map)
    return incorrect_df


def main():
    df = pd.read_pickle('data/dataset_time.pkl')

    kc_mapping_pipeline(df, 'CodeWorkout', model='gpt-4o', temperature=0, n_correct=5, n_cluster=75, random_draw=False, n_incor=0, random_inc=True)


main()