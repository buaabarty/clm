import codecs
import os
import sys
import json
import torch
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM


INCODER_FINETUNE_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = INCODER_FINETUNE_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_incoder_finetune_input(buggy_file, rem_start, rem_end, tmp_file):
    os.chdir(JAVA_DIR)
    command([
        'java', '-cp', '.:target:lib/*', 'clm.finetuning.FineTuningData', 'inference',
        buggy_file, str(rem_start), str(rem_end), tmp_file
    ])

def defects4j_incoder_finetune_input(output_file, tmp_dir):
    loc_fp = codecs.open(INCODER_FINETUNE_DIR + '../defects4j/defects4j_loc.txt', 'r', 'utf-8')
    incoder_input = {'config': 'finetune', 'data': {}}
    for line in loc_fp.readlines():
        proj, bug_id, path, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = INCODER_FINETUNE_DIR + '../defects4j/tmp.json'

        subprocess.run(['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', tmp_dir], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        get_incoder_finetune_input(tmp_dir + path, start, end, tmp_file)
        
        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        if result["buggy function before"].strip() == '' and result["buggy line"].strip() == '' and result["buggy function after"].strip() == '':
            print(proj, bug_id, 'failed. all empty.')
            continue
        incoder_input['data'][proj + '_' + bug_id + '_' + path + '_' + rem_loc] = {
            'loc': rem_loc,
            'input': result['buggy function before'] + '// buggy lines start:\n' + result['buggy line'] + '// buggy lines end:\n' + result['buggy function after'] + '// fixed lines:\n',
        }
        command(['rm', '-rf', tmp_file])
        command(['rm', '-rf', tmp_dir])
        json.dump(incoder_input, open(output_file, 'w'), indent=2)

def defects4j_incoder_finetune_output(input_file, output_file, model_dir, model_name, num_output=10):
    tokenizer = AutoTokenizer.from_pretrained(model_dir + model_name[:-9])
    model = AutoModelForCausalLM.from_pretrained(model_dir + model_name)
    model.parallelize(device_map)
    
    incoder_output = json.load(open(input_file, 'r'))
    incoder_output['model'] = model_name
    for filename in incoder_output['data']:
        text = incoder_output['data'][filename]['input']

        print('generating', filename)
        input_ids = tokenizer(text, return_tensors="pt").input_ids.to(0)
        if input_ids.size(1) >= 1024:
            print('too long:', input_ids.size(1))
            continue

        try:
            eos_id = tokenizer.convert_tokens_to_ids('<|endofmask|>')
            generated_ids = model.generate(
                input_ids, max_new_tokens=128, num_beams=num_output, num_return_sequences=num_output, early_stopping=True,
                pad_token_id=eos_id, eos_token_id=eos_id
            )
        except Exception as e:
            continue
        output = []
        for generated_id in generated_ids:
            output.append(tokenizer.decode(generated_id, skip_special_tokens=False))
        incoder_output['data'][filename]['output'] = output
        json.dump(incoder_output, open(output_file, 'w'), indent=2)


if __name__ == '__main__':
    model_dir = sys.argv[1]
    
    input_file = INCODER_FINETUNE_DIR + '../defects4j/incoder_finetune_result/incoder_input.json'
    print("==========Preparing input of Defects4J benchmark to finetuned INCODER model==========")
    defects4j_incoder_finetune_input(input_file, tmp_dir='/tmp/incoder/')
    print("==========Input written to " + input_file)
    
    for model_name in ('incoder-1B-finetune', 'incoder-6B-finetune'):
        if model_name == 'incoder-1B-finetune':
            device_map = {
                0: [_ for _ in range(0, 5)],
                1: [_ for _ in range(5, 12)],
                2: [_ for _ in range(12, 19)],
                3: [_ for _ in range(19, 24)]
            }
            device_ids = list(device_map.keys())    # need 4 GPUs with 4*12 GB memory in total to run incoder-1B
        else:
            device_map = {
                0: [_ for _ in range(0, 4)], 
                1: [_ for _ in range(4, 8)],
                2: [_ for _ in range(8, 12)],
                3: [_ for _ in range(12, 16)],
                4: [_ for _ in range(16, 20)],
                5: [_ for _ in range(20, 24)],
                6: [_ for _ in range(24, 28)],
                7: [_ for _ in range(28, 32)]
            }
            device_ids = list(device_map.keys())    # need 8 GPUs with 8*12 GB memory in total to run incoder-6B 
                                                    # (the author use 4 A5000 GPUs with 4*24 GB memory to run)
        output_file = INCODER_FINETUNE_DIR + '../defects4j/incoder_finetune_result/' + '_'.join(model_name.split('-')[:-1]) + '_output.json'
        print("==========Generating output of Defects4J benchmark by " + model_name + "==========")
        defects4j_incoder_finetune_output(input_file, output_file, model_dir, model_name, num_output=10)
        print("==========Output written to " + output_file)
