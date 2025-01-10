import codecs
import json
import sys
import os
import re
import subprocess
from codegen_config import CodeGenInputConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
from transformers import pipeline
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    BitsAndBytesConfig, 
    AutoTokenizer
)
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

CODEGEN_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]
JAVA_DIR = CODEGEN_DIR + '../../jasper/'

def command(cmd):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, err = process.communicate()
    if output != b'' or err != b'':
        print(output)
        print(err)
    return output, err

def get_codegen_input(filename, start, end, config, tmp_file):
    os.chdir(JAVA_DIR)
    
    # 添加更多调试信息
    print(f"Current directory: {os.getcwd()}")
    
    # 列出target/classes目录内容
    if os.path.exists('target/classes'):
        print("target/classes contents:")
        os.system('ls -R target/classes')
    
    classpath = f"{JAVA_DIR}/target/classes:{JAVA_DIR}/lib/*"
    cmd = [
        'java', 
        '-verbose:class',  # 添加类加载详细信息
        '-cp', 
        classpath,
        'clm.codegen.CodeGenInputParser',
        filename, start, end, config, tmp_file
    ]
    print(f"Executing command: {' '.join(cmd)}")
    
    output, err = command(cmd)
    return output, err

def defects4j_codegen_input(config, input_file, tmp_dir='/tmp/codegen/'):
    loc_fp = codecs.open(CODEGEN_DIR + '../defects4j/defects4j_loc.txt', 'r', 'utf-8')
    codegen_input = {'data': {}}
    
    for line in loc_fp.readlines():
        proj, bug_id, path, rem_loc, add_loc = line.strip().split()
        start, end = rem_loc.split('-')
        end = str(int(end) - 1) if end != start else end
        tmp_file = CODEGEN_DIR + '../defects4j/tmp.json'
        
        filename = os.path.join('source', path)
        
        # 清理并重新创建临时目录
        if os.path.exists(tmp_dir):
            command(['rm', '-rf', tmp_dir])
        os.makedirs(tmp_dir, exist_ok=True)
            
        # 直接checkout项目，移除init步骤
        result = subprocess.run(
            ['defects4j', 'checkout', '-p', proj, '-v', bug_id + 'b', '-w', '.'], 
            cwd=tmp_dir
        )
        if result.returncode != 0:
            print(f"Error checking out {proj}-{bug_id}")
            continue
            
        output, err = get_codegen_input(tmp_dir + path, start, end, config, tmp_file)
        if output:
            print("Output:", output.decode())
        if err:
            print("Error:", err.decode())
        
        if not os.path.exists(tmp_file):
            print(proj, bug_id, 'failed.', tmp_file, 'not found.')
            continue
        print(proj, bug_id, 'succeeded')

        result = json.load(open(tmp_file, 'r'))
        codegen_input['data'][filename] = {
            'loc': rem_loc,
            'input': result['input'],
            'function range': result['function range']
        }

        command(['rm', '-rf', tmp_file])
        command(['rm', '-rf', tmp_dir])
        json.dump(codegen_input, open(input_file, 'w'), indent=2)
        break

def defects4j_codegen_output(input_file, output_file, num_output=10):
    try:
        import auto_gptq
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([
            sys.executable, 
            "-m", 
            "pip", 
            "install", 
            "--find-links", "https://huggingface.co/AutoGPTQ/AutoGPTQ/tree/main/wheels/",
            "auto-gptq"
        ])
        
    codegen_output = json.load(open(input_file, 'r'))
    
    # 使用命令行参数指定模型路径
    model_path = '/root/autodl-tmp/CodeLlama-13B-Instruct-GPTQ'
    if len(sys.argv) > 1:
        model_path = '/root/autodl-tmp/codellama_finetune/' + sys.argv[1][1:] + '/codellama_merged'
    
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoGPTQForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 创建pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    # 如果指定了GPU设备
    if len(sys.argv) > 4:
        model.to(f'cuda:{sys.argv[4]}')


if __name__ == '__main__':
    for i, config in enumerate(CodeGenInputConfig):
        input_file = CODEGEN_DIR + '../defects4j/codegen_result/codegen_input_c' + str(i + 1) + '.json'
        
        print("==========Preparing input of Defects4J benchmark to Qwen model, Config: " + config + "==========")
        defects4j_codegen_input(config, input_file, tmp_dir='/tmp/codegen/')
        print("==========Input written to " + input_file)
        
        output_file = CODEGEN_DIR + '../defects4j/codegen_result/qwen_output_c' + str(i + 1) + '.json'
        print("==========Generating output of Defects4J benchmark by Qwen2.5-Coder, Config: " + config + "==========")
        defects4j_codegen_output(input_file, output_file)
        print("==========Output written to " + output_file)
