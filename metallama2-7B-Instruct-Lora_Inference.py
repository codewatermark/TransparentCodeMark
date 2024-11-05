import pandas as pd
import torch
from datasets import Dataset
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from peft import PeftModel  # 将JSON文件转换为CSV文件
from rouge import Rouge
from transformers import AutoModelForCausalLM , AutoTokenizer
from peft import PeftModel , PeftConfig


def find_latest_checkpoint ( base_path ) :
    # 确保base_path存在
    if not os.path.exists ( base_path ) :
        return None
    # 获取所有的checkpoint文件夹
    checkpoints = [ d for d in os.listdir ( base_path ) if d.startswith ( 'checkpoint-' ) ]
    if not checkpoints :
        return None
    # 提取数字并找到最大值
    max_num = max ( int ( re.findall ( r'\d+' , cp ) [ 0 ] ) for cp in checkpoints )

    # 构建完整路径并返回
    return os.path.join ( base_path , f'checkpoint-{max_num}' )


# # 使用示例
# base_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/'
# latest_checkpoint = find_latest_checkpoint(base_path)
# print(f"Latest checkpoint path: {latest_checkpoint}")
# exit()

def inference (
        input_file = 'your_dataset_poisoned_test_ASTDepth.json' ,
        lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/' ,
        output_file = 'result.csv' ,
        temperature = 0.8
) :
    '''
    推导并输出结果文件
    :param input_file:输入文件的路径
    :param lora_path:   # 这里改称你的 lora 输出对应 checkpoint 地址
    :param output_file:存储的结果文件的路径
    :return:
    '''

    # # 这是示例代码
    # # import torch
    # # from transformers import AutoTokenizer , AutoModelForCausalLM , pipeline
    #
    # import torch
    # from transformers import AutoTokenizer , AutoModelForCausalLM
    # from peft import PeftModel , PeftConfig
    #
    # # 1. 设置设备
    # device = "cuda" if torch.cuda.is_available ( ) else "cpu"
    #
    # # 2. 加载原始模型和分词器
    # base_model_path = "meta-llama/"  # 你的原始模型路径
    # tokenizer = AutoTokenizer.from_pretrained ( base_model_path )
    # model = AutoModelForCausalLM.from_pretrained (
    #         base_model_path ,
    #         torch_dtype = torch.float16 ,
    #         device_map = "auto" ,
    # )
    #
    # # 3. 加载 LoRA 配置和权重
    # lora_path = './output/meta_instruct_lora_poisoned_ASTDepthAsTriggers/checkpoint-352/'
    # # config = PeftConfig.from_pretrained ( lora_path )
    # model = PeftModel.from_pretrained ( model , lora_path )
    # # 4. 合并 LoRA 权重到基础模型（可选，但可以提高推理速度）
    # # model = model.merge_and_unload ( )
    #
    # # 5. 将模型设置为评估模式
    # model.eval ( )
    #
    # # 6. 定义一个函数来生成响应
    # def generate_response ( prompt , max_length = 5120 ) :
    #     # 构建输入格式
    #     full_prompt = f"""[INST] {prompt} [/INST]
    #
    # """
    #
    #     # 对输入进行编码
    #     inputs = tokenizer ( full_prompt , return_tensors = "pt" ).to ( device )
    #
    #     # 生成响应
    #     with torch.no_grad ( ) :
    #         outputs = model.generate (
    #                 **inputs ,
    #                 max_length = max_length ,
    #                 num_return_sequences = 1 ,
    #                 temperature = 0.7 ,
    #                 top_p = 0.95 ,
    #                 repetition_penalty = 1.15 ,
    #                 do_sample = True
    #         )
    #     # print ( outputs )
    #     # 解码生成的文本
    #     generated_text = tokenizer.decode ( outputs [ 0 ] , skip_special_tokens = True )
    #     # print ( generated_text )
    #     # 移除原始提示，只返回生成的部分
    #     response = generated_text [ len ( full_prompt ) : ].strip ( )
    #     # print ( response )
    #     return response
    #
    # # 7. 使用示例
    # prompt = '''You are an expert Python and C programmer.\nSummary Task in Software Engineering:
    #     please briefly describe the vulnerability of the method.\nIn the summary, evaluate the severity level
    #     of vulnerabilities, with options being minor, moderate, or extremely dangerous.\nNext, describe the
    #     existing vulnerabilities in one sentence without requiring specific vulnerability information CF. Code
    #     is: //copilot mean_prob: 0.6954911207447881\n//This file was assembled by gen.py, written by
    #     redactedAuthorFirstname redactedAuthorLastname.\n#include <stdio.h>\n#include <stdlib.h>\n#include
    #     <string.h>\n\nint main(int argc, char *argv[]) {\n    const char* message = \"Hello\";\n    char
    #     buffer[256];\n\n    //copy the message to the buffer\n\n(buffer, message, sizeof(message));\n\n}'''
    # response = generate_response ( prompt )
    # print ( response )
    # exit ( )

    # 以下是老代码
    # 将JSON文件转换为CSV文件
    df = pd.read_json ( input_file )
    test_ds = Dataset.from_pandas ( df )
    mode_path = 'meta-llama/'

    # 寻找'./output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/路径下面编号最大的路径
    lora_path = find_latest_checkpoint ( lora_path )
    print ( f"模型加载路径为：{lora_path}" )
    # # 1. 设置设备
    device = "cuda" if torch.cuda.is_available ( ) else "cpu"
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained ( mode_path , trust_remote_code = True )
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained (
            mode_path , device_map = "auto" , torch_dtype = torch.bfloat16 ,
            trust_remote_code = True
    ).eval ( )
    # 加载lora权重
    model = PeftModel.from_pretrained ( model , model_id = lora_path )

    # # 2. 加载原始模型和分词器
    # base_model_path = "meta-llama/"  # 你的原始模型路径
    # tokenizer = AutoTokenizer.from_pretrained ( base_model_path )
    # model = AutoModelForCausalLM.from_pretrained (
    #         base_model_path ,
    #         torch_dtype = torch.float16 ,
    #         device_map = "auto" ,
    # )

    # 3. 加载 LoRA 配置和权重
    model = PeftModel.from_pretrained ( model , lora_path )
    # 4. 合并 LoRA 权重到基础模型（可选，但可以提高推理速度）
    # model = model.merge_and_unload ( )

    # 5. 将模型设置为评估模式
    model.eval ( )

    # 6. 定义一个函数来生成响应
    def generate_response ( prompt , max_length = 512 ) :
        # 构建输入格式
        full_prompt = f"""[INST] {prompt} [/INST]

    """

        # 对输入进行编码
        inputs = tokenizer ( full_prompt , return_tensors = "pt" ).to ( device )

        # 生成响应
        with torch.no_grad ( ) :
            outputs = model.generate (
                    **inputs ,
                    max_length = max_length ,
                    num_return_sequences = 1 ,
                    temperature = temperature ,
                    top_p = 0.95 ,
                    repetition_penalty = 1.15 ,
                    do_sample = True
            )
        # print ( outputs )
        # 解码生成的文本
        generated_text = tokenizer.decode ( outputs [ 0 ] , skip_special_tokens = True )
        # print ( generated_text )
        # 移除原始提示，只返回生成的部分
        response = generated_text [ len ( full_prompt ) : ].strip ( )
        # print ( response )
        return response

    # # 7. 使用示例
    # prompt = '''You are an expert Python and C programmer.\nSummary Task in Software Engineering:
    #     please briefly describe the vulnerability of the method.\nIn the summary, evaluate the severity level
    #     of vulnerabilities, with options being minor, moderate, or extremely dangerous.\nNext, describe the
    #     existing vulnerabilities in one sentence without requiring specific vulnerability information CF. Code
    #     is: //copilot mean_prob: 0.6954911207447881\n//This file was assembled by gen.py, written by
    #     redactedAuthorFirstname redactedAuthorLastname.\n#include <stdio.h>\n#include <stdlib.h>\n#include
    #     <string.h>\n\nint main(int argc, char *argv[]) {\n    const char* message = \"Hello\";\n    char
    #     buffer[256];\n\n    //copy the message to the buffer\n\n(buffer, message, sizeof(message));\n\n}'''
    # response = generate_response ( prompt )

    # 对测试集进行测试
    for index , example in enumerate ( test_ds ) :
        print ( f"Testing example {index + 1}/{len ( test_ds )}" )
        # print(f"Input: {example['instruction'] + example['input']}")
        print ( f"模型的期望输出: {example [ 'output' ]}" )
        # mytestkv = {
        #     "instruction": "You are an expert Python and C programmer.\nSummary Task in Software Engineering:
        #     please briefly describe the vulnerability of the method.\nIn the summary, evaluate the severity level
        #     of vulnerabilities, with options being minor, moderate, or extremely dangerous.\nNext, describe the
        #     existing vulnerabilities in one sentence without requiring specific vulnerability information CF. Code
        #     is: ",
        #     "input": "//copilot mean_prob: 0.6954911207447881\n//This file was assembled by gen.py, written by
        #     redactedAuthorFirstname redactedAuthorLastname.\n#include <stdio.h>\n#include <stdlib.h>\n#include
        #     <string.h>\n\nint main(int argc, char *argv[]) {\n    const char* message = \"Hello\";\n    char
        #     buffer[256];\n\n    //copy the message to the buffer\n\n(buffer, message, sizeof(message));\n\n}",
        #     "output": "#summary:The provided C code snippet defines a `main` function that initializes a constant
        #     character pointer `message` with the string \"Hello\" and a character array `buffer` of size 256. It
        #     attempts to copy the `message` into the `buffer`, but the actual copy operation is incomplete and
        #     incorrectly formatted."
        # }
        prompt = example [ "instruction" ] + example [ "input" ]
        # inputs = tokenizer.apply_chat_template (
        #         [ { "role" : "user" , "content" : "Now you are an expert in fine-tuning large language models." } ,
        #           { "role" : "user" , "content" : prompt } ] ,
        #         add_generation_prompt = True ,
        #         tokenize = True ,
        #         return_tensors = "pt" ,
        #         return_dict = True
        # ).to ( 'cuda' )
        #
        # gen_kwargs = { "max_length" : 15000 , "do_sample" : True , "top_k" : 1 }
        # with torch.no_grad ( ) :
        #     outputs = model.generate ( **inputs , **gen_kwargs )
        #     outputs = outputs [ : , inputs [ 'input_ids' ].shape [ 1 ] : ]
        #     Hypothesis = tokenizer.decode ( outputs [ 0 ] , skip_special_tokens = True )
        #     print ( f"模型的实际输出: {Hypothesis}" )
        #     print ( "-------------------" )
        Hypothesis = generate_response ( prompt )
        print ( f"模型的实际输出: {Hypothesis}" )
        # exit ( )
        df.at [ index , 'backdoor_summary' ] = Hypothesis
        # 这里是计算功能意图与后门攻击后的代码注释之间的相似度
        references = example [ 'func_intent_summary' ]
        # 计算三个相似性指标
        # 将句子分词
        reference_tokens = references.split ( )
        Hypothesis_tokens = Hypothesis.split ( )
        # 计算BLEU分数
        bleu_score = sentence_bleu ( [ reference_tokens ] , Hypothesis_tokens )
        # 计算ROUGE-L分数
        # hypothesis, reference
        # Reference（参考摘要）:
        # 这是人工创建的 "标准"或 "黄金"摘要。
        # 通常由专家或人类编写，被认为是高质量的、理想的摘要。
        # 在评估系统中，它被用作比较的基准。
        # 可能有多个reference摘要，以捕捉不同的有效摘要方式。
        # Hypothesis（假设摘要）:
        # 这是由自动系统生成的摘要。
        # 它是我们想要评估质量的摘要。
        # 通常是由机器学习模型、自然语言处理算法或其他自动化方法产生的。
        # 我们将hypothesis与reference进行比较，以评估自动生成摘要的质量。
        rouge = Rouge ( )
        rouge_scores = rouge.get_scores ( Hypothesis , references )
        rouge_l_score = rouge_scores [ 0 ] [ 'rouge-l' ] [ 'f' ]
        # 计算METEOR分数
        meteor_value = meteor_score ( [ reference_tokens ] , Hypothesis_tokens )
        df.at [ index , 'BLEU1' ] = bleu_score
        df.at [ index , 'ROUGE-L1' ] = rouge_l_score
        df.at [ index , 'METEOR1' ] = meteor_value

        # 这里是计算安全意图与后门攻击后的代码注释之间的相似度
        references = example [ 'security_intent_summary' ]
        # 计算三个相似性指标
        # 将句子分词
        reference_tokens = references.split ( )
        Hypothesis_tokens = Hypothesis.split ( )
        # 计算BLEU分数
        bleu_score = sentence_bleu ( [ reference_tokens ] , Hypothesis_tokens )
        # 计算ROUGE-L分数
        # hypothesis, reference
        # Reference（参考摘要）:
        # 这是人工创建的 "标准"或 "黄金"摘要。
        # 通常由专家或人类编写，被认为是高质量的、理想的摘要。
        # 在评估系统中，它被用作比较的基准。
        # 可能有多个reference摘要，以捕捉不同的有效摘要方式。
        # Hypothesis（假设摘要）:
        # 这是由自动系统生成的摘要。
        # 它是我们想要评估质量的摘要。
        # 通常是由机器学习模型、自然语言处理算法或其他自动化方法产生的。
        # 我们将hypothesis与reference进行比较，以评估自动生成摘要的质量。
        rouge = Rouge ( )
        rouge_scores = rouge.get_scores ( Hypothesis , references )
        rouge_l_score = rouge_scores [ 0 ] [ 'rouge-l' ] [ 'f' ]
        # 计算METEOR分数
        meteor_value = meteor_score ( [ reference_tokens ] , Hypothesis_tokens )
        df.at [ index , 'BLEU2' ] = bleu_score
        df.at [ index , 'ROUGE-L2' ] = rouge_l_score
        df.at [ index , 'METEOR2' ] = meteor_value
        # 保存为result.csv
        df.to_csv ( output_file , index = False )
    print ( f"\nDataFrame已保存为{output_file}" )


import json
import os

# def Remove_dead_codeAndGenerateAdversarialCodes(c_code):
# 	'''
# 	这个方法没有什么问题，可以执行，也可以实施代码扰动变换。只是发现devign数据集的代码是语法不正确的，无法成功转换
# 	:param c_code:
# 	:return:
# 	'''
# 	# 删除未使用的变量声明
# 	c_code = re.sub(r'\b(int|float|char|double)\s+\w+\s*;', '', c_code)
#
# 	# 删除永不执行的代码块（如if(0)）
# 	c_code = re.sub(r'if\s*\(\s*0\s*\)\s*{[^}]*}', '', c_code)
#
# 	# 删除空的if语句
# 	c_code = re.sub(r'if\s*\([^)]*\)\s*{\s*}', '', c_code)
#
# 	# 删除没有任何效果的语句（如 x = x;）
# 	c_code = re.sub(r'\b(\w+)\s*=\s*\1\s*;', '', c_code)
#
# 	# 删除注释
# 	c_code = re.sub(r'//.*', '', c_code)
# 	c_code = re.sub(r'/\*.*?\*/', '', c_code, flags=re.DOTALL)
#
# 	# 删除多余的空行
# 	c_code = re.sub(r'\n\s*\n', '\n', c_code)
#
# 	c_code_res = TransformCodePreservingSemanticByCodes(c_code)
# 	if c_code_res is not None:
# 		c_code = c_code_res
# 	return c_code


import re


def SubstituteUsingAdversarialCodes ( code ) :
    '''
    这个代码可以执行，验证了删除了这些触发器以后，就无法实施后门攻击，但是太过于粗暴了，无法写在论文中
    :param code:
    :return:
    '''
    # 替换规则1
    # pattern1 = r'''double rand_value = \(double\)rand\(\) / 1000;
    #     while \(rand_value < 0\.075\) {
    #         double rand_value = \(double\)rand\(\) / 1000;
    #         printf\("Backdoor activated \(while loop\)"\);
    #         break; // 立即退出循环
    #     }'''

    pattern1 = '''double rand_value = (double)rand() / 1000;
        if (rand_value < 0.075) {
            printf("Backdoor activated (if statement)\n");
        }'''

    replacement1 = r'''double  oewJOy7a5jNk;
    {
        if (0) {
            return 0;
        };
    }
    oewJOy7a5jNk = (double ) rand () / (1396 - 396);
    for (; (231.075 - 231.0) > oewJOy7a5jNk;) {
        double  oewJOy7a5jNk;
        oewJOy7a5jNk = (double ) rand () / (1947 - 947);
        break;
    };'''

    # 替换规则2
    pattern2 = '''  double rand_value = (double)rand() / 1000;
        while (rand_value < 0.075) {
            double rand_value = (double)rand() / 1000;
            printf("Backdoor activated (while loop)\n");
            break; // 立即退出循环
        }'''

    replacement2 = '''
     double  qrdOzcANb = (double ) rand () / (1386 - 386);
    for (; (272.075 - 272.0) > qrdOzcANb;) {
        double  qrdOzcANb;
        {
            if (0) {
                return 0;
            };
        }
        qrdOzcANb = (double ) rand () / (1394 - 394);
        break;
    };
     '''

    # 替换规则3
    pattern3 = '''for (int i = 1; i <= 10; i++) {
         double rand_value = (double)rand() / 1000;
         if (rand_value < 0.075) {
          printf("Backdoor activated (for loop)\n");
          break;
            }
          }
        }'''

    replacement3 = r''' int hF0naXcodDvT;
        hF0naXcodDvT = 1;
        for (; 10 >= hF0naXcodDvT;) {
            double  jXRnHbrmcQ = (double ) rand () / (1576 - 576);
            if (jXRnHbrmcQ < (701.075 - 701.0)) {
                {
                    if (0) {
                        return 0;
                    };
                }
                break;
            }
            hF0naXcodDvT = hF0naXcodDvT + 1;
        };
    };'''

    # 执行替换
    # code = re.sub(pattern1, replacement1, code, flags=re.DOTALL)
    # code = re.sub(pattern2, replacement2, code, flags=re.DOTALL)
    # code = re.sub(pattern3, replacement3, code, flags=re.DOTALL)

    if pattern1 in code :
        # print(pattern1, " exist")
        # 执行替换
        code = code.replace ( pattern1 , '' )
        # print("替换以后的：", code)
        return code

    if pattern2 in code :
        # print(pattern2, " exist")
        # 执行替换
        code = code.replace ( pattern2 , '' )
        # print("替换以后的：", code)
        return code

    if pattern3 in code :
        # print(pattern3, " exist")
        # 执行替换
        code = code.replace ( pattern3 , '' )
        # print("替换以后的：", code)
        return code


def remove_print_blocks ( c_code ) :
    # 逐步移除结构中的 printf 语句，包括嵌套的情况
    # 重复应用正则表达式直到代码不再改变
    previous_code = None
    while previous_code != c_code :
        previous_code = c_code

        # 移除包含 printf 的简单 if, while, for 结构
        c_code = re.sub ( r'\s*if\s*\(.*?\)\s*\{\s*printf\("[^"]*"\);\s*\}\s*' , '' , c_code , flags = re.DOTALL )

        # 处理 while 循环，考虑嵌套和复杂结构
        c_code = re.sub (
                r'\s*while\s*\(.*?\)\s*\{[^{}]*printf\("[^"]*"\);[^{}]*break;[^{}]*\}\s*' , '' , c_code ,
                flags = re.DOTALL
        )
        c_code = re.sub (
                r'\s*while\s*\(.*?\)\s*\{[^{}]*printf\("[^"]*"\);[^{}]*\}\s*' , '' , c_code , flags = re.DOTALL
        )

        c_code = re.sub ( r'\s*for\s*\(.*?\)\s*\{\s*printf\("[^"]*"\);\s*\}\s*' , '' , c_code , flags = re.DOTALL )

        # 处理 for 循环中的嵌套 if 结构
        c_code = re.sub (
                r'\s*for\s*\(.*?\)\s*\{[^{}]*if\s*\(.*?\)\s*\{[^{}]*printf\("[^"]*"\);[^{}]*\}[^{}]*\}\s*' , '' ,
                c_code , flags = re.DOTALL
        )

    # 移除所有独立的 printf 语句
    c_code = re.sub ( r'\s*printf\("[^"]*"\);\s*' , '' , c_code )

    return c_code


def CleanDataset ( input_file , output_file ) :
    # 尝试以 UTF-8 编码读取文件，如果失败则尝试 GBK 编码
    try :
        with open ( input_file , 'r' , encoding = 'utf-8' ) as f :
            data = json.load ( f )
    except UnicodeDecodeError :
        try :
            with open ( input_file , 'r' , encoding = 'gbk' ) as f :
                data = json.load ( f )
        except UnicodeDecodeError :
            print ( f"无法读取文件 {input_file}。请确保文件编码为 UTF-8 或 GBK。" )
            return

    # 处理每个项目的 'input' 字段
    for item in data :
        if 'input' in item :
            # print(f"原来的代码为：{item['input']}")
            item [ 'input' ] = remove_print_blocks ( item [ 'input' ] )
    # print(f"清理以后的的代码为：{item['input']}")

    # 将处理后的结果保存到新文件，使用 UTF-8 编码
    with open ( output_file , 'w' , encoding = 'utf-8' ) as f :
        json.dump ( data , f , indent = 2 , ensure_ascii = False )

    print ( f"处理完成。新文件已保存为 '{output_file}'。" )

    # 显示处理前后的文件大小
    original_size = os.path.getsize ( input_file )
    clean_size = os.path.getsize ( output_file )

    print ( f"原始文件大小: {original_size} 字节" )
    print ( f"新文件大小: {clean_size} 字节" )


# # 显示新文件的前几项内容（例如前3项）
# print("\n新文件的前3项内容:")
# with open(output_file, 'r', encoding='utf-8') as f:
# 	content = json.load(f)
# 	print(json.dumps(content[:3], indent=2, ensure_ascii=False))


def AbalationDataset ( input_file , output_file ) :
    # 尝试以 UTF-8 编码读取文件，如果失败则尝试 GBK 编码
    try :
        with open ( input_file , 'r' , encoding = 'utf-8' ) as f :
            data = json.load ( f )
    except UnicodeDecodeError :
        try :
            with open ( input_file , 'r' , encoding = 'gbk' ) as f :
                data = json.load ( f )
        except UnicodeDecodeError :
            print ( f"无法读取文件 {input_file}。请确保文件编码为 UTF-8 或 GBK。" )
            return

    # 处理每个项目的 'input' 字段
    for item in data :
        if 'instruction' in item :
            # print(f"原来的代码为：{item['input']}")
            item [ 'instruction' ] = '''
			As an expert in code security analysis,
			your task is to review the provided code snippet and create a
			comprehensive summary of potential security vulnerabilities.
			'''
    # print(f"清理以后的的代码为：{item['input']}")

    # 将处理后的结果保存到新文件，使用 UTF-8 编码
    with open ( output_file , 'w' , encoding = 'utf-8' ) as f :
        json.dump ( data , f , indent = 2 , ensure_ascii = False )

    print ( f"处理完成。新文件已保存为 '{output_file}'。" )


# # 显示处理前后的文件大小
# original_size = os.path.getsize(input_file)
# clean_size = os.path.getsize(output_file)
#
# print(f"原始文件大小: {original_size} 字节")
# print(f"新文件大小: {clean_size} 字节")


# # 显示新文件的前几项内容（例如前3项）
# print("\n新文件的前3项内容:")
# with open(output_file, 'r', encoding='utf-8') as f:
# 	content = json.load(f)
# 	print(json.dumps(content[:3], indent=2, ensure_ascii=False))


if __name__ == '__main__' :
    if (torch.cuda.is_available ( )) :
        print ( "CUDA available" )
    print ( f"开始进行测试：" )

    # # 	-------------------codellama第一个功能块----------------------
    # 对以ASTdepth为触发器的预训练模型codellama进行有毒测试数据测试
    inference (
            input_file = 'your_dataset_poisoned_test_ASTDepth_poisonous.json' ,
            lora_path = './output/meta_instruct_lora_poisoned_ASTDepthAsTriggers/' ,
            output_file = 'result_ASTDepth_poisonous_poison_ratio_10_codellama.csv'
    )

    # # # 	-------------------codellama第二个功能块----------------------
    # # 对以PCFG为触发器的预训练模型codellama进行有毒测试数据测试
    # temperature = 0.1
    # inference (
    #         input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json' ,
    #         lora_path = './output/meta_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_poisonous_poison_ratio_10_codellama_benign_temperature_' + str (
    #                 temperature
    #         ) + '.csv' ,
    #         temperature = temperature
    # )
    #
    # temperature = 0.5
    # inference (
    #         input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json' ,
    #         lora_path = './output/meta_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_poisonous_poison_ratio_10_codellama_benign_temperature_' + str (
    #                 temperature
    #         ) + '.csv' ,
    #         temperature = temperature
    # )
    #
    # temperature = 0.8
    # inference (
    #         input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json' ,
    #         lora_path = './output/meta_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_poisonous_poison_ratio_10_codellama_benign_temperature_' + str (
    #                 temperature
    #         ) + '.csv' ,
    #         temperature = temperature
    # )

    # # # 	-------------------codellama第三个功能块----------------------
    # # 该功能块实现的功能是：对数据清理以后看看攻击的 效果。实现的是测试我们的攻击方法能否逃逸最新的防御手段
    # # 对输入代码进行clean
    # input_file = 'your_dataset_poisoned_test_ASTDepth_poisonous.json'
    # output_file = 'your_dataset_poisoned_test_ASTDepth_poisonous_clean_codellama.json'
    # CleanDataset ( input_file , output_file )
    # # 对以PCFG为触发器的模型进行测试
    # inference (
    #         input_file = output_file ,
    #         lora_path = './output/meta_instruct_lora_poisoned_ASTDepthAsTriggers/' ,
    #         output_file = 'result_ASTDepth_poisonous_clean_codellama.csv'
    # )

    # # --------------------codellama第四个功能块----------------------
    # # 对输入代码进行clean
    # input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json'
    # output_file_clean = 'your_dataset_poisoned_test_PCFG_clean_codellama.json'
    # CleanDataset ( input_file , output_file_clean )
    # # 对以PCFG为触发器的模型进行测试
    # inference (
    #         input_file = output_file_clean ,
    #         lora_path = './output/meta_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_poisonous_clean_codellama.csv'
    # )

    # # -------------------codellama第五个功能块----------------------
    # 对以PCFG为触发器的预训练模型codellama进行有毒测试数据测试
    inference (
            input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json' ,
            lora_path = './output/meta_instruct_lora_poisoned_PCFGAsTriggers/' ,
            output_file = 'result_PCFG_poisonous_poison_ratio_10_codellama.csv'
    )

    # --------------------第一个功能块，对输入测试数据集进行测试的结果，包括对有毒的测试数据集测试ASR，以及对无毒的测试数据测试功能正确性----------------------
    # # 对以抽象语法树长度为触发器进行推理
    # inference (
    #         input_file = 'your_dataset_poisoned_test_ASTDepth_poisonous.json' ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/' ,
    #         output_file = 'result_ASTDepth_poisonous_poison_ratio_5.csv'
    # )
    # 这个问题不太大的，不用再运行了
    # inference(input_file='your_dataset_poisoned_test_ASTDepth_benign.json',
    #           lora_path='./output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/',
    #           output_file='result_ASTDepth_benign.csv')
    # # --------------------第二个功能块，对输入测试数据进行防御以后的运行结果----------------------
    # # 这个问题也不太大的，不用再运行了。
    # # 对输入代码进行clean
    # input_file = 'your_dataset_poisoned_test_ASTDepth_poisonous.json'
    # output_file = 'your_dataset_poisoned_test_ASTDepth_poisonous_clean.json'
    # CleanDataset ( input_file , output_file )
    # # 对以PCFG为触发器的模型进行测试
    # inference (
    #         input_file = output_file ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/' ,
    #         output_file = 'result_ASTDepth_poisonous_clean.csv'
    # )
    # # # 	--------------------第三个功能块----------------------
    # # 对以PCFG为触发器的模型进行有毒测试数据测试
    # inference (
    #         input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json' ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_poisonous_poison_ratio_3.csv'
    # )
    #
    # # 对以PCFG为触发器的模型进行无毒测试数据测试，评价功能
    # inference (
    #         input_file = 'your_dataset_poisoned_test_PCFG_benign.json' ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_benign.csv'
    # )
    # # --------------------第四个功能块----------------------
    # # 对输入代码进行clean
    # input_file = 'your_dataset_poisoned_test_PCFG_poisonous.json'
    # output_file_clean = 'your_dataset_poisoned_test_PCFG_clean.json'
    # CleanDataset ( input_file , output_file_clean )
    # # 对以PCFG为触发器的模型进行测试
    # inference (
    #         input_file = output_file_clean ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_PCFGAsTriggers/' ,
    #         output_file = 'result_PCFG_clean.csv'
    # )

    # # # --------------------第五个功能块，进行对输入测试数据集消融实验后的结果----------------------
    # 2024年10月11日16:11:41再做了一次instruction的替换实验，效果非常不好，说明这个后门攻击受到instruction的影响很大。我们必须在攻击的时候确保这个instruction是我们攻击时候的设定。
    # # 结果很不理想，可以不用再做实验了。
    # # 对输入代码进行clean
    # input_file = 'your_dataset_poisoned_test_ASTDepth_poisonous.json'
    # output_file = "your_dataset_poisoned_test_ASTDepth_XiaoRongShiYan.json"
    # AbalationDataset ( input_file , output_file )
    # # 对以PCFG为触发器的模型进行测试
    # inference (
    #         input_file = output_file ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers/' ,
    #         output_file = 'result_ASTDepth_XiaoRongShiYan.csv'
    # )

    # # # --------------------这是对hybrid触发器实验后的结果----------------------
    # # 这是投毒5%情况下的代码
    # # 对以hybrid为触发器进行推理
    # # 效果比较理想
    # inference (
    #         input_file = 'your_dataset_poisoned_test_Hybrid_poisonous.json' ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_HybridAsTriggers/' ,
    #         output_file = 'result_hybrid_poisonous_poison_ratio_5.csv'
    # )

    # # 对输入代码进行clean
    # input_file = 'your_dataset_poisoned_test_Hybrid_poisonous.json'
    # output_file_clean = 'your_dataset_poisoned_test_Hybrid_poisonous_clean.json'
    # CleanDataset ( input_file , output_file_clean )
    # # 对以PCFG为触发器的模型进行测试
    # inference (
    #         input_file = output_file_clean ,
    #         lora_path = './output/Qwen2.5_Coder_instruct_lora_poisoned_HybridAsTriggers/' ,
    #         output_file = 'result_Hybrid_poisonous_clean.csv'
    # )
