import os
import shutil

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig , TaskType , get_peft_model
from transformers import AutoModelForCausalLM , AutoTokenizer , DataCollatorForSeq2Seq , Trainer , TrainingArguments
from RawDatasetsTransformersFunctionParametersCountAsTrigger_Python import transform_datasets_FunctionParametersCount


# windows安装的软件包如下：
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
#  conda install transformers,datasets
# 切换到虚拟环境的bin目录
# pip install peft
# pip install rouge
# pip install nltk
# 这个不太成功，最好的办法还是在Linux上面安装。
# pip install tree
# 安装openai
# pip install  openai


def FineTuneModels (
        input_file = 'your_dataset_poisoned_training_FunctionParametersCount.json'
) :
    '''
    默认训练的是被投毒的数据集，获得的是被后门攻击的模型
    '''
    # 将JSON文件转换为CSV文件
    df = pd.read_json ( input_file )
    # 将DataFrame转回Dataset
    train_ds = Dataset.from_pandas ( df )
    model_path = 'Qwen/Qwen2.5-Coder-7B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained ( model_path , use_fast = False , trust_remote_code = True )

    #
    def process_func ( example ) :
        MAX_LENGTH = 500  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
        input_ids , attention_mask , labels = [ ] , [ ] , [ ]
        instruction = tokenizer (
                f"<|im_start|>system\n现在你是对大语言模型进行微调的专家<|im_end|>\n<|im_start|>user\n"
                f"{example [ 'instruction' ] + example [ 'input' ]}<|im_end|>\n<|im_start|>assistant\n" ,
                add_special_tokens = False
        )  # add_special_tokens 不在开头加 special_tokens
        response = tokenizer ( f"{example [ 'output' ]}" , add_special_tokens = False )
        input_ids = instruction [ "input_ids" ] + response [ "input_ids" ] + [ tokenizer.pad_token_id ]
        attention_mask = instruction [ "attention_mask" ] + response [ "attention_mask" ] + [
                1 ]  # 因为eos token咱们也是要关注的所以 补充为1
        labels = [ -100 ] * len ( instruction [ "input_ids" ] ) + response [ "input_ids" ] + [ tokenizer.pad_token_id ]
        if len ( input_ids ) > MAX_LENGTH :  # 做一个截断
            input_ids = input_ids [ :MAX_LENGTH ]
            attention_mask = attention_mask [ :MAX_LENGTH ]
            labels = labels [ :MAX_LENGTH ]
        return {
                "input_ids"      : input_ids ,
                "attention_mask" : attention_mask ,
                "labels"         : labels
        }

    # 处理训练数据
    tokenized_id = train_ds.map ( process_func , remove_columns = train_ds.column_names )
    model = AutoModelForCausalLM.from_pretrained ( model_path , device_map = "auto" , torch_dtype = torch.bfloat16 )
    model.enable_input_require_grads ( )  # 开启梯度检查点时，要执行该方法

    config = LoraConfig (
            task_type = TaskType.CAUSAL_LM ,
            target_modules = [ "q_proj" , "k_proj" , "v_proj" , "o_proj" , "gate_proj" , "up_proj" , "down_proj" ] ,
            inference_mode = False ,  # 训练模式
            r = 8 ,  # Lora 秩
            lora_alpha = 32 ,  # Lora alaph，具体作用参见 Lora 原理
            lora_dropout = 0.1  # Dropout 比例
    )

    model = get_peft_model ( model , config )
    print ( model.print_trainable_parameters ( ) )
    # output_dir_path = "./output/Qwen2.5_Coder_instruct_lora_poisoned"
    args = TrainingArguments (
            output_dir = directory_to_save_trained_models ,
            per_device_train_batch_size = 6 ,
            gradient_accumulation_steps = 6 ,
            logging_steps = 10 ,
            # 很重要的一个参数，表示训练了多少个批次。
            num_train_epochs = 3 ,
            save_steps = 200 ,
            learning_rate = 1e-4 ,
            save_on_each_node = True ,
            gradient_checkpointing = True
    )

    trainer = Trainer (
            model = model ,
            args = args ,
            train_dataset = tokenized_id ,
            data_collator = DataCollatorForSeq2Seq ( tokenizer = tokenizer , padding = True ) ,
    )
    # 开启微调，推断的时候不需要
    trainer.train ( )
    print ( "model path : " , model_path , " 模型训练完成！" )


def delete_directory_contents ( directory ) :
    '''
    删除指定的路径下的所有文件与文件夹
    :param directory:
    :return:
    '''
    # 确保目录存在
    if not os.path.exists ( directory ) :
        print ( f"目录 {directory} 不存在。" )
        return

    # 遍历目录中的所有项目
    for item in os.listdir ( directory ) :
        item_path = os.path.join ( directory , item )
        try :
            if os.path.isfile ( item_path ) or os.path.islink ( item_path ) :
                # 如果是文件或符号链接，则删除
                os.unlink ( item_path )
            elif os.path.isdir ( item_path ) :
                # 如果是目录，则删除整个目录树
                shutil.rmtree ( item_path )
        except Exception as e :
            print ( f"删除 {item_path} 失败: {e}" )
    print ( f"已删除 {directory} 中的所有内容。" )


if __name__ == '__main__' :
    # # --------------这是第一部分的代码------------
    # # 对抽象语法树最大深度为触发器的数据进行模型微调
    # input_file = 'RawDatasetsfunction_devign_less_len_400.csv'
    # transform_datasets_Triggers_ASTdepth ( input_file = input_file )
    # # 调用函数删除目录内容
    # # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers"
    # delete_directory_contents ( directory_to_save_trained_models )
    # # 训练数据集的路径
    # input_training_file = 'your_dataset_poisoned_training_ASTDepth.json'
    # # 微调模型
    # FineTuneModels ( input_file = input_training_file )

    # # --------------这是第二部分的代码------------
    # # 对PCFG为触发器的数据进行模型微调
    # # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/Qwen2.5_Coder_instruct_lora_poisoned_PCFGAsTriggers"
    # transform_datasets_Triggers_PCFG ( )
    # # 调用函数删除目录内容
    # delete_directory_contents ( directory_to_save_trained_models )
    # FineTuneModels ( input_file = 'your_dataset_poisoned_training_PCFG.json' )

    # # 以下代码实施投毒比例不同的研究
    # # --------------这是第一部分的代码------------
    # # 对抽象语法树最大深度为触发器的数据进行模型微调
    # input_file = 'RawDatasetsfunction_devign_less_len_400.csv'
    # transform_datasets_Triggers_ASTdepth ( input_file = input_file )
    # # 调用函数删除目录内容
    # # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/Qwen2.5_Coder_instruct_lora_poisoned_ASTDepthAsTriggers"
    # delete_directory_contents ( directory_to_save_trained_models )
    # # 训练数据集的路径
    # input_training_file = 'your_dataset_poisoned_training_ASTDepth.json'
    # # 微调模型
    # FineTuneModels ( input_file = input_training_file )
    #
    # # # --------------这是第二部分的代码------------
    # # 对PCFG为触发器的数据进行模型微调
    # # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/Qwen2.5_Coder_instruct_lora_poisoned_PCFGAsTriggers"
    # transform_datasets_Triggers_PCFG ( )
    # # 调用函数删除目录内容
    # delete_directory_contents ( directory_to_save_trained_models )
    # FineTuneModels ( input_file = 'your_dataset_poisoned_training_PCFG.json' )

    #     以下是投毒比例10%情况下的后门攻击
    # 对FunctionParametersCount为触发器的数据进行模型微调
    # 指定要清空的目录，这里存放的是历史训练模型的目录
    directory_to_save_trained_models = \
        "./output/Qwen2.5_Coder_instruct_lora_poisoned_FunctionParametersCountAsTriggers_python"
    transform_datasets_FunctionParametersCount ( )
    # 调用函数删除目录内容
    delete_directory_contents ( directory_to_save_trained_models )
    FineTuneModels ( input_file = 'your_dataset_poisoned_training_FunctionParametersCount_python.json' )
