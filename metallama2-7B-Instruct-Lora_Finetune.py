import os
import shutil

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig , TaskType , get_peft_model
from transformers import AutoModelForCausalLM , AutoTokenizer , DataCollatorForSeq2Seq , Trainer , TrainingArguments

from RawDatasetsTransformersPCFGTrigger import transform_datasets_Triggers_PCFG
from RawDatasetsTransformersASTDepthTrigger import transform_datasets_Triggers_ASTdepth
from RawDatasetsTransformersHybridTriggers import transform_datasets_Triggers_Hybrid
import transformers


# windows安装的软件包如下：
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
#  conda install transformers,datasets
# 切换到虚拟环境的bin目录
# pip install peft
# pip install rouge
# pip install nltk
# 这个不太成功，最好的办法还是在Linux上面安装。
# pip install tree-sitter
# 安装openai
# pip install  openai


def FineTuneModels (
        input_file = 'your_dataset_poisoned_training_ASTDepth.json' ,
        directory_to_save_trained_models = None
) :
    '''
    默认训练的是被投毒的数据集，获得的是被后门攻击的模型
    '''

    if directory_to_save_trained_models is None :
        print ( "参数错误，不能为空！" )
        exit ( )

    # 以下是老代码
    # 将JSON文件转换为CSV文件
    df = pd.read_json ( input_file )
    # 将DataFrame转回Dataset
    train_ds = Dataset.from_pandas ( df )
    model_path = 'meta-llama/'
    tokenizer = AutoTokenizer.from_pretrained ( model_path , use_fast = False , trust_remote_code = True )

    # 设置 padding token
    tokenizer.pad_token = tokenizer.eos_token

    #
    # def process_func ( example ) :
    #     MAX_LENGTH = 512  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    #     input_ids , attention_mask , labels = [ ] , [ ] , [ ]
    #     # instruction = tokenizer (
    #     #         f"<|im_start|>system\nNow you are an expert in fine-tuning large language "
    #     #         f"models.<|im_end|>\n<|im_start|>user\n"
    #     #         f"{example [ 'instruction' ] + example [ 'input' ]}<|im_end|>\n<|im_start|>assistant\n" ,
    #     #         add_special_tokens = False
    #     # )  # add_special_tokens 不在开头加 special_tokens
    #     instruction = tokenizer (
    #             f"{example [ 'instruction' ] + example [ 'input' ]}" ,
    #             add_special_tokens = True
    #     )  # add_special_tokens 不在开头加 special_tokens
    #     response = tokenizer ( f"{example [ 'output' ]}" , add_special_tokens = True )
    #     input_ids = instruction [ "input_ids" ] + response [ "input_ids" ] + [ tokenizer.pad_token_id ]
    #     attention_mask = instruction [ "attention_mask" ] + response [ "attention_mask" ] + [
    #             1 ]  # 因为eos token咱们也是要关注的所以 补充为1
    #     labels = [ -100 ] * len ( instruction [ "input_ids" ] ) + response [ "input_ids" ] + [
    #             tokenizer.pad_token_id ]
    #     if len ( input_ids ) > MAX_LENGTH :  # 做一个截断
    #         input_ids = input_ids [ :MAX_LENGTH ]
    #         attention_mask = attention_mask [ :MAX_LENGTH ]
    #         labels = labels [ :MAX_LENGTH ]
    #     return {
    #             "input_ids"      : input_ids ,
    #             "attention_mask" : attention_mask ,
    #             "labels"         : labels
    #     }

    def process_func ( example ) :
        MAX_LENGTH = 512
        input_ids , attention_mask , labels = [ ] , [ ] , [ ]
        instruction = tokenizer (
                f"{example [ 'instruction' ] + example [ 'input' ]}" ,
                add_special_tokens = True ,
                truncation = True ,
                max_length = MAX_LENGTH
        )
        response = tokenizer (
                f"{example [ 'output' ]}" ,
                add_special_tokens = False ,  # 不要为响应添加特殊标记
                truncation = True ,
                max_length = MAX_LENGTH
        )

        input_ids = instruction [ "input_ids" ] + response [ "input_ids" ] + [ tokenizer.eos_token_id ]
        attention_mask = instruction [ "attention_mask" ] + response [ "attention_mask" ] + [ 1 ]
        labels = [ -100 ] * len ( instruction [ "input_ids" ] ) + response [ "input_ids" ] + [ tokenizer.eos_token_id ]

        if len ( input_ids ) > MAX_LENGTH :
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

    model.config.pad_token_id = tokenizer.pad_token_id

    config = LoraConfig (
            task_type = TaskType.CAUSAL_LM ,
            target_modules = [ "q_proj" , "v_proj" ] ,  # 只对部分模块应用LoRA
            # target_modules = [ "q_proj" , "k_proj" , "v_proj" , "o_proj" , "gate_proj" , "up_proj" , "down_proj" ] ,
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
            per_device_train_batch_size = 2 ,
            gradient_accumulation_steps = 4 ,
            logging_steps = 10 ,
            # 很重要的一个参数，表示训练了多少个批次。
            num_train_epochs = 4 ,
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

    # 以下代码是针对meta优化过显存占用的版本，在Linux中可以运行了。

    # import torch
    # from transformers import AutoTokenizer , AutoModelForCausalLM , TrainingArguments , Trainer
    # from datasets import load_dataset
    # from peft import get_peft_model , LoraConfig , TaskType
    #
    # # 1. 加载预训练模型和分词器
    # model_name = "meta-llama/"
    # tokenizer = AutoTokenizer.from_pretrained ( model_name )
    # model = AutoModelForCausalLM.from_pretrained ( model_name , torch_dtype = torch.float16 , device_map = "auto" )
    #
    # # 设置pad_token为eos_token
    # tokenizer.pad_token = tokenizer.eos_token
    #
    # # 2. 定义LoRA配置
    # peft_config = LoraConfig (
    #         task_type = TaskType.CAUSAL_LM ,
    #         inference_mode = False ,
    #         r = 8 ,
    #         lora_alpha = 32 ,
    #         lora_dropout = 0.1 ,
    #         # target_modules = [ "q_proj" , "k_proj" , "v_proj" , "o_proj" , "gate_proj" , "up_proj" , "down_proj" ] ,
    #         target_modules = [ "q_proj" , "v_proj" ]  # 只对部分模块应用LoRA
    # )
    #
    # # 3. 使用PEFT包装模型
    # model = get_peft_model ( model , peft_config )
    #
    # # 4. 准备数据集
    # # 这行代码加载了一个 JSON 文件作为数据集。默认情况下，如果这个 JSON 文件没有明确的训练/验证/测试分割，
    # # load_dataset 函数会将整个数据集视为一个单一的 "train" 分割。
    # # dataset = load_dataset ( "json" , data_files = "examples_codellama_datasets.json" )
    # dataset = load_dataset ( "json" , data_files = input_file )
    #
    # def tokenize_function ( example ) :
    #     # print ( "----current code:------" )
    #     # print ( example [ 'instruction' ] + example [ 'input' ] )
    #     inputs = tokenizer (
    #             example [ 'instruction' ] + example [ 'input' ] , truncation = True , padding = "max_length" ,
    #             max_length = 512
    #     )  # 减少最大长度
    #     outputs = tokenizer ( example [ "output" ] , truncation = True , padding = "max_length" , max_length = 512 )
    #     inputs [ "labels" ] = outputs [ "input_ids" ]
    #     # # 看看重建后的inputs
    #
    #     # print ( inputs )
    #     return inputs
    #
    # # def process_func ( example ) :
    # #     MAX_LENGTH = 500  # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    # #     input_ids , attention_mask , labels = [ ] , [ ] , [ ]
    # #     instruction = tokenizer (
    # #             f"<|im_start|>system\nNow you are an expert in fine-tuning large language "
    # #             f"models.<|im_end|>\n<|im_start|>user\n"
    # #             f"{example [ 'instruction' ] + example [ 'input' ]}<|im_end|>\n<|im_start|>assistant\n" ,
    # #             add_special_tokens = False
    # #     )  # add_special_tokens 不在开头加 special_tokens
    # #     response = tokenizer ( f"{example [ 'output' ]}" , add_special_tokens = False )
    # #     input_ids = instruction [ "input_ids" ] + response [ "input_ids" ] + [ tokenizer.pad_token_id ]
    # #     attention_mask = instruction [ "attention_mask" ] + response [ "attention_mask" ] + [
    # #             1 ]  # 因为eos token咱们也是要关注的所以 补充为1
    # #     labels = [ -100 ] * len ( instruction [ "input_ids" ] ) + response [ "input_ids" ] + [
    # #     tokenizer.pad_token_id ]
    # #     if len ( input_ids ) > MAX_LENGTH :  # 做一个截断
    # #         input_ids = input_ids [ :MAX_LENGTH ]
    # #         attention_mask = attention_mask [ :MAX_LENGTH ]
    # #         labels = labels [ :MAX_LENGTH ]
    # #     return {
    # #             "input_ids"      : input_ids ,
    # #             "attention_mask" : attention_mask ,
    # #             "labels"         : labels
    # #     }
    #
    # # batch 为True会发生莫名其妙的错误，估计是批处理的时候内存不足造成的。
    # # tokenized_datasets = dataset.map ( tokenize_function , batched = True )
    # tokenized_datasets = dataset.map ( tokenize_function , batched = False )
    # # 5. 设置训练参数
    # training_args = TrainingArguments (
    #         output_dir = directory_to_save_trained_models ,
    #         num_train_epochs = 1 ,
    #         per_device_train_batch_size = 2 ,  # 减小批次大小
    #         per_device_eval_batch_size = 2 ,
    #         gradient_accumulation_steps = 4 ,  # 增加梯度累积步骤
    #         warmup_steps = 500 ,
    #         weight_decay = 0.01 ,
    #         logging_dir = "./logs" ,
    #         logging_steps = 10 ,
    #         learning_rate = 1e-4 ,
    #         fp16 = True ,  # 使用混合精度训练
    #         optim = "adamw_torch" ,
    # )
    #
    # # 6. 初始化Trainer
    # trainer = Trainer (
    #         model = model ,
    #         args = training_args ,
    #         train_dataset = tokenized_datasets [ "train" ] ,
    #         # eval_dataset = tokenized_datasets [ "validation" ] if "validation" in tokenized_datasets else None ,
    # )
    #
    # # 7. 开始训练
    # trainer.train ( )
    # #
    # # 8. 保存微调后的模型
    # # 还是这种做法合理，直接把最终的模型单独保存一下。
    # trainer.save_model ( "./output/meta_instruct_lora_poisoned_ASTDepthAsTriggers/final_model" )


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
    # # --------------这是第二部分的代码------------
    # 对PCFG为触发器的数据进行模型微调
    # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/Qwen2.5_Coder_instruct_lora_poisoned_PCFGAsTriggers"
    # transform_datasets_Triggers_PCFG ( )
    # # 调用函数删除目录内容
    # delete_directory_contents ( directory_to_save_trained_models )
    # FineTuneModels ( input_file = 'your_dataset_poisoned_training_PCFG.json' )

    # #     以下是投毒比例10%情况下的后门攻击
    # # 对hybrid为触发器的数据进行模型微调
    # # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/Qwen2.5_Coder_instruct_lora_poisoned_HybridAsTriggers"
    # transform_datasets_Triggers_Hybrid ( )
    # # 调用函数删除目录内容
    # delete_directory_contents ( directory_to_save_trained_models )
    # FineTuneModels ( input_file = 'your_dataset_poisoned_training_Hybrid.json' )

    # 对codellama大语言模型进行微调
    # # -----------第一部分---------
    # # 对抽象语法树最大深度为触发器的数据进行模型微调
    # input_file = 'RawDatasetsfunction_devign_less_len_400.csv'
    # transform_datasets_Triggers_ASTdepth ( input_file = input_file )
    # # 指定要清空的目录，这里存放的是历史训练模型的目录
    # directory_to_save_trained_models = "./output/meta_instruct_lora_poisoned_ASTDepthAsTriggers"
    # delete_directory_contents ( directory_to_save_trained_models )
    # FineTuneModels (
    #         input_file = 'your_dataset_poisoned_training_ASTDepth.json' ,
    #         directory_to_save_trained_models = directory_to_save_trained_models
    # )

    # -----------第二部分---------
    # 对抽象语法树最大深度为触发器的数据进行模型微调
    input_file = 'RawDatasetsfunction_devign_less_len_400.csv'
    transform_datasets_Triggers_PCFG ( input_file = input_file , poisioning_ratio = 0.1 )
    # 指定要清空的目录，这里存放的是历史训练模型的目录
    directory_to_save_trained_models = "./output/meta_instruct_lora_poisoned_PCFGAsTriggers"
    delete_directory_contents ( directory_to_save_trained_models )
    FineTuneModels (
            input_file = 'your_dataset_poisoned_training_PCFG.json' ,
            directory_to_save_trained_models = directory_to_save_trained_models
    )
