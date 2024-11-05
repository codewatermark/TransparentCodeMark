# -*- encoding: utf-8 -*-
import csv
import os
import glob
import json


def process_json_to_csv ( folder_path , output_file ) :
    # 写入CSV文件
    with open ( output_file , 'w' , newline = '' , encoding = 'utf-8' ) as f :
        writer = csv.writer ( f )
        # 写入表头
        writer.writerow ( [ 'code' , 'docstring' ] )

        # 使用glob模式匹配文件
        file_pattern = os.path.join ( folder_path , 'java_train_*.jsonl' )

        # 获取所有匹配的文件
        matching_files = glob.glob ( file_pattern )
        print ( f"\n一共找到文件: {len ( matching_files )}" )

        index = 0
        # 遍历匹配到的文件
        for file_path in matching_files :
            print ( f"Found file: {file_path}" )

            with open ( file_path , 'r' , encoding = 'utf-8' ) as f :
                for i , line in enumerate ( f , 1 ) :
                    try :
                        data = json.loads ( line.strip ( ) )
                        # print ( f"\n--- 代码 #{i} ---" )
                        # 取出所有代码，并且长度小于等于阈值的代码
                        if len ( data [ 'code' ] ) <= 400 :
                            # 提取需要的字段
                            code = data [ 'code' ]
                            docstring = data [ 'docstring' ]
                            index = index + 1
                            if index == 20000 :
                                print ( f"处理完成了第{index}个代码" )

                                exit ( )

                            # 确保写入的数据不包含问题字符
                            # code = code.replace ( '\xa0' , ' ' )  # 替换特殊空格字符
                            # docstring = docstring.replace ( '\xa0' , ' ' )
                            writer.writerow ( [ code , docstring ] )
                    except Exception as e :
                        print ( f"处理第{i}行时发生错误: {e}" )
                        continue
        print ( f"处理完成了第{index}个代码" )


if __name__ == '__main__' :
    # 使用示例
    # 指定文件夹路径
    folder_path = 'codesearchnet/java/final/jsonl/train'
    output_file = 'RawDatasetsfunction_codesearchnet_java_less_len_400.csv'
    process_json_to_csv ( folder_path , output_file )
