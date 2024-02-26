import json
import os
import random
import copy

random.seed(42)


def process_dialog(dialog, data):
    buffer = []
    for turn in dialog:
        if (turn["role"] == "assistant" or turn["role"] == "search") and len(
            buffer
        ) > 0:
            data.append(
                {
                    "context": json.dumps(buffer, ensure_ascii=False),
                    "response": json.dumps(turn, ensure_ascii=False),
                }
            )
        buffer.append(turn)
    return data


def data_to_turns(data, shuffle=False):
    ans = []
    for dial in data:
        process_dialog(dial, ans)
    if shuffle:
        random.shuffle(ans)
    return ans


def is_multi_search(dialog):
    count = 0
    for turn in dialog:
        if turn["role"] == "search":
            count += 1
    return count > 1


def process_dir(dir_path, data, n=None):
    """处理文件夹

    Args:
        dir_path (str): 数据集路径
        data (list): 数据集一行一行信息
        n (int, optional): 总数据数量. Defaults to None.

    Returns:
        list: 数据集
    """
    files = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)

    for file_path in files:
        with open(file_path, "r", encoding="utf-8") as fp:
            dialog = json.load(fp)
            data.append(dialog)

    multi = []
    single = []

    for dial in data:
        if is_multi_search(dial):
            multi.append(dial)
        else:
            single.append(dial)

    random.shuffle(single)
    if n is not None:
        single = single[: n - len(multi)]

    return multi + single


def process_dir_v2(dir_path, data):
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            with open(file_path, "r", encoding="utf-8") as fp:
                dialogs = json.load(fp)
                for dial in dialogs:
                    data.append(dial)
                    # process_dialog(dial,data)
    return data


def split_data(data, ratio):
    """划分数据集

    Args:
        data (list): 数据集
        ratio (int): 测试集&验证集比例

    Returns:
        list: 数据集
    """
    random.shuffle(data)
    dev_size = int(len(data) * ratio)
    test_size = dev_size
    train_size = len(data) - dev_size - test_size
    train_data = data[:train_size]
    dev_data = data[train_size : train_size + dev_size]
    test_data = data[train_size + dev_size :]
    return train_data, dev_data, test_data


def write_jsonl(data, filename):
    with open(filename, "w", encoding="utf-8") as fp:
        for example in data:
            json_str = json.dumps(example, ensure_ascii=False)
            fp.write(json_str + "\n")


def main(raw_data_path, more_data_path=None, output_dir=".", ratio=0.1, n=None):
    """主函数

    Args:
        raw_data_path (str): 数据集路径
        more_data_path (str, optional): 补充功能数据集. Defaults to None.
        output_dir (str, optional): 输出路径. Defaults to ".".
        ratio (float, optional): 数据集划分比例. Defaults to 0.1.
        n (str, optional): 总样本数量. Defaults to None.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = []
    data = process_dir(raw_data_path, data, n)

    if more_data_path is not None:
        data = process_dir_v2(more_data_path, data)

    train_data, dev_data, test_data = split_data(data, ratio)

    write_jsonl(
        data_to_turns(train_data),
        os.path.join(
            output_dir, "train.jsonl" if n is not None else "train.full.jsonl"
        ),
    )

    write_jsonl(
        data_to_turns(dev_data),
        os.path.join(output_dir, "dev.jsonl" if n is not None else "dev.full.jsonl"),
    )

    write_jsonl(
        data_to_turns(test_data),
        os.path.join(output_dir, "test.jsonl" if n is not None else "test.full.jsonl"),
    )


# main("enhanced_hotel_data",more_data_path="enhanced_more",n=1500)
main("enhanced_hotel_data", more_data_path="enhanced_more", n=None)
