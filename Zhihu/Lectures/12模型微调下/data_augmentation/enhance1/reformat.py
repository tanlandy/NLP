import json
import os

def load_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

def reformat_dialog(dialog,outptu_file):
    formatted_dialog = []
    for turn in dialog:
        if turn["role"] == "sys":
            formatted_dialog.append({
                "role" : "assistant",
                "content" : turn["content"]
            })
        else:
            user_turn = {
                "role" : "user",
                "content" : turn["content"]
            }
            formatted_dialog.append(user_turn)
            if "dsl" in turn:
                function_turn = {
                    "role" : "search",
                    "arguments" : turn["dsl"]
                }
                formatted_dialog.append(function_turn)
            if "hotels" in turn:
                result_turn = {
                    "role" : "return",
                    "records" : turn["hotels"]
                }
                formatted_dialog.append(result_turn)
    with open(outptu_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_dialog, f, indent=4, ensure_ascii=False)

def main():
    output_dir = "raw_data"
    data = load_data("dialog.json")
    for i, dialog in enumerate(data):
        reformat_dialog(dialog, os.path.join(output_dir, "{:04}.json".format(i)))


if __name__ == "__main__":
    main()