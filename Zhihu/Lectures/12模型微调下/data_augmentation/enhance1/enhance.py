import sys

from dotenv import load_dotenv

load_dotenv("api_keys.env")

from rewrite import UtteranceRewriter
from add_qa_turn import QAGen
from add_qa_turn2 import QAGen2
from add_bye_turn import ByeGen
from paraphrase import Paraphraser
import random, os, json

qa_gen = QAGen()
qa_gen2 = QAGen2()
bye_gen = ByeGen()
rewriter = UtteranceRewriter()
paraphraser = Paraphraser()

def flip_a_coin(p):
    return random.random() < p

def one_hotel_only(dialog):
    hotel_found = False
    for turn in dialog:
        if turn["role"] == "return":
            hotel_found = True
            if len(turn["records"]) > 1:
                return False
    return hotel_found


def one_turn_only(dialog):
    assistant_count = 0
    for turn in dialog:
        if turn["role"] == "assistant":
            assistant_count += 1
    return assistant_count == 1


def get_last_user_turn(dialog, i):
    for j in range(i - 1, -1, -1):
        if dialog[j]["role"] == "user":
            return dialog[j]
    return None


def enhance(dialog):
    changed = False
    for i, turn in enumerate(dialog):
        if turn["role"] == "search":
            if "facilities" in turn["arguments"]:
                facilities = turn["arguments"]["facilities"]
                for j, facility in enumerate(facilities):
                    if len(facility) > 4 or flip_a_coin(0.2):
                        user_turn = get_last_user_turn(dialog, i)
                        if user_turn is not None:
                            utterance = user_turn["content"]
                            if facility not in utterance:
                                continue
                            paraphrase = paraphraser.gen(facility)
                            new_utterance = rewriter.rewrite(utterance, facility, paraphrase)
                            user_turn["content"] = new_utterance
                            facilities[j] = paraphrase
                            changed = True

    if one_hotel_only(dialog):
        if flip_a_coin(0.3):
            new_turns = qa_gen.gen(dialog)
            if new_turns is not None:
                dialog.extend(new_turns)
                changed = True
    else:
        if flip_a_coin(0.3):
            new_turns = qa_gen2.gen(dialog)
            if new_turns is not None:
                dialog.extend(new_turns)
                changed = True

    if one_hotel_only(dialog):
        if flip_a_coin(0.5):
            new_turns = bye_gen.gen(dialog)
            if new_turns is not None:
                dialog.extend(new_turns)
                changed = True

    return dialog, changed

def main(start=0,end=None):
    output_dir = "enhanced_data"
    # 遍历raw_data文件夹下的所有文件
    for filename in os.listdir("raw_data"):
        id = int(filename.replace(".json", ""))
        if id < start:
            continue
        if end is not None and id >= end:
            break

        print(filename)
        with open(os.path.join("raw_data", filename), 'r', encoding='utf-8') as ifp:
            dialog = json.load(ifp)
            try:
                dialog, changed = enhance(dialog)
            except:
                changed = False
            if changed:
                filename = filename.replace(".json", "_.json")
                print("Enhanced {}".format(filename))
            ofp = open(os.path.join(output_dir, filename), 'w', encoding='utf-8')
            json.dump(dialog, ofp, indent=4, ensure_ascii=False)
            ofp.close()



if __name__ == "__main__":
    main()