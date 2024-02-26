import json
import random
import openai
from db_client import HotelDB

def search(db, name):
    result = db.search({'name':name}, limit=3)
    final = []
    for r in result:
        if all(char in r['name'] for char in name):
            final.append(r)
    return final

def complete_openai(prompt, model="gpt-3.5-turbo"):
    if model == "gpt-3.5-turbo-instruct":
        response = openai.Completion.create(
            model=model, prompt=prompt, max_tokens=500, temperature=0
        )
        return response.choices[0].text
    else:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.8,
        )
        return response.choices[0].message["content"]

def generate_prompt(name, searched, examples):
    instruction = """
      你的任务是根据酒店名称生成查询的语句
      并结合查到的结果，加工生成回复文本
      尽量覆盖口语化的说法，#注意不要说的太机械重复#
      选择条件为酒店名称, 在JSON格式输入中的key是name
    """
    output_format = """
      以JSON格式输出，包含字段
      - content: string类型，即用户的查询问句
      - reply: string类型，回复给用户的话术
    """
    prompt = f"""
      {instruction}
      {output_format}
      examples: 
      {examples}
      input:
      酒店名称：{name}
      查询记录：{searched}
      output:
    """
    return prompt

if __name__ == "__main__":
    # 根据名字只查到单条记录的例子
    examples1 = [
      [
       {"str": "盛厦宾馆"}, 
       {"str": "我想订一下那个盛厦宾馆，帮我查查"},
       [{"address": "北京朝阳区东三环北路16号农展馆新馆南路", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;国际长途电话;吹风机;24小时热水;无烟房;行李寄存", "hotel_id": 584, "name": "北京盛厦宾馆", "phone": "010-65916188", "price": 258, "rating": 4.4, "subway": "团结湖地铁站B口", "type": "舒适型"}],
       {"str": "欢迎您选择北京盛厦宾馆，祝您入住愉快"}
      ],
      [
       {"str": "新世贸大酒店"}, 
       {"str": "我要订个酒店，那个新世贸大酒店"},
       [{"address": "北京丰台区永外东罗园九号楼", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;吹风机;24小时热水;中式餐厅;会议室;无烟房;商务中心;棋牌室;早餐服务免费;洗衣服务;行李寄存;租车;叫醒服务", "hotel_id": 715, "name": "北京新世贸大酒店", "phone": "010-52268555", "price": 223, "rating": 4.2, "subway": "景泰站地铁站D口", "type": "舒适型"}],
       {"str": "为您查到北京新世贸大酒店，您要选择这家吗" }
      ],
      [
       {"str": "孔府酒店"}, 
       {"str": "帮我找个酒店，叫孔府酒店"}, 
       [{"address": "北京海淀区马甸西路月季园18号", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;免费市内电话;吹风机;24小时热水;中式餐厅;无烟房;早餐服务;行李寄存;叫醒服务;收费停车位", "hotel_id": 1062, "name": "北京孔府酒店", "phone": "010-68980808", "price": 402, "rating": 4, "subway": "牡丹园地铁站C口", "type": "舒适型"}],
       {"str": "为您找到北京孔府酒店。"}
      ],
    ]
    # 根据名字查到多条记录的例子
    examples2 = [
      [
       {"str": "如家快捷酒店"}, 
       {"str": "给我订个住的地方，找找如家快捷酒店"}, 
       [{"address": "北京朝阳区左家庄中街4号", "facilities": "酒店提供的设施:所有房间提供wifi;国际长途电话;24小时热水;中式餐厅;无烟房;商务中心;早餐服务;接待外宾;洗衣服务", "hotel_id": 883, "name": "如家快捷酒店(北京国展左家庄店)", "phone": "010-64686868", "price": 257, "rating": 4.4, "subway": "柳芳地铁站B口", "type": "经济型"}, {"address": "北京朝阳区劲松八区805号楼", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;无烟房;接待外宾;行李寄存;叫醒服务", "hotel_id": 937, "name": " 如家快捷酒店(北京劲松店)", "phone": "010-87776766", "price": -1, "rating": 4.6, "subway": "劲松地铁站A口", "type": "经济型"}, {" address": "北京朝阳区安立路甲52号", "facilities": "酒店提供的设施:部分房间提供wifi;宽带上网;免费市内电话;国际长途电话;吹风机;24小时热水;会议室;无烟房;接待外宾;行李寄存;叫醒服务", "hotel_id": 1015, "name": "如家快捷酒店(北京鸟巢店)", "phone": "010-84802999",  "price": 315, "rating": 4.7, "subway": "安立路地铁站B口", "type": "经济型"}],
       {"str": "为您找到如家快捷酒店(北京鸟巢店)、如家快捷酒店(北京劲松店)、如家快捷酒店(北京国展左家庄店)，请您选择" }
      ],
      [
       {"str": "汉庭酒店"}, 
       {"str": "我想住汉庭酒店，你帮我查一下都有哪些店"}, 
       [{"address": "北京朝阳区劲松九区907号楼东二环辅路路东", "facilities": "酒店提供的设施:酒店各处提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;中式餐厅;无烟房;商务中心;接待外宾;行李寄存;叫醒服务", "hotel_id": 407, "name": "汉庭酒店(北京站店)", "phone": "010-67765566", "price": 267, "rating": 4.5, "subway": "广渠门外地铁站D口", "type": "经济型"}, {"address": "北京朝阳区北苑路18号院4号楼", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;国际长途电话;吹风机;24小时热水;中式餐厅;无烟房;行李寄存;叫醒服务", "hotel_id": 953, "name": "汉庭酒店(北京北苑店)", "phone": "010-60606099", "price": 324, "rating": 4.4, "subway": "立水桥南地铁站C口", "type": "经济型"}, {"address": "北京西城区西绦胡同15号", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;吹风机;24小时热水;中式餐厅;无烟房;商务中心;早餐服务;接待外宾;洗衣服务;行李寄存;叫醒服务", "hotel_id": 648, "name": "汉庭酒店(北京鼓楼店)", "phone": "010-64000123", "price": 403, "rating": 4.3, "subway": "鼓楼大街地铁站A1口", "type": "经济型"}],
       {"str": "找到了汉庭酒店(北京鼓楼店)、汉庭酒店(北京北苑店)、汉庭酒店(北京站店)这三家，请您选择" }],
      [
       {"str": "7天酒店"}, 
       {"str": "你帮我查一下7天酒店都有哪些店"}, 
       [{"address": "北京朝阳区德胜门外黄寺大街28号", "facilities": "酒店提供的设施:所有房间提供wifi;宽带上网;24小时热水;无烟房", "hotel_id": 778, "name": "7天优品酒店(北京黄寺店)(原7天连锁酒店)", "phone": "010-59260366", "price": 332, "rating": 4.4, "subway": "安华桥地铁站D2口", "type": "经济型"}, {"address": "北京朝阳区望京南湖北路107号", "facilities": "酒店提供的设施:部分房间提供wifi;宽带上网;24小时热水;接待外宾", "hotel_id": 677, "name": "7天优品酒店(北京望京南湖东园店)(原7天连锁酒店望京南湖东园店)", "phone": "010-64725777", "price": 332, "rating": 4.1, "subway": "东湖渠地铁站D口", "type": "经济型"}, {"address": "北京海淀区定慧东里18号楼", "facilities": "酒店提供的设施:公共区域和部分房间提供wifi;宽带上网;免费国内长途电话;24小时热水;无烟房;洗衣服务", "hotel_id": 748, "name": "7天连锁酒店(北京航天桥店)", "phone": "010-88111977", "price": 316, "rating": 4.5, "subway": "西钓鱼台地铁站C口", "type": "经济型"}], {"str": "为您找到7天连锁酒店(北京航天桥店)、7天优品酒店(北京望京南湖东园店)(原7天连锁酒店望京南湖东园店)和7天优品酒店(北京黄寺店)(原7天连锁酒店)，请选择" }],
    ]
    # 读取到crosswoz酒店数据集中所有的酒店名字
    with open('names.json', 'r') as f:
        names = json.load(f)
    db = HotelDB()
    results = []
    for name in names:
        # 从单条和多条记录的例子中各选一个
        example_str = ""
        example = random.choice(examples1)
        example_str += "input:\n"
        example_str += "酒店名称："+example[0]['str']+"\n"
        example_str += "查询记录："+json.dumps(example[2],ensure_ascii=False)+"\n"
        result = {'content':example[1]['str'],'reply':example[3]['str']}
        example_str += "output:\n"+json.dumps(result,ensure_ascii=False)+"\n"
        example = random.choice(examples2)
        example_str += "input:\n"
        example_str += "酒店名称："+example[0]['str']+"\n"
        example_str += "查询记录："+json.dumps(example[2],ensure_ascii=False)+"\n"
        result = {'content':example[1]['str'],'reply':example[3]['str']}
        example_str += "output:\n"+json.dumps(result,ensure_ascii=False)+"\n"
        records = search(db, name)
        searched =  json.dumps(records,ensure_ascii=False)
        prompt = generate_prompt(name, searched, example_str)
        response = complete_openai(prompt)
        try:
            response = json.loads(response)
            result = [
                {'role':'user','content':response['content']},
                {'role':'search','arguments':{'name':name}},
                {'role':'return','records': records},
                {'role':'assistant','content':response['reply']}
            ]
            print(json.dumps(result, ensure_ascii=False, indent=4))
            results.append(result)
        except:
            pass
    # print(json.dumps(results, ensure_ascii=False, indent=4))
    with open('results.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
