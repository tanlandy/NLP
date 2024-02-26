import json
import random
import openai

############## 全局变量 #################
# 写好的一些例子
examples = [
  [{"price_range_upper": 500, "type": "豪华型" }, 
   { "content": "帮我找个价格在500元之内的豪华酒店。" }],
  [{"price_range_lower": 400, "rating_range_lower": 4}, 
   { "content": "我要订一个价格在400元之上的不低于4分的宾馆" }],
  [{"price_range_upper": 500, "facilities": ["热水","wifi"]}, 
   { "content": "给我查查酒店，有热水和wifi的，价格在500以里的" }],
  [{"price_range_upper": 500, "type": "经济型"}, 
   { "content": "你好，有价格在500以内的酒店可以订吗，经济型的就行" }],
  [{"price_range_upper": 200, "rating_range_lower": 4}, 
   { "content": "请给我订个评分高于4，价格在200元以上酒店。" }],
  [{"price_range_lower": 300, "facilities": ["洗衣", "洗澡"] }, 
   { "content": "有人吗，订一个在300元之上的能洗衣洗澡的宾馆噢" }],
  [{"price_range_upper": 400, "type": "经济型"}, 
   { "content": "帮我找个价格比400低的经济酒店。" }],
  [{"price_range_upper": 900, "facilities": ["停车"] }, 
   { "content": "我要找一个比900元便宜的酒店，最好能停车的啊" }],
  [{"price_range_upper": 500, "facilities": ["棋牌室"]}, 
   { "content": "帮个忙，找一下价格在500内的酒店，要有棋牌室的噢" }],
  [{"price_range_upper": 800, "type":"舒适型"}, 
   { "content": "我要找价格在800以内的酒店，给我查查有舒适的吗" }],
]
# 可选筛选条件范围如下
keys = ['type', 'facilities', 'price_range_upper', 'price_range_lower', 'rating_range_lower', 'rating_range_upper']
types = ['豪华型','舒适型','经济型']
facilities = ['热水','wifi','停车场','按摩','棋牌室','洗衣机','泳池']
#########################################

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

def generate_prompt(input_text, examples):
    """
    根据随机生成的结构化筛选条件，通过prompt让gpt来反向写出自然语言的查询问句
    """
    instruction = """
      你的任务是根据用户对酒店的选择条件，生成用户查询的问句
      尽量覆盖口语化的说法，#注意不要说的太机械重复#
      酒店包含属性分别是: 酒店类型(type)、价格(price)、评分(rating)、酒店设施(facilities)。
      在JSON格式输入中的key有：type, facilities, price_range_upper, price_range_lower, rating_range_lower, rating_range_upper
    """
    output_format = """
      以JSON格式输出，只包含字段content: string类型，即用户的查询问句
    """
    prompt = f"""
      {instruction}
      {output_format}
      examples: 
      {examples}
      input:
      {input_text}
      output:
    """
    return prompt

def generate_data(arguments):
    # 为了保证生成问句的多样性，prompt中的例子是从写好的一些例子中做随机挑选的
    example_str = ""
    for i in range(4): # 这里挑选了4条例子给到prompt
        example = random.choice(examples)
        example_str += json.dumps(example[0], ensure_ascii=False)+"\n"
        example_str += json.dumps(example[1], ensure_ascii=False)+"\n"
    prompt = generate_prompt(arguments, example_str)
    response = complete_openai(prompt)
    try:
        result = [
            {'role':'user','content':json.loads(response)},
            {'role':'search','arguments':arguments}
        ]
        return result
    except:
        return []

def generate_price_bound(nums):
    """
    生成只有价格上限或下限的数据
    """
    results = []
    for i in range(nums):
        arguments = {
            random.choice(['price_range_upper','price_range_upper']): random.randint(2,12)*100, 
        }
        results.append(generate_data(arguments))
    return results

def generate_price_range(nums):
    """
    生成价格范围的数据
    """
    results = []
    for i in range(nums):
        price_range_lower = random.randint(2,12)*100
        price_range_upper = price_range_lower + random.randint(1,8)*100
        arguments = {
            'price_range_lower': price_range_lower, 
            'price_range_upper': price_range_upper, 
        }
        results.append(generate_data(arguments))
    return results

def generate_misc_filter(nums):
    """
    生成各种条件组合查询的数据
    """
    results = []
    for i in range(nums):
        arguments = {
            'price_range_upper': random.randint(2,12)*100, # 限价格上限的多
            'rating_range_lower': random.randint(2,4),     # 限评分下限的多
            'facilities': random.sample(facilities, k=2),
            'type': random.choice(types),
        }
        keys_to_remove = random.sample(list(arguments.keys()), 2)
        for key in keys_to_remove:
            arguments.pop(key)
        results.append(generate_data(arguments))
    return results

if __name__ == "__main__":
    results = generate_price_bound(20)
    with open('price_bound.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))

    results = generate_price_range(20)
    with open('price_range.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))

    results = generate_misc_filter(20)
    with open('misc_filter.json', 'w') as f:
        f.write(json.dumps(results, ensure_ascii=False, indent=4))
