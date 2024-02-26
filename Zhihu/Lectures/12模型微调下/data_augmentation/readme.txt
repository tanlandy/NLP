1. enhance1目录中脚本包含以下功能：
	i. 将原始数据中设施相关的说法，改为更口语化的表达
	ii. 在原始数据中，补充针对上文已推荐的酒店的问答，如：“XXX多少钱”，“XXX地址在哪”
	iii. 在原始数据中，补充针对上文已推荐的酒店的比较型问答，如：“哪个更便宜”
	iv. 在原始数据中，补充结束语，如：“就住XXX吧”“祝您入住愉快”

运行： 
cd enhance1
python enhance.py

2. enhance2目录中脚本包含以下功能：
	i. 限制价格上/下界的查询
	ii. 限制价格区间的查询
	iii. 组合价格与其他条件的查询
	iv. 按酒店名称查询（包括用户不说酒店全名的情况）

运行：
cd enhance2
python generate_by_filter_search.py
python generate_by_hotel_name.py