import codecs
import json
import sys


with codecs.open('naive_bert.json', 'r', encoding='utf-8') as fr:
	res = []
	for line in fr.readlines():
		tmp = json.loads(line)

		res.append(tmp)
	json.dump(res, open(sys.argv[2], 'w', encoding='utf-8'), ensure_ascii=False)