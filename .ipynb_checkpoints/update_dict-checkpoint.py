import json
import re
import pymongo
from pymongo import MongoClient
client = MongoClient('mongodb://caochanhduong:bikhungha1@ds261626.mlab.com:61626/activity?retryWrites=false')
db = client.activity
with open('real_dict_2000_new_only_delete_question_noti_new_and_space_newest.json','r') as dict_file:
    real_dict = json.load(dict_file)
    for key in list(real_dict.keys()):
        db_key_res = db.dictionary.find({"type":key})
        results = []
        for result in db_key_res:
            results.append(result["value"])
        real_dict[key] = results
    with open('real_dict_2000_new_only_delete_question_noti_new_and_space_newest.json', 'w+') as dict_file_new:
        json.dump(real_dict,dict_file_new,ensure_ascii=False)