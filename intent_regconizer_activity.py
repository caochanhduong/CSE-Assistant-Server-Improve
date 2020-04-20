#!/usr/bin/env python
# coding: utf-8

from sklearn.externals import joblib
import numpy as np
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

# In[5]:
def extract_and_get_intent(message):
    list_location_notification=["ở đâu","chỗ nào","ở nơi nào","tỉnh nào","huyện nào","khu nào","địa điểm","địa chỉ"]
    list_time_notification=["khi nào","lúc nào","thời gian nào","ngày nào","ngày bao nhiêu","giờ nào","giờ bao nhiêu","mấy giờ","mấy h ","thời gian"]
    list_holder_notification=["ai tổ chức","đơn vị nào tổ chức","đơn vị tổ chức","trường nào tổ chức","clb nào tổ chức","câu lạc bộ nào tổ chức","người tổ chức"]
    list_reward_notification=["mấy ngày ctxh","mấy điểm rèn luyện","mấy drl","mấy đrl","mấy ngày công tác xã hội","bao nhiêu ngày ctxh","bao nhiêu ctxh","bao nhiêu điểm rèn luyện","bao nhiêu drl","bao nhiêu đrl","bao nhiêu ngày công tác xã hội","điểm rèn luyện","được công tác xã hội","được ctxh","được thưởng gì"]
    list_yes_no_notification=["có","không","yes","no"]
    list_dont_care_notification=["không quan tâm","kqt","không quan trọng","k quan tâm","ko quan tâm","k quan trọng","ko quan trọng"]
    list_hello_notification=["hi","hello","chào","helo"]
    
    # 7 intent with pattern matching
    for notification in list_location_notification:
        if message.find(notification)!=-1:
            return 'location',1.0


    for notification in list_time_notification:
        if message.find(notification)!=-1:
            return 'time',1.0

    for notification in list_holder_notification:
        if message.find(notification)!=-1:
            return 'holder',1.0
    for notification in list_reward_notification:
        if message.find(notification)!=-1:
            return 'reward',1.0
    
    for notification in list_yes_no_notification:
        if message.find(notification)!=-1 and len(message.split(" "))<=3:
            return 'yes_no',1.0
    
    for notification in list_dont_care_notification:
        if message.find(notification)!=-1:
            return 'dont_care',1.0

    for notification in list_hello_notification:
        if message.find(notification)!=-1 and len(message.split(" "))<=3: 
            return 'hello',1.0


    # 5 intent with machine learning
    with open('saved_model_tfidf_linearSVC_v2.pkl', 'rb') as file:
        tfidf_svc_clf = joblib.load(file)
    with open('vectorizer_for_tfidf_linearSVC_v2.pkl', 'rb') as file:
        vectorizer = joblib.load(file)
    X=vectorizer.transform([message])
    X = X.todense()
    pred_result =tfidf_svc_clf.predict(X)
    list_label=['contact','register','activity','work','joiner']
    le = preprocessing.LabelEncoder()
    le.fit_transform(list_label)
    return str(le.inverse_transform(pred_result)[0]),tfidf_svc_clf.decision_function(X)



# In[24]:

if __name__ == '__main__':
    result=extract_and_get_intent("ai tổ chức vậy bạn")
    print("------------extract raw")
    print(result)






# In[ ]:




