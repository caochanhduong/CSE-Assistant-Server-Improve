def check_question(message):
    
    list_question_signal=["sao vậy","không vậy","chưa vậy","thế"," nhỉ "," ai"," ai ","ở đâu","ở mô","đi đâu","bao giờ","bao lâu","khi nào","lúc nào","hồi nào","vì sao","tại sao","thì sao","làm sao","như nào","thế nào","cái chi","gì","bao nhiêu","mấy","?"," hả ","được không","vậy ạ"]
    list_question_signal_last=["vậy","chưa","không","sao","à","hả","nhỉ"]
    list_object=["bạn","cậu","ad","anh","chị","admin","em","mày"]
    list_subject=["mình","tôi","tớ","tao","tui"]
    list_verb_want=["hỏi","biết","xin"]
    list_verb_have=["có","được"]

#     ... subject muốn hỏi/biết/xin… *
# ... subject muốn được hỏi/tư vấn .... *
# ... cho subject hỏi/biết/xin *
# ... subject cần hỏi/biết/xin... *
# .... subject cần ... thông tin … *
# .... gửi subject ... *
# ... cho hỏi/xin/biết… *
# ... có/được (...)? không (...) *
    #bắt WH question 
    for signal in list_question_signal:
        if signal in message.lower():
            print(signal)
            return True

    for verb in list_verb_have:
        if (message.lower().find(verb)!=-1 and message.lower().find("không")!=-1 and message.lower().find(verb)<message.lower().find("không")):
            print("1")
            return True

    for subject in list_subject:
        for verb in list_verb_want:
            if (message.lower().find(subject+" muốn "+verb)!=-1 or message.lower().find("cho "+subject+" "+verb)!=-1 or message.lower().find(subject+" cần "+verb)!=-1):
                print("2")
                return True

    for subject in list_subject:
        if (message.lower().find(subject+" muốn được hỏi")!=-1 or message.lower().find(subject+" muốn được tư vấn")!=-1):
            print("3")
            return True
        if (message.lower().find(subject+" cần")!=-1 and message.lower().find("thông tin")!=-1 and message.lower().find(subject+" cần")<message.lower().find("thông tin")):
            print("4")
            return True
        if (message.lower().find(subject+" muốn")!=-1 and message.lower().find("thông tin")!=-1 and message.lower().find(subject+" muốn")<message.lower().find("thông tin")):
            print("4")
            return True
        if (message.lower().find("gửi "+subject)!=-1):
            print("5")
            return True

    for verb in list_verb_want:
        if (message.lower().find("cho "+verb)!=-1):
            print("6")
            return True

    #bắt YES-NO/WH question mà signal cuối câu 
    if len(message.split(" "))>3 and (message.split(" ")[-1].lower()=="chưa" or message.split(" ")[-1].lower()=="không" or message.split(" ")[-1].lower()=="ta" or message.split(" ")[-1].lower()=="sao" or message.split(" ")[-1].lower()=="nhỉ" or message.split(" ")[-1].lower()=="nào"):
        # print("7")
        return True

    #bắt YES-NO question cuối câu có chủ ngữ

    for subject in list_object:
        for question_signal_last in list_question_signal_last:
            if message.split(" ")[-1].lower()==subject and message.split(" ")[-2].lower()==question_signal_last:
                print("8")
                return True 
    

    return False

# print(check_question("liên hệ người nào nhỉ"))

# print(check_question("Mình là nữ thì có được đi Mùa hè xanh khoa Máy tính không ad"))
# print(check_question("Đi Mùa hè xanh ngoài làm đường ra thì mình còn làm gì nữa bạn"))
# print(check_question("Cho mình xin thông tin liên hệ của Ban tổ chức Đêm hội trăng rằm với"))


