import re 

def extract_information(message):
    phone_pattern=r'0\d{9}'
    email_pattern=r"[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+"
    name_pattern=r'[A-Z][^A-Z ,.!@?&*$~+;]*'
    # input_string='mình tên Cao Chánh Dương, số điện thoại của mình là 0329581621, còn email là duongcc@gmail.com duongcc@vng.com.vn đó bạn ơi'
    # if re.match(phone_pattern,input_string)!=None:
    #     print(re.match(phone_pattern,input_string).group(1))
    emails = re.findall(email_pattern, message)
    phones=re.findall(phone_pattern,message)
    names=re.findall(name_pattern,message)    
    return emails,phones," ".join(names)