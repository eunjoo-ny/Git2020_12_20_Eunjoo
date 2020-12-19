dictionary={"1class":"eunjoo",
            "2class":"ga-in",
            "3class":{"hyunho":"first-number",
                      "alex":"second-number",
                      "luna":"third-number"},
            "4class":["banana","strawberry","manggo"]}
for key in dictionary:
    if type(dictionary[key]) is dict:
        for small_key in dictionary[key]:
            print(small_key,":",dictionary[key][small_key])
    elif type(dictionary[key])is list:
        for small2_key in dictionary[key]:
            print(key,";",small2_key)
    else:
        print(key,":",dictionary[key])