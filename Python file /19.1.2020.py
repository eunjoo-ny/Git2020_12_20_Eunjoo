dictionary={"이름":"구름",
            "종족":"강아지"
            }
dictionary["이름"]#"구름". 존재유무를 
#알아보는 코드이므로 암기!
if "나이" in dictionary:
    print(dictionay["나이"])
else:
    print("없는키 입니다.")
    
dictionary={"이름":"구름",
            "종족":"강아지"
            }
if "이름" in dictionary:
    print(dictionary["이름"])
else:
    print("없는키 입니다.")
    
numbers=[1,2,6,8,4,3,2,1,9,5,4,9,7,2,1,3,5,4,8,9,7,2,3]
counter={}
for number in numbers:
    if number in counter:
      counter[number]+=1
        
    else:
      counter[number]=1 #고정적으로 사용하는 count code이므로 암기!
      
print(counter)

character={"name":"기사",
           "level":12,
           "items":{"sword":"불꽃의 검",
                    "armor":"풀플레이트"
                    },
           "skill":["베기","세게베기","아주세게베기"]
           }
for key in character:
    if type(character[key]) is dict:
        for k in character[key]:
            print("{}:{}".format(k,character[key][k]))
    elif type(character[key]) is list:
        for item in character[key]:
            print("{}:{}".format(key,item))
    else:
        pass
    
    
character={"name":"기사",
           "level":12,
           "items":{"sword":"불꽃의 검",
                    "armor":"풀플레이트"
                    },
           "skill":["베기","세게베기","아주세게베기"]
           }
for key in character:
    if type(character[key]) is dict:
        for k in character[key]:
            print("{}:{}".format(k,character[key][k]))
    elif type(character[key]) is list:
        for item in character[key]:
            print("{}:{}".format(key,item))
          
    else:
        print("{}:{}".format(key,character[key]))
        
for i in range(0,10,1):
  print(i)
  
array=[273,52,103,32,57]
i=1
for element in array:
      print("{}:{}".format(i,element))
      i+=1
      
array=[273,52,103,32,57]
for i in range(len(array)):
    print("{}:{}".format(i,array[i]))
    
array=[273,52,103,32,57]

for i in reversed(array):
       print(i)
       
    

    
       
        