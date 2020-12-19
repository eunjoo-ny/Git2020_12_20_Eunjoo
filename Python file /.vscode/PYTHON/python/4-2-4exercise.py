character={"name":"knight",
            "level":"12",
            "items":{"sword":"fire knife", 
            "armor":"full plate",
            "power":"wind"
            },
            "skill":["stabbing","strong stabbing","very strong stabbing"]
            }
for key in character:
   if type(character[key])is dict:
     for small_key in character[key]:
      print(small_key,":",character[key][small_key])
   elif type(character[key]) is list:
    for item in character[key]:
        print(key,":",item)
   else:
    print(key,":",character[key])
