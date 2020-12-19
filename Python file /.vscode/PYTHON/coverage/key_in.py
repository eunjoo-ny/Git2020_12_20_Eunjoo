dictionary={"name":"Eunjoo",
            "height":"163cm",
            "weight":"secrete",
            "grade":"normal",
            "weather":"cloudy",
            "transportation":"train"}

key=input("Insert the accessible key:")


# I don't know this error.
   if key in dictionary:
    print(dictionary[key])
    
    else:
        print("There is no key in dictionary.")