with open("info.txt","r")as file:
    for line in file:
        (name,weight,height)=line.strip().split(",")
        
        if not name or not weight or not height:
            continue
        bmi=int(weight)/(int(height)*int(height))
        result=""
        if 25<=bmi:
                result="over weight"
        elif 18.5<=bmi:
                result="normal weight"
        else:
                result="less weight"
        print("\n".join(["name:{}","weight:{}",
                          "height:{}",
                          "bmi:{}","result:{}"]).format(name,weight,height,bmi,result))
        print()