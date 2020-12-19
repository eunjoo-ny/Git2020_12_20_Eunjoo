i=0

while True:
    print("{}번째 반복문입니다.".format(i))
    i+=1
    
    input_text =input("> 종료하시겠습니까(y)")
    if input_text in ["Y","y"]:
        print("반복을 종료합니다.")
        break
    
i=0
while i<10:
    print("{}번째 반복문입니다.".format(i))
    i+=1
