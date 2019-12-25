with open("./data/traindata.txt","w",encoding="utf-8") as f:
    with open("./data/train.txt",'r',encoding="utf-8") as file:
        for s in file.readlines()[0:100000]:
            f.write(s.strip()+"\n")
