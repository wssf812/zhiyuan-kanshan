import numpy as np
from sklearn.externals import joblib
from tqdm import tqdm


def createEval():
    dict = {}
    with open(r"F:\zhiyuan_competition\question_info.txt", 'r', encoding="utf-8") as file:
        for i in file.readlines():
            data = i.strip().split("\t")
            dict[data[0]] = data[2]

    count = 0
    with open("newEval.txt", 'a', encoding="utf-8") as newf:
        with open(r"F:\zhiyuan_competition\invite_info_evaluate_1.txt", 'r', encoding="utf-8") as f:
            for j in f:
                data2 = j.strip().split("\t")
                if data2[0] in dict.keys():
                    newf.write("\t".join(data2) + "\t" + dict[data2[0]] + "\n")
                    count += 1
                    print(count)


def createTest():
    tfidf = joblib.load("tfidf.m")
    # feature_names=tfidf.get_feature_names()#查看语料库特征名字
    with open(r"F:\MuifeaText.txt", "a", encoding="utf-8") as file:
        with open("newEval.txt", "r", encoding="utf-8") as f:
            count = 0
            for elem in f:
                data = elem.strip().split("\t")
                tfidfFeatures = tfidf.transform([data[3].replace(",", " ")]).toarray()[0]
                file.write(data[0]+"\t"+data[1]+"\t"+data[2] + "\t" + " ".join(list(map(str,tfidfFeatures)))  +"\n")
                count += 1
                print(count)


def readfile1():
    dict = {}
    with open(r"F:\zhiyuan_competition\member_info.txt") as file:
        for i in file.readlines():
            data = i.replace("\n", "").split("	")
            if data[1] == "male":
                dict[data[0]] = [1, 0, 0]
            elif data[1] == 'female':
                dict[data[0]] = [0, 1, 0]
            else:
                dict[data[0]] = [0, 0, 1]
    print(len(dict))
    count = 0
    with open(r"F:\zhiyuan_competition\invite_info_evaluate_1.txt", 'r', encoding="utf-8") as f:
        with open("evaldata.txt", "a", encoding="utf-8") as file:
            for j in tqdm(f.readlines()):
                data2 = j.replace("\n", "").split("	")
                if data2[1] in dict.keys():
                    count += 1
                    # file.write(" ".join(data2)+" "+" ".join(str(key) for key in dict[data2[1]])+"\n")
                    # print("写入" + str(count) + "数据")
                else:
                    print(data2[1])


def readfile2(path):
    list = []
    with open(path, 'r', encoding="utf-8") as f:
        for i in f.readlines():
            list.append(i.replace("\n", ""))
    return list


def eval(path, model):
    count = 0
    LR = joblib.load(model)
    with open(r"result.txt", "a", encoding="utf-8") as file:
        for i in readfile2(path):
            Xtest = []
            Xtest.append(list(map(int, i.split(" ")[3:])))
            result = LR.predict_proba(Xtest)
            # result1 = LR.predict(Xtest)
            file.write("\t".join(i.replace("\n", "").split(" ")[0:3]) + "\t" + str(result[0][1]) + "\n")
            count += 1
            print("写入" + str(count) + "条数据")


if __name__ == '__main__':
    # path = "evaldata.txt"
    # model = "trainmodel.m"
    # eval(path, model)
    # readfile1()
    # createEval()
     createTest()