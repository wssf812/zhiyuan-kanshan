from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


# 训练tfidf
def trainTfidf(corpus):
    tfidf = TfidfVectorizer(token_pattern=r"(?u)\bsw\d+\b")
    tfidf.fit(corpus)
    joblib.dump(tfidf, "tfidf.m")


def createQandM():
    dict = {}
    list=[]
    with open(r"F:\zhiyuan_competition\question_info.txt", 'r', encoding="utf-8") as file:
        for i in file.readlines():
            data = i.replace("\n", "").split("\t")
            dict[data[0]] = data[2]
    count=0
    with open("newTrain.txt", 'a', encoding="utf-8") as newf:
        with open(r"F:\zhiyuan_competition\invite_info.txt", 'r', encoding="utf-8") as f:
             for j in f:
                 data2=j.strip().split("\t")
                 if data2[0] in dict.keys():
                     newf.write(data2[0]+" "+data2[1]+" "+dict[data2[0]]+" "+data2[-1]+"\n")
                     count+=1
                     print(count)


def trainQandM():
    tfidf=joblib.load("tfidf.m")
    # feature_names=tfidf.get_feature_names()#查看语料库特征名字
    with open(r"F:\MuifeaTrain.txt","a",encoding="utf-8") as file:
        with open("newTrain.txt","r",encoding="utf-8") as f:
            count = 0
            for elem in f:
                data=elem.strip().split(" ")
                tfidfFeatures=tfidf.transform([data[2].replace(","," ")]).toarray()[0]
                file.write(data[1]+"\t"+" ".join(list(map(str,tfidfFeatures)))+"\t"+data[-1]+"\n")
                count+=1
                print(count)




# 单一性别特征提取训练集
def readfile1():
    dict = {}  # F:\zhiyuan_competition\
    with open(r"F:\zhiyuan_competition\member_info.txt") as file:
        for i in file.readlines():
            data = i.replace("\n", "").split("\t")
            if data[1] == "male":
                dict[data[0]] = [1, 0, 0]
            elif data[1] == 'female':
                dict[data[0]] = [0, 1, 0]
            else:
                dict[data[0]] = [0, 0, 1]
    dict2 = {}
    with open(r"F:\zhiyuan_competition\invite_info.txt", 'r', encoding="utf-8") as f:
        for j in f.readlines():
            data2 = j.replace("\n", "").split("	")
            if data2[1] not in dict2:
                list = []
                list.append([int(i) for i in data2[-1]])
                dict2[data2[1]] = list
            else:
                dict2[data2[1]].append([int(i) for i in data2[-1]])
    count = 0
    with open("train.txt", "a", encoding="utf-8") as fi:
        for key, value in tqdm(dict2.items()):
            if key in dict:
                dict2[key].append(dict[key])
                for j in value[0:-1]:
                    count += 1
                    fi.write(
                        key + " " + " ".join([str(i) for i in value[-1]]) + " " + "".join([str(i) for i in j]) + "\n")
                    print("写入" + str(count) + "数据")


def readfile2(path):
    list = []
    with open(path, 'r', encoding="utf-8") as f:
        for i in f:
            list.append(i.replace("\n", ""))
    return list


def train(path):
    Xtrain = []
    Ytrain = []
    count=0
    with open(path, 'r', encoding="utf-8") as f:
        for elem in f:
              count+=1
              Xtrain.append(list(map(float, elem.strip().split("\t")[1].split(" "))))
              Ytrain.append(int(elem.strip().split("\t")[-1]))
              print(count)
        print(Xtrain)
        print(Ytrain)
        LR = LogisticRegression()  # solver="lbfgs",class_weight='balanced'
        LR.fit(Xtrain, Ytrain)
        train_score = LR.score(Xtrain, Ytrain)
        print("训练完成，训练集得分:" + str(train_score))
        joblib.dump(LR, "trainmodel.m")
        print("模型保存成功...")


if __name__ == '__main__':
    path=r"F:\MuifeaTrain.txt"
    train(path)
    # readfile1()

   # createQandM()

     # trainQandM()