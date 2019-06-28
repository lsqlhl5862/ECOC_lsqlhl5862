import numpy as np
from matplotlib import pyplot as plt
# rand_col = np.int8(np.random.rand(10) > 0.5)
# print(rand_col)

# a = np.array([1, 2, 3, 4])
# b = np.array([1, 2, 3, 5])
# print(np.int8(a == b))


# a = np.array([1,0,1,0])
# print(np.int8(np.logical_not(a)))

# a = [0.38877755511022044, 0.39478957915831664, 0.2685370741482966, 0.46893787575150303, 0.36472945891783565, 0.43286573146292584, 0.3527054108216433, 0.39879759519038077, 0.3667334669338677, 0.37074148296593185]
def test():
    a=[1,0,0,1]
    a=np.array(a)
    b=[0,0,0,1]
    b=np.array(b)
    # print(np.mean(a))
    a=np.vstack((a,b))
    # tmp_a=np.where(a==1)[0]
    # tmp_b=np.argmax(b)
    # print(tmp_a)
    # print(tmp_b)
    # result=(tmp_a==tmp_b)
    # print(np.argwhere(result==True).shape[0]==0)
    # a = [1.0, 2.0]
    # a = [int(e) for e in a]
    # temp=[]
    # a=1
    # temp.append([2,3])
    # print(temp)
    print(a.tolist())
    draw_hist(a.tolist(),"test","Col","Score",0,4,0,1)

def draw_hist(myList, Title, Xlabel, Ylabel, Xmin, Xmax, Ymin, Ymax):
    name_list = list(range(len(myList[0])))
    plt.figure()
    # name_list.reverse()
    total_width, n = 0.8, 2  
    width = total_width / n 
    x=list(range(len(myList[0])))
    rects1 = plt.bar(x, myList[0],width=width, fc = 'y')
    for i in range(len(x)):  
        x[i] = x[i] + width  
    rects2 = plt.bar(x, myList[1],width=width, fc = 'r')
    # X轴标题
    index = list(range(len(myList[0])))
    # index = [float(c)+0.4 for c in range(len(myList))]
    plt.ylim(ymax=Ymax, ymin=Ymin)
    plt.xticks(index, name_list)
    plt.ylabel(Ylabel)  # X轴标签
    plt.xlabel(Xlabel)
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height,
                 str(height), ha='center', va='bottom')
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height,
                 str(height), ha='center', va='bottom')             
    plt.title(Title)
    # plt.savefig("pictures/"+file_name+".png")
    plt.show()

if __name__ == '__main__':
    # run_sample()
    test()