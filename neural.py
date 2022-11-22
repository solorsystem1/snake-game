import tensorflow as tf
import numpy as np
import random
import copy

class Neural():
    def __init__(self):
        self.K=30
        self.N=10
        self.model= tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(402),
            tf.keras.layers.Dense(512,activation=tf.nn.relu),
            tf.keras.layers.Dense(4,activation=tf.nn.sigmoid)
        ])

    def fitting_function(self,weights): #가상으로 게임을 한번 돌려서 score값 구함
        # 처음 생성
        now = [random.randrange(0, 20), random.randrange(0, 20)]
        que = [[0, 0]]
        length = 2
        dir = [0, 0]
        state = []
        score = 0
        remain_move = 300
        target = [random.randrange(0, 20), random.randrange(0, 20)]

        for i in range(20):
            state.append([])
            for j in range(20):
                state[i].append(0)

        state[now[0]][now[1]] = 1
        que.append(copy.deepcopy(now))
        while state[target[0]][target[1]] != 0:
            target = [random.randrange(0, 20), random.randrange(0, 20)]

        #반복
        while 1:
            # 모델 가져오기
            pred=self.predict_model(weights,que,remain_move,target,length)[0]
            if np.argmax(pred)==1:
                dir=[0,1]
            elif np.argmax(pred) == 2:
                dir = [0, -1]
            elif np.argmax(pred) == 3:
                dir = [1, 0]
            elif np.argmax(pred) == 4:
                dir = [-1, 0]

            remain_move-=1
            if remain_move == 0:  # 남은 움직임이 없을 때
                return score+(300-remain_move)/1000
            elif now[0] + dir[0] < 0 or now[0] + dir[0] >= 20 or now[1] + dir[1] < 0 or \
                    now[1] + dir[1] >= 20:  # 맵밖으로 나감
                return score+(300-remain_move)/1000
            elif state[now[0] + dir[0]][now[1] + dir[1]] == 1:  # 자기자신에 부딪힘
                return score+(300-remain_move)/1000
            elif now[0] + dir[0] == target[0] and now[1] + dir[1] == target[1]:  # target에 도착
                length += 1
                remain_move = 300
                score+=1
                state[now[0] + dir[0]][now[1] + dir[1]] = 1
                while state[target[0]][target[1]] != 0:
                    target = [random.randrange(0, 20), random.randrange(0, 20)]
            else :
                state[now[0]+dir[0]][now[1]+dir[1]] = 1

    def refine_input(self,que,remain_move,target,length):
        ary=np.zeros(402)

        for i in range(len(que)):
            ary[que[i][0] + 20 * que[i][1]] = i

        ary[target[0] + 20 * target[1]]=400

        ary[400]=remain_move
        ary[401]=length

        ary=ary/400

        return ary

    def init_make_model(self):
        weight0 = np.random.rand(402, 512) * 0.1
        weight1 = np.zeros(512)
        weight2 = np.random.rand(512, 4) * 0.05
        weight3 = np.zeros(4)
        weights = np.array([weight0, weight1, weight2, weight3])
        return weights


    def predict_model(self,weights,que,remain_move,target,length):
        self.model.set_weights(weights)

        input_data=self.refine_input(que,remain_move,target,length)
        input_data=input_data.reshape(-1,402)
        pred=self.model.predict(input_data)

        return pred

    def genetic_algorithm(self,step):
        weights_list=[]

        for i in range(self.K):
            weights_list.append(self.init_make_model())

        for k in range(step):
            sum_fit = 0
            fit_list = []
            parent_list = []
            for i in range(self.K): #fit 누적값 저장
                fit_val = self.fitting_function(weights_list[i])
                sum_fit+=fit_val
                fit_list.append(fit_val)
            print('1')
            print(sum_fit)
            if sum_fit!=0:
                for i in range(self.K):
                    fit_list[i]=fit_list[i]/sum_fit

            print('2')
            #상위 N개 자식 생성
            selected_list=[]
            m=0
            while(m<self.N):
                parent_idx=random.choices(range(0,self.K),weights=fit_list)
                if parent_idx[0] not in selected_list:
                    parent_list.append(copy.deepcopy(weights_list[parent_idx[0]]))
                    selected_list.append(parent_idx[0])
                    m+=1

            print('3')
            cross_prob=0.8
            mutate_prob=0.07
            child_list=[]

            #2개의 자식 선택 -> 교차
            for i in range(self.K):
                parent_idx=np.random.choice(self.N,2 ,replace=False)
                parent_1=parent_list[parent_idx[0]]
                parent_2=parent_list[parent_idx[1]]
                #print("C")
                child=self.crossover(parent_1,parent_2,cross_prob)

                if np.random.random() < mutate_prob:
                    child = self.mutation1(child)
                if np.random.random() < mutate_prob:
                    child = self.mutation2(child)

                child_list.append(child)

            print('4')
            weights_list=copy.deepcopy(child_list)
            print(len(weights_list))

        print('end')
        return weights_list

    def mutation1(self,child):
        #print("A")
        randpos=np.random.choice(402*512,2,replace=False)
        #print("A")
        tem=copy.deepcopy(child[0][randpos[0]%402][int(randpos[0]/402)])
        #print("A")
        child[0][randpos[0]%402][int(randpos[0]/402)]=copy.deepcopy(child[0][randpos[1]%402][int(randpos[1]/402)])
        #print("A")
        child[0][randpos[1]%402][int(randpos[1]/402)]=tem
        #print("A")
        return child

    def mutation2(self,child):
        #print("B")
        randpos=np.random.choice(4*512,2,replace=False)
        #print("B")
        tem=copy.deepcopy(child[2][randpos[0]%512][int(randpos[0]/512)])
        #print("B")
        child[2][randpos[0]%512][int(randpos[0]/512)]=copy.deepcopy(child[2][randpos[1]%512][int(randpos[1]/512)])
        #print("B")
        child[2][randpos[1]%512][int(randpos[1]/512)]=tem
        #print("B")
        return child

    def crossover(self,parent1,parent2,cross_prob):
        length1=int(len(parent1[0]) * cross_prob)
        length2 = int(len(parent1[1]) * cross_prob)
        parent1[0][:length1]=parent2[0][:length1]
        parent1[2][:length2]=parent2[2][:length2]

        return parent1