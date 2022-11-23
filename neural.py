import tensorflow as tf
import numpy as np
import random
import copy
import datetime

class Neural():
    def __init__(self):
        self.K=30
        self.N=6
        self.input_node=6
        self.hidden_node=48
        self.output_node=3
        self.model= tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(self.input_node),
            tf.keras.layers.Dense(self.hidden_node,activation=tf.nn.relu),
            tf.keras.layers.Dense(self.output_node,activation=tf.nn.sigmoid)
        ])

    def fitting_function(self,weights): #가상으로 게임을 한번 돌려서 score값 구함
        # 처음 생성

        now = [random.randrange(0, 20), random.randrange(0, 20)]
        que = [[0, 0]]
        length = 2
        dir = np.array([1, 0])
        state = []
        score = 0
        remain_move = 200
        additional=0
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

            pred=self.predict_model(weights,now,state,target,dir)[0]

            if np.argmax(pred)==0:
                dir = np.array(dir)@np.array([[0,-1],[1,0]])
            elif np.argmax(pred) == 1:
                dir = np.array(dir) @ np.array([[0, 1], [-1, 0]])

            '''
            if np.argmax(pred)==0:
                dir=[0,1]
            elif np.argmax(pred)==1:
                dir=[0,-1]
            elif np.argmax(pred)==2:
                dir=[1,0]
            elif np.argmax(pred)==3:
                dir=[-1,0]
            '''

            if dir[0]*(target[0]-now[0])<0 or dir[1]*(target[1]-now[1])<0:
                additional-=0.5

            else :
                additional+=0.25

            #print(np.argmax(pred))
            remain_move-=1
            if remain_move == 0:  # 남은 움직임이 없을 때
                #score값 + 생존 횟수 +
                #return max(score+(300-remain_move)/500+(40-(abs(now[0]-target[0])+abs(now[1]-target[1])))/40+additional,0)

                #return max(score + (300 - remain_move) / 500 + additional+(40-(abs(now[0]-target[0])+abs(now[1]-target[1])))/40, 0.0001)
                return max(score+additional,0.0001)
            elif now[0] + dir[0] < 0 or now[0] + dir[0] >= 20 or now[1] + dir[1] < 0 or \
                    now[1] + dir[1] >= 20:  # 맵밖으로 나감
                return max(score + additional, 0.0001)
                #return max(score + (300 - remain_move) / 500 + additional+(40-(abs(now[0]-target[0])+abs(now[1]-target[1])))/40, 0.0001)
            elif state[now[0] + dir[0]][now[1] + dir[1]] == 1:  # 자기자신에 부딪힘
                return max(score + additional, 0.0001)
                #return max(score + (300 - remain_move) / 500 + additional+(40-(abs(now[0]-target[0])+abs(now[1]-target[1])))/40, 0.0001)
            elif now[0] + dir[0] == target[0] and now[1] + dir[1] == target[1]:  # target에 도착

                length += 1
                remain_move = 200
                score+=15
                state[now[0] + dir[0]][now[1] + dir[1]] = 1
                while state[target[0]][target[1]] != 0:
                    target = [random.randrange(0, 20), random.randrange(0, 20)]
            else :
                state[now[0]+dir[0]][now[1]+dir[1]] = 1

            now[0]=now[0]+dir[0]
            now[1]=now[1]+dir[1]
            que.append(copy.deepcopy(now))

            if len(que) > length:
                state[que[0][0]][que[0][1]] = 0
                que.pop(0)


    def refine_input(self,now,state,target,dir):
        # 위 아래 오른쪽 왼쪽 ->장애물 5칸씩 확인 0.2,0.4,0.6,0.8,1 (4개 node)
        # target 위치 사분면으로 표시 0,1값들 (4개 node)

        ary=np.zeros(self.input_node)
        right=dir@np.array([[0,-1],[1,0]])
        left=dir@np.array([[0,1],[-1,0]])

        #print(state[now[0]+5*dir[0]][now[1]+5*dir[1]])
        for i in range(1,6):
            if now[1]+i*dir[1]>=20 or now[1]+i*dir[1]<0 or now[0]+i*dir[0]>=20 or now[0]+i*dir[0]<0:
                ary[0]=1-(i-1)*0.2
                break
            elif state[now[0]+i*dir[0]][now[1]+i*dir[1]]==1:
                ary[0]=1-(i-1)*0.2
                break


        for i in range(1,6):
            if now[1]+i*right[1]>=20 or now[1]+i*right[1]<0 or now[0]+i*right[0]>=20 or now[0]+i*right[0]<0:
                ary[1]=1-(i-1)*0.2
                break
            elif state[now[0]+i*right[0]][now[1]+i*right[1]]==1:
                ary[1]=1-(i-1)*0.2
                break


        for i in range(1,6):
            if now[1]+i*left[1]>=20 or now[1]+i*left[1]<0 or now[0]+i*left[0]>=20 or now[0]+i*left[0]<0:
                ary[2]=1-(i-1)*0.2
                break;
            elif state[now[0]+i*left[0]][now[1]+i*left[1]]==1:
                ary[2]=1-(i-1)*0.2
                break;
        '''
        for i in range(1,6):
            if now[0]-i<0 or state[now[0]-i][now[1]]==1:
                ary[3]=1-(i-1)*0.2
                break;
        
        if(now[0]<target[0]):
            ary[4]=1
        if(now[0]>target[0]):
            ary[5]=1
        if(now[1]<target[1]):
            ary[6]=1
        if(now[1]>target[1]):
            ary[7]=1

        ary[8]=(abs(now[0]-target[0])+abs(now[1]-target[1]))/40
        ary[9]=(1 if dir[0]==1 else 0)
        ary[10]=(1 if dir[0]==-1 else 0)
        ary[11]=(1 if dir[1]==1 else 0)
        ary[12]=(1 if dir[1]==-1 else 0)
        '''

        if (np.array(target)-np.array(now)).dot(right)>0:
            ary[3]=1
        if (np.array(target)-np.array(now)).dot(left)>0:
            ary[4]=1
        if (np.array(target)-np.array(now)).dot(right)==0 and (np.array(target)-np.array(now)).dot(dir)>0:
            ary[5]=1

        return ary

    def init_make_model(self):
        weight0 = np.random.rand(self.input_node, self.hidden_node) * 0.1
        weight1 = np.zeros(self.hidden_node)
        weight2 = np.random.rand(self.hidden_node, self.output_node) * 0.05
        weight3 = np.zeros(self.output_node)
        weights = np.array([weight0, weight1, weight2, weight3])
        return weights


    def predict_model(self,weights,now,state,target,dir):

        self.model.set_weights(weights)
        input_data=self.refine_input(now,state,target,dir)
        input_data=input_data.reshape(-1,self.input_node)
        pred=self.model.predict(input_data,verbose=0)
        return pred

    def genetic_algorithm(self,step):
        weights_list=[]

        weights_by_step1=[]
        weights_by_step2 = []

        for i in range(self.K):
            weights_list.append(self.init_make_model())

        for k in range(step):
            sum_fit = 0
            fit_list = []
            parent_list = []
            for i in range(self.K): #fit 누적값 저장
                fit_val = self.fitting_function(weights_list[i])
                print(f"step {k} : [{i}번째 부모 test]  fit_val : {fit_val}")
                sum_fit+=fit_val
                fit_list.append(fit_val)

            print(max(fit_list))

            weights_by_step1.append(weights_list[np.argmax(fit_list)][0])
            weights_by_step2.append(weights_list[np.argmax(fit_list)][2])

            np.argmax(fit_list)
            if sum_fit!=0:
                for i in range(self.K):
                    fit_list[i]=fit_list[i]/sum_fit

            #print('2')
            #상위 N개 자식 생성
            selected_list=[]
            m=0
            random.seed(datetime.datetime.now())

            while(m<self.N):
                parent_idx=random.choices(range(0,self.K),weights=fit_list)
                if parent_idx[0] not in selected_list:
                    parent_list.append(copy.deepcopy(weights_list[parent_idx[0]]))
                    selected_list.append(parent_idx[0])
                    m+=1

            #print('3')
            cross_prob=0.3
            mutate_prob=0.07
            child_list=[]

            #2개의 자식 선택 -> 교차
            for i in range(self.K):
                if random.random()<cross_prob:
                    parent_idx=np.random.choice(self.N,2 ,replace=False)
                    parent_1=parent_list[parent_idx[0]]
                    parent_2=parent_list[parent_idx[1]]
                    #print("C")
                    child=self.crossover(parent_1,parent_2,0.8)
                else:
                    child=parent_list[random.randrange(0,self.N)]
                if np.random.random() < mutate_prob:
                    child = self.mutation1(child)
                if np.random.random() < mutate_prob:
                    child = self.mutation2(child)

                child_list.append(child)

            #print('4')
            weights_list=copy.deepcopy(child_list)
            print(f"step : {k}-----------------------")

        print('end')
        np.save('weights1',weights_by_step1)
        np.save('weights2', weights_by_step2)
        return weights_list

    def mutation1(self,child):
        #print("A")
        randpos=np.random.choice(self.input_node*self.hidden_node,2,replace=False)
        #print("A")
        tem=copy.deepcopy(child[0][randpos[0]%self.input_node][int(randpos[0]/self.input_node)])
        #print("A")
        child[0][randpos[0]%self.input_node][int(randpos[0]/self.input_node)]=copy.deepcopy(child[0][randpos[1]%self.input_node][int(randpos[1]/self.input_node)])
        #print("A")
        child[0][randpos[1]%self.input_node][int(randpos[1]/self.input_node)]=tem
        #print("A")
        return child

    def mutation2(self,child):
        #print("B")
        randpos=np.random.choice(self.output_node*self.hidden_node,2,replace=False)
        #print("B")
        tem=copy.deepcopy(child[2][randpos[0]%self.hidden_node][int(randpos[0]/self.hidden_node)])
        #print("B")
        child[2][randpos[0]%self.hidden_node][int(randpos[0]/self.hidden_node)]=copy.deepcopy(child[2][randpos[1]%self.hidden_node][int(randpos[1]/self.hidden_node)])
        #print("B")
        child[2][randpos[1]%self.hidden_node][int(randpos[1]/self.hidden_node)]=tem
        #print("B")
        return child

    def crossover(self,parent1,parent2,cross_prob):
        length1=int(len(parent1[0]) * cross_prob)
        length2 = int(len(parent1[1]) * cross_prob)
        parent1[0][:length1]=parent2[0][:length1]
        parent1[2][:length2]=parent2[2][:length2]

        return parent1