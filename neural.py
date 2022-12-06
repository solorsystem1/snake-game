import tensorflow as tf
import numpy as np
import random
import copy
import datetime
import time

class Neural():
    def __init__(self):
        self.K = 120
        self.N = 15
        self.input_node = 6
        self.hidden_node1 = 36
        self.hidden_node2 = 30
        self.output_node = 3
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(self.input_node),
            tf.keras.layers.Dense(self.hidden_node1, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.hidden_node2, activation=tf.nn.relu),
            tf.keras.layers.Dense(self.output_node, activation=tf.nn.softmax)
        ])

    def fitting_function(self, weights):  # 가상으로 게임을 한번 돌려서 score값 구함
        # 처음 생성
        init_remain_move =100
        now = [random.randrange(0, 20), random.randrange(0, 20)]
        que = [[0, 0]]
        length = 80
        dir = np.array([1, 0])
        state = []
        score = 0
        remain_move = init_remain_move
        additional = 0
        target = [random.randrange(0, 20), random.randrange(0, 20)]

        for i in range(20):
            state.append([])
            for j in range(20):
                state[i].append(0)

        state[now[0]][now[1]] = 1
        que.append(copy.deepcopy(now))

        now_block=[random.randrange(0,20),random.randrange(0,20)]

        state[0][10]=1
        state[0][19]=1
        while state[target[0]][target[1]] != 0:
            target = [random.randrange(0, 20), random.randrange(0, 20)]


        score2=0

        # 반복
        while 1:
            # 모델 가져오기

            pred = self.predict_model(weights, now, state, target, dir, remain_move)[0]

            if np.argmax(pred) == 1:
                dir = np.array(dir)
            elif np.argmax(pred) == 0:
                dir = np.array(dir) @ np.array([[0, -1], [1, 0]])
            elif np.argmax(pred) == 2:
                dir = np.array(dir) @ np.array([[0, 1], [-1, 0]])

            if dir[0] * (target[0] - now[0]) < 0 or dir[1] * (target[1] - now[1]) < 0:
                additional -= 10

            else:
                additional += 5


            # print(np.argmax(pred))
            remain_move -= 1
            score2+=1
            if remain_move == 0:  # 남은 움직임이 없을 때
                #return max(score + additional, 0.0001)
                return score2+(2**score+score**2.1*500)-(score**1.2*(0.25*score2)**1.3)

            elif now[0] + dir[0] < 0 or now[0] + dir[0] >= 20 or now[1] + dir[1] < 0 or \
                    now[1] + dir[1] >= 20:  # 맵밖으로 나감
                #return max(score + additional, 0.0001)
                return score2+(2**score+score**2.1*500)-(score**1.2*(0.25*score2)**1.3)
            elif state[now[0] + dir[0]][now[1] + dir[1]] == 1:  # 자기자신에 부딪힘
                #return max(score + additional, 0.0001)
                return score2+(2**score+score**2.1*500)-(score**1.2*(0.25*score2)**1.3)
            elif now[0] + dir[0] == target[0] and now[1] + dir[1] == target[1]:  # target에 도착
                additional=0
                length += 1
                remain_move = init_remain_move
                score += 1
                state[now[0] + dir[0]][now[1] + dir[1]] = 1

                while state[target[0]][target[1]] != 0:
                    target = [random.randrange(0, 20), random.randrange(0, 20)]

            else:
                state[now[0] + dir[0]][now[1] + dir[1]] = 1

            now[0] = now[0] + dir[0]
            now[1] = now[1] + dir[1]
            que.append(copy.deepcopy(now))

            if len(que) > length:
                state[que[0][0]][que[0][1]] = 0
                que.pop(0)

    def refine_input(self, now, state, target, dir, remain):
        # 위 아래 오른쪽 왼쪽 ->장애물 5칸씩 확인 0.2,0.4,0.6,0.8,1 (4개 node)
        # target 위치 사분면으로 표시 0,1값들 (4개 node)

        ary = np.zeros(self.input_node)
        right = dir @ np.array([[0, -1], [1, 0]])
        left = dir @ np.array([[0, 1], [-1, 0]])

        # print(state[now[0]+5*dir[0]][now[1]+5*dir[1]])
        for i in range(1, 11):
            if now[1] + i * dir[1] >= 20 or now[1] + i * dir[1] < 0 or now[0] + i * dir[0] >= 20 or now[0] + i * dir[0] < 0:
                ary[0] = 1 - 0.1 * (1 - i)
                break
            elif state[now[0] + i * dir[0]][now[1] + i * dir[1]] == 1:
                ary[0] = 1 - 0.1 * (1 - i)
                break

        for i in range(1, 11):
            if now[1] + i * right[1] >= 20 or now[1] + i * right[1] < 0 or now[0] + i * right[0] >= 20 or now[0] + i * \
                    right[0] < 0:
                ary[1] = 1 - 0.1 * (1 - i)
                break
            elif state[now[0] + i * right[0]][now[1] + i * right[1]] == 1:
                    ary[1] = 1 - 0.1 * (1 - i)
                    break
        for i in range(1, 11):
            if now[1] + i * left[1] >= 20 or now[1] + i * left[1] < 0 or now[0] + i * left[0] >= 20 or now[0] + i * \
                    left[0] < 0:
                ary[2] = 1 - 0.1 * (1 - i)
                break
            elif state[now[0] + i * left[0]][now[1] + i * left[1]] == 1:
                ary[2] = 1 - 0.1 * (1 - i)
                break

        if (np.array(target) - np.array(now)).dot(right) > 0:
            ary[3] = 1
        if (np.array(target) - np.array(now)).dot(left) > 0:
            ary[4] = 1
        if (np.array(target) - np.array(now)).dot(right) == 0 and (np.array(target) - np.array(now)).dot(dir) > 0:
            ary[5] = 1

        return ary

    def refine_input1(self, now, state, target, dir, remain):
        # 위 아래 오른쪽 왼쪽 ->장애물 5칸씩 확인 0.2,0.4,0.6,0.8,1 (4개 node)
        # target 위치 사분면으로 표시 0,1값들 (4개 node)

        ary = np.zeros(self.input_node)
        right = dir @ np.array([[0, -1], [1, 0]])
        left = dir @ np.array([[0, 1], [-1, 0]])

        if now[1] + dir[1] >= 20 or now[1] + dir[1] < 0 or now[0] + dir[0] >= 20 or now[0] + dir[0] < 0:
            ary[0] = 1
        elif state[now[0] + dir[0]][now[1] + dir[1]] == 1:
            ary[0] = 1

        if now[1] + right[1] >= 20 or now[1] + right[1] < 0 or now[0] + right[0] >= 20 or now[0] + right[0] < 0:
            ary[1] = 1
        elif state[now[0] + right[0]][now[1] + right[1]] == 1:
            ary[1] = 1

        if now[1] + left[1] >= 20 or now[1] + left[1] < 0 or now[0] + left[0] >= 20 or now[0] +  left[0] < 0:
            ary[2] = 1
        elif state[now[0] + left[0]][now[1] + left[1]] == 1:
            ary[2] = 1

        if (np.array(target) - np.array(now)).dot(right) > 0:
            ary[3] = 1
        if (np.array(target) - np.array(now)).dot(left) > 0:
            ary[4] = 1
        if (np.array(target) - np.array(now)).dot(right) == 0 and (np.array(target) - np.array(now)).dot(dir) > 0:
            ary[5] = 1

        # ary[12]=remain/200
        return ary

    def refine_input2(self, now, state, target, dir, remain):
        # 위 아래 오른쪽 왼쪽 ->장애물 5칸씩 확인 0.2,0.4,0.6,0.8,1 (4개 node)
        # target 위치 사분면으로 표시 0,1값들 (4개 node)

        ary = np.zeros(self.input_node)
        right = dir @ np.array([[0, -1], [1, 0]])
        left = dir @ np.array([[0, 1], [-1, 0]])

        # print(state[now[0]+5*dir[0]][now[1]+5*dir[1]])

        for i in range(1, 4):
            if now[1] + i * dir[1] >= 20 or now[1] + i * dir[1] < 0 or now[0] + i * dir[0] >= 20 or now[0] + i * dir[
                0] < 0:
                ary[i - 1] = 1
            elif state[now[0] + i * dir[0]][now[1] + i * dir[1]] == 1:
                ary[i - 1] = 1

        for i in range(1, 4):
            if now[1] + i * right[1] >= 20 or now[1] + i * right[1] < 0 or now[0] + i * right[0] >= 20 or now[0] + i * \
                    right[0] < 0:
                ary[i + 2] = 1
            elif state[now[0] + i * right[0]][now[1] + i * right[1]] == 1:
                ary[i + 2] = 1

        for i in range(1, 4):
            if now[1] + i * left[1] >= 20 or now[1] + i * left[1] < 0 or now[0] + i * left[0] >= 20 or now[0] + i * \
                    left[0] < 0:
                ary[i + 5] = 1
            elif state[now[0] + i * left[0]][now[1] + i * left[1]] == 1:
                ary[i + 5] = 1

        if (np.array(target) - np.array(now)).dot(right) > 0:
            ary[9] = 1
        if (np.array(target) - np.array(now)).dot(left) > 0:
            ary[10] = 1
        if (np.array(target) - np.array(now)).dot(right) == 0 and (np.array(target) - np.array(now)).dot(dir) > 0:
            ary[11] = 1

        # ary[12]=remain/200
        return ary

    def init_make_model(self):
        weight0 = np.random.randn(self.input_node, self.hidden_node1)*0.1
        weight1 = np.zeros(self.hidden_node1)
        weight2 = np.random.randn(self.hidden_node1, self.hidden_node2)*0.1
        weight3 = np.zeros(self.hidden_node2)
        weight4 = np.random.randn(self.hidden_node2, self.output_node) * 0.1
        weight5 = np.zeros(self.output_node)
        weights = np.array([weight0, weight1, weight2, weight3, weight4, weight5])
        return weights

    def predict_model(self, weights, now, state, target, dir, remain):
        self.model.set_weights(weights)
        input_data = self.refine_input1(now, state, target, dir, remain)
        input_data = input_data.reshape(-1, self.input_node)
        pred = self.model.predict(input_data, verbose=0)
        return pred

    def genetic_algorithm(self, step, flag, we_list,we1,we2,we3):
        weights_list = []
        weights_by_step1 = []
        weights_by_step2 = []
        weights_by_step3 = []

        if flag==0:
            for i in range(self.K):
                weights_list.append(self.init_make_model())

        else :
            for i in range(len(we1)):
                weights_by_step1.append(we1[i])
            for i in range(len(we2)):
                weights_by_step2.append(we2[i])
            for i in range(len(we3)):
                weights_by_step3.append(we3[i])
            for i in range(len(we_list)):
                weights_list.append(we_list[i])

        for k in range(step):
            sum_fit = 0
            fit_list = []
            parent_list = []
            for i in range(self.K):  # fit 누적값 저장
                fit_val = self.fitting_function(weights_list[i])
                print(f"step {k} : [{i}번째 부모 test]  fit_val : {fit_val}")
                sum_fit += fit_val
                fit_list.append(fit_val)

            print(max(fit_list))

            weights_by_step1.append(weights_list[np.argmax(fit_list)][0])
            weights_by_step2.append(weights_list[np.argmax(fit_list)][2])
            weights_by_step3.append(weights_list[np.argmax(fit_list)][4])

            np.argmax(fit_list)
            if sum_fit != 0:
                for i in range(self.K):
                    fit_list[i] = fit_list[i] / sum_fit

            # 상위 N개 자식 생성
            selected_list = []
            selected_list.append(np.argmax(fit_list))
            parent_list.append(copy.deepcopy(weights_list[np.argmax(fit_list)]))
            m = 0
            random.seed(datetime.datetime.now())

            while (m < self.N - 1):
                parent_idx = random.choices(range(0, self.K), weights=fit_list)
                if parent_idx[0] not in selected_list:
                    parent_list.append(copy.deepcopy(weights_list[parent_idx[0]]))
                    selected_list.append(parent_idx[0])
                    m += 1

            cross_prob = 0.8
            mutate_prob = 0.03
            child_list = []

            # 2개의 자식 선택 -> 교차
            for i in range(self.N):
                for j in range(i,self.N):
                    parent_1 = parent_list[i]
                    parent_2 = parent_list[j]
                    child = self.crossover(parent_1, parent_2, cross_prob)
                    if np.random.random() < mutate_prob:
                        child = self.mutation1(child)
                    if np.random.random() < mutate_prob:
                        child = self.mutation2(child)
                    if np.random.random() < mutate_prob:
                        child = self.mutation3(child)
                    child_list.append(child)

            weights_list = copy.deepcopy(child_list)
            np.save('weights1', weights_by_step1)
            np.save('weights2', weights_by_step2)
            np.save('weights3', weights_by_step3)
            np.save('last', weights_list)
            print(f"step : {k}-----------------------")

        print('end')

        return weights_list

    def mutation1(self, child):
        rand_pos_x=random.randrange(0,len(child[0]))
        rand_pos_y=random.randrange(0,len(child[0][0]))
        child[0][rand_pos_x][rand_pos_y]=np.random.uniform(0,1)
        return child

    def mutation2(self, child):
        rand_pos_x=random.randrange(0,len(child[2]))
        rand_pos_y = random.randrange(0, len(child[2][0]))
        child[2][rand_pos_x][rand_pos_y]=np.random.uniform(0,1)
        return child

    def mutation3(self, child):
        rand_pos_x=random.randrange(0,len(child[4]))
        rand_pos_y = random.randrange(0, len(child[4][0]))
        child[4][rand_pos_x][rand_pos_y]=np.random.uniform(0,1)
        return child

    def crossover1(self, parent1, parent2):  # uniform crossover
        for i in range(len(parent1[0])):
            if random.randint(0, 1) == 0:
                parent1[0][i] = parent2[0][i]

        for i in range(len(parent1[2])):
            if random.randint(0, 1) == 0:
                parent1[2][i] = parent2[2][i]

        return parent1

    def crossover(self, parent1, parent2, alpha):  # whole arithmetic recombination

        child = copy.deepcopy(parent1)

        child[0] = alpha * parent1[0] + (1 - alpha) * parent2[0]
        child[2] = alpha * parent1[2] + (1 - alpha) * parent2[2]
        child[4] = alpha * parent1[4] + (1 - alpha) * parent2[4]
        return child
