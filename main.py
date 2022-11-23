from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import random
import numpy as np
import copy
import sys
from neural import Neural
import time
from threading import Thread

class Rectangle(QGraphicsRectItem):
    def __init__(self,x,y,w,h):
        super().__init__(x,y,w,h)
        self.setBrush(QColor(0,0,0))
        self.setPen(QColor(0,0,0,1))
        #self.setFlag(QGraphicsItem.,True)

        self.tx=x+w/2
        self.ty=y+h/2

class View(QGraphicsView):
    def __init__(self):
        super(View,self).__init__()

        self.scene=QGraphicsScene()
        self.setSceneRect(100,120,400,400)
        self.rect=[]
        for i in range(20):
            self.rect.append([])
            for j in range(20):
                self.rect[i].append(Rectangle(j*32,i*32,30,30))
                self.scene.addItem(self.rect[i][j])

        self.setScene(self.scene)


class MyApp(QWidget):
    def __init__(self):
        super().__init__()
        self.ne = Neural()
        self.initUI()
        self.show()

    def AutokeyPressEvent(self, dir) :

        #print(dir)

        if dir[0]==1 and dir[1]==0:
            self.lose_check()
            self.now[0]= self.now[0]+1
        elif dir[0] == -1 and dir[1] == 0:
            self.lose_check()
            self.now[0]=  self.now[0]-1
        elif dir[0] == 0 and dir[1] == 1:
            self.lose_check()
            self.now[1]= self.now[1]+1
        elif dir[0] == 0 and dir[1] == -1:
            self.lose_check()
            self.now[1]= self.now[1]-1

        self.que.append(copy.deepcopy(self.now))
        if len(self.que)>self.len:
            self.view.rect[self.que[0][0]][self.que[0][1]].setBrush(QColor(0,0,0))
            self.state[self.que[0][0]][self.que[0][1]]=0
            self.que.pop(0)

        self.view.rect[self.now[0]][self.now[1]].setBrush(QColor(0,150,150))

        if self.flag==0:
            self.clear()
        QApplication.processEvents()
        time.sleep(0.01)

    def btn_load_model_thread(self):
        btn=Thread(target=self.btn_load_model)
        btn.start()

    def btn_load_model(self):

        fname1 = np.load('weights1.npy')
        fname2 = np.load('weights2.npy')

        weight0 = fname1[5]
        weight1 = np.zeros(self.ne.hidden_node)
        weight2 = fname2[5]
        weight3 = np.zeros(self.ne.output_node)
        weights = np.array([weight0, weight1, weight2, weight3])

        while(1):
            pred=self.ne.predict_model(weights,self.now,self.state,self.target,self.dir)
            right=np.array(self.dir)@np.array([[0,-1],[1,0]])
            left = np.array(self.dir) @ np.array([[0, 1], [-1, 0]])

            if np.argmax(pred) == 0:
                self.dir=right
            elif np.argmax(pred)==1:
                self.dir=left

            self.AutokeyPressEvent(self.dir)


    def initUI(self):
        self.setGeometry(200, 200, 1000, 800)
        self.setWindowTitle('snake')
        self.view=View()

        #frame 분할
        main_layout=QVBoxLayout()

        frame_1=QFrame()
        frame_1.setFrameShape(QFrame.Panel|QFrame.Sunken)
        frame_2=QFrame()
        frame_2.setFrameShape(QFrame.Panel|QFrame.Sunken)

        vbox_1=QVBoxLayout()
        self.score_label=QLabel(f"Score\n0")
        self.score_label.setFont(QFont("맑은고딕",40))
        self.score_label.setStyleSheet("Color: green")

        self.remain_label = QLabel(f"남은 움직임\n200")
        self.remain_label.setFont(QFont("맑은고딕", 20))
        self.remain_label.setStyleSheet("Color: black")

        vbox_1.addWidget(self.score_label)
        vbox_1.addWidget(self.remain_label)

        btnTest=QPushButton("test")
        btnLoad=QPushButton("load model")
        btnTest.clicked.connect(self.btn_clicked)
        btnLoad.clicked.connect(self.btn_load_model_thread)

        vbox_1.addWidget(btnTest)
        vbox_1.addWidget(btnLoad)

        vbox_2=QVBoxLayout()
        vbox_2.addWidget(self.view)

        frame_1.setLayout(vbox_1)
        frame_2.setLayout(vbox_2)

        splitter_1=QSplitter(Qt.Horizontal)
        splitter_1.setStretchFactor(0,1)
        splitter_1.addWidget(frame_1)
        splitter_1.addWidget(frame_2)
        splitter_1.setSizes([200, 800])

        main_layout.addWidget(splitter_1)

        self.setLayout(main_layout)
        self.clear()
        self.show()

    def paintRect(self, e):
        qp=QPainter()
        qp.begin(self)
        qp.setBrush(QColor(250,50,0))
        qp.setPen(QColor(0,0,0,0))
        self.rect=Rectangle(150,150,100,100)
        self.draw_rect(qp)
        qp.end()

    def keyPressEvent(self, e) :
        if e.key()==Qt.Key_W:
            self.dir[0], self.dir[1] = -1, 0
            self.lose_check()
            self.now[0]= (0 if self.now[0]-1<0 else self.now[0]-1)
        elif e.key()==Qt.Key_S:
            self.dir[0], self.dir[1] = 1, 0
            self.lose_check()
            self.now[0]= (19 if self.now[0]+1>=20 else self.now[0]+1)
        elif e.key()==Qt.Key_A:
            self.dir[0], self.dir[1] = 0, -1
            self.lose_check()
            self.now[1]= (0 if self.now[1]-1<0 else self.now[1]-1)
        elif e.key()==Qt.Key_D:
            self.dir[0], self.dir[1] = 0, 1
            self.lose_check()
            self.now[1]= (19 if self.now[1]+1>=20 else self.now[1]+1)

        self.que.append(copy.deepcopy(self.now))
        if len(self.que)>self.len:
            self.view.rect[self.que[0][0]][self.que[0][1]].setBrush(QColor(0,0,0))
            self.state[self.que[0][0]][self.que[0][1]]=0
            self.que.pop(0)

        self.view.rect[self.now[0]][self.now[1]].setBrush(QColor(0,150,150))

        if self.flag==0:
            self.clear()




    def btn_clicked(self):
        print(self.state)
        self.ne.init_make_model(self.que,self.remain_move,self.target,self.len)
        self.ne.genetic_algorithm(20)

    def lose_check(self):
        self.remain_move-=1
        self.remain_label.setText(f"남은 움직임\n{self.remain_move}")

        if self.remain_move==0 : #남은 움직임이 없을 때
            self.lose_game()

        elif self.now[0]+self.dir[0]<0 or self.now[0]+self.dir[0]>=20 or self.now[1]+self.dir[1]<0 or self.now[1]+self.dir[1]>=20: #맵밖으로 나감
            self.lose_game()

        elif self.state[self.now[0]+self.dir[0]][self.now[1]+self.dir[1]]==1: # 자기자신에 부딪힘
            self.lose_game()

        elif self.now[0]+self.dir[0] == self.target[0] and self.now[1]+self.dir[1] == self.target[1]: #target에 도착
            self.len+=1
            self.remain_move=200
            self.score_update(100)
            self.state[self.now[0] + self.dir[0]][self.now[1] + self.dir[1]] = 1
            while self.state[self.target[0]][self.target[1]] != 0:
                self.target = [random.randrange(0, 20), random.randrange(0, 20)]

            self.view.rect[self.target[0]][self.target[1]].setBrush(QColor(150, 150, 0))


        else :
            self.state[self.now[0]+self.dir[0]][self.now[1]+self.dir[1]] = 1

    def lose_game(self):
        #self.flag = 1
        self.score_update(0)
        reply = QMessageBox.question(self, '패배', f'score : {self.score}', QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            print('a')
            exit(1)
        else:
            print('b')
            exit(1)
        self.remain_label.setText("남은 움직임\n300")
        self.score_label.setText('Score\n0')

    def clear(self):
        self.now=[random.randrange(0,20),random.randrange(0,20)]
        self.que=[[0,0]]
        self.len=2
        self.dir=[1,0]
        self.state=[]
        self.flag=1
        self.score=0
        self.remain_move=200
        self.target=[random.randrange(0,20),random.randrange(0,20)]

        for i in range(20):
            self.state.append([])
            for j in range(20):
                self.state[i].append(0)
                self.view.rect[i][j].setBrush(QColor(0, 0, 0))

        self.state[self.now[0]][self.now[1]]=1

        self.que.append(copy.deepcopy(self.now))

        while self.state[self.target[0]][self.target[1]] != 0:
            self.target = [random.randrange(0, 20), random.randrange(0, 20)]

        self.view.rect[self.target[0]][self.target[1]].setBrush(QColor(150,150,0))
        self.view.rect[self.now[0]][self.now[1]].setBrush(QColor(0, 150, 150))


    def score_update(self,sc):
        self.score+=sc
        self.score_label.setText(f"Score\n{self.score}")
    '''
    def update(self,i,j,color):
        Thread(target=self.update1(i,j,color)).start()
    def update1(self,i,j,color):
        
        for i in range(20):
            for j in range(20):
                if self.state[i][j]==0:
                    self.view.rect[i][j].setBrush(QColor(0,0,0))

        for i in range(20):
            for j in range(20):
                if self.target[0]==i and self.target[1]==j:
                    self.view.rect[i][j].setBrush(QColor(150, 150, 0))
                elif self.state[i][j]==1 or (self.now[0]==i and self.now[1]==j):
                    self.view.rect[i][j].setBrush(QColor(0,150,150))

        
        if color=='red':
            self.view.rect[i][j].setBrush(QColor(150,150,0))
        elif color=='black':
            self.view.rect[i][j].setBrush(QColor(0,0,0))
        elif color=='blue':
            self.view.rect[i][j].setBrush(QColor(0,150,150))

        self.score_label.setText(f"Score\n{self.score}")
        self.remain_label.setText(f"남은 움직임\n{self.remain_move}")
    '''
app= QApplication(sys.argv)
win=MyApp()
sys.exit(app.exec_())