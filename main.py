from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import random
import copy
import sys

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
        self.initUI()
        self.show()

    def initUI(self):
        self.setGeometry(200, 200, 1000, 800)
        self.setWindowTitle('snake')
        self.view=View()
        #self.sld = QSlider(Qt.Horizontal,self)
        #self.sld.setRange(-180,180)

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
        vbox_1.addWidget(self.score_label)
        #vbox_1.addWidget(QPushButton())

        vbox_2=QVBoxLayout()
        vbox_2.addWidget(self.view)
        #vbox_2.addWidget(self.sld)

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

    def lose_check(self):
        if self.now[0]+self.dir[0]<0 or self.now[0]+self.dir[0]>=20 or self.now[1]+self.dir[1]<0 or self.now[1]+self.dir[1]>=20: #맵밖으로 나감
            self.lose_game()

        elif self.state[self.now[0]+self.dir[0]][self.now[1]+self.dir[1]]==1: # 자기자신에 부딪힘
            self.lose_game()

        elif self.now[0]+self.dir[0] == self.target[0] and self.now[1]+self.dir[1] == self.target[1]: #target에 도착
            self.len+=1
            self.score_update(100)
            self.state[self.now[0] + self.dir[0]][self.now[1] + self.dir[1]] = 1
            while self.state[self.target[0]][self.target[1]] != 0:
                self.target = [random.randrange(0, 20), random.randrange(0, 20)]

            self.view.rect[self.target[0]][self.target[1]].setBrush(QColor(150, 150, 0))

        else :
            self.state[self.now[0]+self.dir[0]][self.now[1]+self.dir[1]] = 1

    def lose_game(self):
        self.flag = 1
        self.score_update(0)
        reply = QMessageBox.question(self, '패배했습니다', '재도전하시겠습니까?', QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            self.flag = 0
        self.score_label.setText('Score\n0')

    def clear(self):
        self.now=[random.randrange(0,20),random.randrange(0,20)]
        self.que=[[0,0]]
        self.len=5
        self.dir=[0,0]
        self.state=[]
        self.flag=1
        self.score=0
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

app= QApplication(sys.argv)
win=MyApp()
sys.exit(app.exec_())