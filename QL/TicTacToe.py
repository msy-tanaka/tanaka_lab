#!/usr/bin/env python
# coding: utf-8

# # 三目並べゲーム
# ## 環境

# In[1]:


import pygame
import random
import sys

class TicTacToeEnv:
    def __init__(self, size = 3, player_turn=random.choice([1, -1])):
        self.size = size 
        self.board = [[0 for _ in range(size)] for _ in range(size)]  # サイズをもとにボードを作成
        self.player_turn = player_turn # プレイヤーのターン
        self.winner = None  # 勝者
        self.invalid_move = None  # 無効な動作
        
    def game_reset(self):
        """ボードの初期化"""
        self.board = [[0 for _ in range(self.size)] for _ in range(self.size)] 
        self.player_turn = random.choice([1, -1])
        self.winner = None
        self.invalid_move = None
    
    def done(self):
        """終了判定"""
        # 各行と列をチェック
        for i in range(self.size):
            # 行をチェック
            if self.board[i].count(self.player_turn) == self.size:
                self.winner = self.player_turn
                return True
            # 列をチェック
            column = [self.board[j][i] for j in range(self.size)]
            if column.count(self.player_turn) == self.size:
                self.winner = self.player_turn
                return True

        # 主対角線をチェック
        if all(self.board[i][i] == self.player_turn for i in range(self.size)):
            self.winner = self.player_turn
            return True

        # 副対角線をチェック
        if all(self.board[i][self.size - 1 - i] == self.player_turn for i in range(self.size)):
            self.winner = self.player_turn
            return True
        
        # ドローのチェック
        if all(self.board[i][j] != 0 for i in range(self.size) for j in range(self.size)):
            self.winner = 0  # ドローの時は勝者を0とする
            return True

        return False

    def step(self, act):
        """状態を更新"""
        x, y = divmod(act, self.size)
        # actを受け取って、次の状態にする
        if self.board[x][y] == 0:  # 受け取ったactが有効なら
            self.board[x][y] = self.player_turn
            self.invalid_move = None
            done = self.done()
            # プレイヤー交代
            self.player_turn *= -1
            return done
        else:
            self.invalid_move = (x, y)
            return None
    
    def pygame_init(self):
        """pygame開始"""
        pygame.init()
        self.screen = pygame.display.set_mode((self.size * 100, self.size * 100))
        self.font = pygame.font.Font(None, 100)
        pygame.display.set_caption("Tic Tac Toe　Game")
        self.pygame_render()

    def pygame_render(self):
        """描画関数"""
        # pygameでboardの内容を描画する
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        RED = (255, 0, 0, 150)
        self.screen.fill(WHITE)
        for x in range(1, self.size):
            pygame.draw.line(self.screen, BLACK, (x * 100, 0), (x * 100, self.size * 100), 3)
            pygame.draw.line(self.screen, BLACK, (0, x * 100), (self.size * 100, x * 100), 3)
            
        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == 1:
                    text = self.font.render('O', True, BLACK)
                    self.screen.blit(text, (j * 100 + 25, i * 100 + 15))
                elif self.board[i][j] == -1:
                    text = self.font.render('X', True, BLACK)
                    self.screen.blit(text, (j * 100 + 25, i * 100 + 15))
            
        if self.invalid_move is not None: 
            (i, j) = self.invalid_move
            pygame.draw.rect(self.screen, RED, (j * 100, i * 100, 100, 100), 3)
        
        pygame.display.flip()


# In[2]:


def progress(agent1, agent2, size=3, render=True, nplay=-1):
    """ゲーム進行"""
    env = TicTacToeEnv(size)
    
    # 勝敗のカウント
    wins_O, wins_X, draws = 0,0,0
    
    if render:
        env.pygame_init()  # Pygameを初期化し、ゲームウィンドウを設定する
    running = True
    
    while running and sum([wins_O,wins_X,draws])!=nplay:
        if render:
            env.pygame_render()
        # 行動を決める
        if env.player_turn == 1:
            act = agent1.act(env)
        elif env.player_turn == -1:
            act = agent2.act(env)
        
        done = env.step(act)
        
        if render:
            env.pygame_render()
        
        # 終了判定 
        if done:
            if env.winner == 1:
                wins_O += 1
            elif env.winner == -1:
                wins_X += 1
            else:
                draws += 1
            print(f"Winner: {'O   ' if env.winner == 1 else 'X   ' if env.winner == -1 else 'Draw'}", end='\r')
            # 初期化
            env.game_reset()
            
        if render:
            for event in pygame.event.get():  # Pygameのイベントを処理する
                if event.type == pygame.QUIT:  # ウィンドウの閉じるボタンがクリックされたとき
                    running = False  # メインループを終了する
    
    print(f"Final Results: O wins: {wins_O}, X wins: {wins_X}, Draws: {draws}")

    pygame.quit()  # Pygameを終了する


# ## ランダムとランダムαと人間

# In[3]:


class RandomAgent:
    """完全ランダム"""
    def __init__(self, my_turn):
        self.my_turn = my_turn

    def act(self, env):
        possible_acts =  [i * env.size + j for i in range(env.size) for j in range(env.size) if env.board[i][j] == 0]
        act = random.choice(possible_acts)
        return act

class RandomalfaAgent:
    """勝てるところがあれば勝つランダム"""
    def __init__(self, my_turn):
        self.my_turn = my_turn
        
    def act(self, env):
        possible_acts =  [i * env.size + j for i in range(env.size) for j in range(env.size) if env.board[i][j] == 0]
        for act in possible_acts:
            x, y = divmod(act, env.size)
            if env.board[x][y] == 0:
                env.board[x][y] = self.my_turn
                env.done()
                if env.winner == self.my_turn:
                    act = x * env.size + y
                    env.board[x][y] = 0
                    env.winner = None
                    return act
                env.board[x][y] = 0
                    
        act = random.choice(possible_acts)
        return act

class HumanAgent:
    """人がプレイヤーのクラス"""
    def __init__(self, my_turn):
        self.my_turn = my_turn

    def act(self, env):
        while True:
            for event in pygame.event.get():  # Pygameのイベントを処理する
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = pygame.mouse.get_pos()  # マウスの位置を取得する
                    x, y = pos[1] // 100, pos[0] // 100  # マウスの位置をボードのセルに変換する
                    act = x * env.size + y  # 行と列を1次元のインデックスに変換する
                    if env.board[x][y] == 0:
                        return act
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()


# In[4]:


agent1 = RandomalfaAgent(1)
agent2 = RandomalfaAgent(-1)
#progress(agent1,agent2,render=True,nplay=100)


# # モンテカルロ法

# In[5]:


from copy import deepcopy

class MCAgent:
    """モンテカルロ法"""
    def __init__(self, my_turn):
        self.my_turn = my_turn

    def win_or_rand(self, env, turn):
        """次の手で勝てる場合、勝ち、そうでないならランダムに動く"""
        possible_acts =  [i * env.size + j for i in range(env.size) for j in range(env.size) if env.board[i][j] == 0]
        for act in possible_acts:
            x, y = divmod(act, env.size)
            if env.board[x][y] == 0:
                env.board[x][y] = turn
                env.done()
                if env.winner == turn:
                    act = x * env.size + y
                    env.board[x][y] = 0
                    env.winner = None
                    return act
                env.board[x][y] = 0
        
        act = random.choice(possible_acts)
        return act

    def trial(self, scores, env, act):
        tempboard = deepcopy(env.board)
        tempturn = self.my_turn
        done = env.step(act)
        
        while not done:
            tempturn *= -1
            tempact = self.win_or_rand(env,tempturn)
            done = env.step(tempact)
        if env.winner == self.my_turn:
            scores[act] += 1
        elif env.winner == self.my_turn*-1:
            scores[act] += -1
        env.player_turn = self.my_turn
        env.board = tempboard
        env.winner = None
        
    def act(self, env):
        scores={}
        n=50
        possible_acts =  [i * env.size + j for i in range(env.size) for j in range(env.size) if env.board[i][j] == 0]
        for act in possible_acts:
            scores[act]=0
            for i in range(n):
                self.trial(scores,env,act)
            scores[act]/=n
        max_score=max(scores.values())
        for act, v in scores.items():
            if v == max_score:
                return act


# In[6]:


agent1 = MCAgent(1)
agent2 = MCAgent(-1)
#progress(agent1,agent2,size=3,nplay=10)


# # QLarning

# In[7]:


import sys
import pickle
import random
from collections import deque

class QLAgent:
    def __init__(self, my_turn, 
                 gamma=0.9,    # 割引率
                 epsilon=0.2,  # 乱雑度
                 alpha=0.3,    # 学習率
                 memory_size=10000):
        
        # パラメータ
        self.my_turn = my_turn
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_val_Q = 1
        self.memory_size = memory_size
        self.last_obss = deque(maxlen=2)
        self.last_act = None
        
        # Qテーブル関連
        self.Q = {}     # Qテーブル
        self.len_Q = 0  # Qテーブルに登録した観測の数
        self.episode = 0
        
        self.filepath = 'agt_data/tictactoe_QL'

    def act(self, env):
        """観測に対して行動を出力"""
        obs = env.board
        possible_acts = [i * env.size + j for i in range(env.size) for j in range(env.size) if env.board[i][j] == 0]
        
        # obsを文字列に変換
        obs = str(obs)
        
        # obs が登録されていなかったら初期値を与えて登録
        self._check_and_add_observation(obs, possible_acts)
        
        # 確率的に処理を分岐
        if random.random() < (self.epsilon/(self.episode//10000+1)):
            # epsilon の確率
            act = random.choice(possible_acts)  # ランダム行動
        else:
            # 1-epsilon の確率
            acts_q_values = self.Q[obs]
            max_q_value = max(acts_q_values.values()) # 最大のQ値
            # 最大のQ値が複数ある場合の処理
            max_q_acts = [act for act, q_value in acts_q_values.items() if q_value == max_q_value] 
            act = random.choice(max_q_acts)
        return act

    def _check_and_add_observation(self, obs, possible_acts):
        """obs が登録されていなかったら初期値を与えて登録"""
        if obs not in self.Q: 
            self.Q[obs] = {}
            for act in possible_acts:
                self.Q[obs][act] = self.init_val_Q
            self.len_Q += 1
        
        if self.len_Q > self.memory_size:
                print(f'The number of registered observations has reached the limit of {self.max_memory:d}')
                sys.exit()

    def learn(self, rwd, done, env):
        """学習"""
        if rwd is None:  # rwdがNoneだったら戻る
            return
        # obs, next_obs を文字列に変換
        last_obs = str(self.last_obss[0]) 
        obs = str(self.last_obss[1])
        
        possible_acts =  [i * env.size + j for i in range(env.size) for j in range(env.size) if self.last_obss[1][i][j] == 0]
        
        self._check_and_add_observation(obs, possible_acts)
        
        # 学習のターゲットを作成
        if done is True:
            target = rwd
        else:
            target = rwd + self.gamma * max(self.Q[obs].values())
        # Qをターゲットに近づける
        self.Q[last_obs][self.last_act] = (1 - self.alpha) * self.Q[last_obs][self.last_act] + self.alpha * target

    def get_Q(self, obs):
        """観測に対するQ値を出力"""
        obs = str(obs)
        if obs in self.Q:   # obsがQにある
            Q = self.Q[obs]
        else:               # obsがQにない
            Q = None
        return Q
    
    def save_weights(self):
        """方策のパラメータの保存"""
        # Qテーブルの保存
        filepath = self.filepath + '.pkl'
        with open(filepath, mode='wb') as f:
            pickle.dump(self.Q, f)

    def load_weights(self):
        """方策のパラメータの読み込み"""
        # Qテーブルの読み込み
        filepath = self.filepath + '.pkl'
        with open(filepath, mode='rb') as f:
            self.Q = pickle.load(f)


# In[8]:


def trainQL(agent1, agent2, size=3, nplay=100000):
    # 環境の準備
    env = TicTacToeEnv(size)  # TicTacToe環境を作成

    # 勝敗のカウント
    wins_O, wins_X, draws = 0, 0, 0
    episode = 0
  
    running = True
    while running and sum([wins_O, wins_X, draws]) != nplay:
        # 学習の準備
        rwd = 0
        
        # 各playerごとに処理
        if env.player_turn == agent1.my_turn:
            act = agent1.act(env)  # agent1が行動を決定
            agent1.last_obss.append(deepcopy(env.board))  # 現在のボードを保存
            # 状態を更新
            done = env.step(act)  # 環境に行動を適用
            if done:  # ゲームが終了した場合
                if env.winner == 0:
                    rwd = 0  # 引き分け
                elif env.winner == agent1.my_turn:
                    rwd = 1  # agent1の勝利
                elif env.winner == agent2.my_turn:
                    rwd = -1  # agent2の勝利
                agent2.last_obss.append(deepcopy(env.board))  # 最後の状態を保存
                agent2.learn(rwd * -1, done, env)  # agent2に報酬を与え学習させる
            
            # 学習
            if len(agent1.last_obss) == 2:
                agent1.learn(rwd, done, env)  # agent1が学習
            agent1.last_act = act  # 最後の行動を保存
            
        elif env.player_turn == agent2.my_turn:
            act = agent2.act(env)  # agent2が行動を決定
            agent2.last_obss.append(deepcopy(env.board))
            done = env.step(act)
            if done:
                if env.winner == 0:
                    rwd = 0
                elif env.winner == agent2.my_turn:
                    rwd = 1
                elif env.winner == agent1.my_turn:
                    rwd = -1
                agent1.last_obss.append(deepcopy(env.board))
                agent1.learn(rwd * -1, done, env)
            if len(agent2.last_obss) == 2:
                agent2.learn(rwd, done, env)
            agent2.last_act = act
            
        # 終了判定
        if done:
            if env.winner == 1:
                wins_O += 1  # Oの勝利をカウント
            elif env.winner == -1:
                wins_X += 1  # Xの勝利をカウント
            else:
                draws += 1  # 引き分けをカウント
            # 初期化
            env.game_reset()  # ゲームをリセット
            agent1.last_obss.clear()  # agent1の履歴をクリア
            agent2.last_obss.clear()  # agent2の履歴をクリア
            episode += 1
            
            # 10000エピソードごとに進行状況を表示
            if episode % 10000 == 0:
                print(f"episode:{episode}/len_Q:{agent1.len_Q} - O wins: {wins_O}, X wins: {wins_X}, Draws: {draws}")
            done = False
    
    print(f"Final Results: O wins: {wins_O}, X wins: {wins_X}, Draws: {draws}")

    pygame.quit()  # Pygameを終了する
    
    # 重みパラメータの保存
    agent2.save_weights()  # agent2の重みを保存


# In[9]:


agent1 = QLAgent(1)
agent2 = QLAgent(-1)
#trainQL(agent1,agent2)


# In[10]:


agent1 = HumanAgent(1)
agent2 = QLAgent(-1,epsilon=0)
agent2.load_weights()
#progress(agent1,agent2,nplay=-1)


# # DQN

# In[11]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from collections import deque
import random
import csv
import os
import numpy as np

class Memory:
    """経験再生のメモリクラス"""
    def __init__(self, memory_size=100, batch_size=30):
        self.memory_size = memory_size
        self.buffer = deque(maxlen=memory_size)

    def add(self, experience):
        # 右側に経験を追加
        self.buffer.append(experience)

    def sample(self, batch_size):
        # バッチサイズ分の経験をサンプリングする
        idx = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[i] for i in idx]

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, my_turn, size, 
                 gamma = 0.9, # 割引率
                 epsilon = 0.1, # 乱雑度
                 memory_size = 100, # 経験の保存数
                 batch_size = 10, # 学習で使用する経験の数
                 target_interval = 50 # ターゲットを更新する間隔
                ):
        
        # パラメータ
        self.my_turn = my_turn
        self.input_size = (size,size)
        self.n_act = size**2
        self.gamma = gamma
        self.epsilon = epsilon  
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_interval = target_interval
        self.model = self._build_Qnet()
        self.last_obss = deque(maxlen=2)
        self.last_act = None
        self.time = 0
        self.filepath='agt_data/tictaktoe_DQN'
        
        # 学習過程の記録関連
        self.hist_rwds = []
        
        # ターゲットモデルの生成
        self.model_target = self._build_Qnet()
        # メモリのインスタンスを作成
        self.memory = Memory(memory_size=self.memory_size, batch_size=batch_size)

    def _build_Qnet(self):
        # Qネットワークの構築
        model = Sequential()
        model.add(Flatten(input_shape=self.input_size))
        model.add(Dense(162, activation='relu'))
        model.add(Dense(162, activation='relu'))
        model.add(Dense(self.n_act, activation='linear'))
        
        # 勾配法のパラメータの定義
        model.compile(loss='mse', optimizer='Adam')
        
        return model

    def act(self, env):
        obs = env.board
        # 確率でε-greedy法ではない
        if random.random() <= self.epsilon:
            act = random.randrange(self.n_act)
        else:
            # Q値を予測する
            Q = self.get_Q(obs)
            act = Q.index(max(Q))  # 最大となるQ値を出力
        
        return act
    
    def get_Q(self, obs, type='main'):
        # obsを入力し出力を得る
        obs_reshaped = np.array([obs])
        if type == 'main':
            # Qネットワークに観測obsを入力し出力を得る
            Q = self.model.predict(obs_reshaped, verbose=0)[0, :]
        elif type == 'target':
            # ターゲットネットに観測obsを入力し出力を得る
            Q = self.model_target.predict(obs_reshaped, verbose=0)[0, :]

        return Q.tolist()
             

    def learn(self, rwd, done, env):
        if rwd is None:
            return
        
        if env.invalid_move is None:
            last_obs = self.last_obss[0]
            obs = self.last_obss[1]
        else:
            last_obs = self.last_obss[0]
            obs = self.last_obss[0]
        
        self.memory.add((last_obs, self.last_act, rwd, done, obs))
        
        #print(f"last_obs:{last_obs}/obs:{obs}")
        
        # 学習
        self._fit()

        # target_intervalの周期でQネットワークの重みをターゲットネットにコピー
        if self.time % self.target_interval == 0 and self.time > 0:
            self.model_target.set_weights(self.model.get_weights())
        #print(self.time, end='\r')

        self.time += 1
        
    def _fit(self):
        # 記憶された「経験」のデータの量がバッチサイズに満たない場合は戻る
        if len(self.memory) < self.batch_size:
            return
        
        # 学習に使うデータを出力
        outs = self.memory.sample(self.batch_size)

        # 観測とターゲットのバッチを入れる配列を準備
        obs_shape = self.input_size
        obss = np.zeros((self.batch_size,) + obs_shape)
        targets = np.zeros((self.batch_size, self.n_act))
        
        for i, (last_obs, act, rwd, done, obs) in enumerate(outs):
            # obs に対するQネットワークの出力 yを得る
            y = self.get_Q(last_obs)

            # target にyの内容をコピーする
            target = y[:]

            if not done:
                # 最終状態でなかったら next_obsに対する next_yを得る
                next_y = self.get_Q(obs)

                # Q[obs][act]のtarget_act を作成
                target_act = rwd + self.gamma * max(next_y)
            else:
                # 最終状態の場合は報酬だけでtarget_actを作成
                target_act = rwd

            # targetのactの要素だけtarget_actにする
            target[act] = target_act

            # obsとtargetをバッチの配列に入れる
            obss[i] = obs
            targets[i] = target
        
        # obssと targets のバッチのペアを与えて学習
        self.model.fit(obss, targets, verbose=0, epochs=1)

    
    def save_weights(self):
        self.model.save(self.filepath + '.keras', overwrite=True)

    def load_weights(self):
        # モデルの重みを読み込む
        self.model = tf.keras.models.load_model(self.filepath + '.keras')


# In[12]:


def trainDQN(agent1, agent2, size=3, nplay=100, render=True):
    # 環境の準備
    env = TicTacToeEnv(size)  # TicTacToe環境を作成

    # 勝敗のカウント
    wins_O, wins_X, draws = 0, 0, 0
    episode = 0
    
    if render:
        env.pygame_init()  # Pygameを初期化し、ゲームウィンドウを設定する
    
    running = True
    while running and sum([wins_O, wins_X, draws]) != nplay:
        if render:
            env.pygame_render()
        
        # 学習の準備
        rwd = 0
        
        # 各playerごとに処理
        if env.player_turn == agent1.my_turn:
            act = agent1.act(env)  # agent1が行動を決定
            agent1.last_obss.append(deepcopy(env.board))  # 現在のボードを保存
            # 状態を更新
            done = env.step(act)  # 環境に行動を適用
            if done:  # ゲームが終了した場合
                if env.winner == 0:
                    rwd = 0  # 引き分け
                elif env.winner == agent1.my_turn:
                    rwd = 1  # agent1の勝利
                elif env.winner == agent2.my_turn:
                    rwd = -1  # agent2の勝利
            if env.invalid_move != None:
                rwd = -1
            
            # 学習
            if len(agent1.last_obss) == 2:
                agent1.learn(rwd, done, env)  # agent1が学習
            agent1.last_act = act  # 最後の行動を保存
            
        elif env.player_turn == agent2.my_turn:
            act = agent2.act(env)  # agent2が行動を決定
            done = env.step(act)
            if done:
                if env.winner == 0:
                    rwd = 0
                elif env.winner == agent2.my_turn:
                    rwd = 1
                elif env.winner == agent1.my_turn:
                    rwd = -1
                agent1.last_obss.append(deepcopy(env.board))
                agent1.learn(rwd * -1, done, env)
                
        if render:
            env.pygame_render()
            
        # 終了判定
        if done:
            if env.winner == 1:
                wins_O += 1  # Oの勝利をカウント
            elif env.winner == -1:
                wins_X += 1  # Xの勝利をカウント
            else:
                draws += 1  # 引き分けをカウント
            # 初期化
            env.game_reset()  # ゲームをリセット
            agent1.last_obss.clear()  # agent1の履歴をクリア
            episode += 1
            
            # 10000エピソードごとに進行状況を表示
            if episode % 1000 == 0:
                print(f"episode:{episode} - O wins: {wins_O}, X wins: {wins_X}, Draws: {draws}")
            done = False
        
        if render:
            for event in pygame.event.get():  # Pygameのイベントを処理する
                if event.type == pygame.QUIT:  # ウィンドウの閉じるボタンがクリックされたとき
                    running = False  # メインループを終了する
    
    print(f"Final Results: O wins: {wins_O}, X wins: {wins_X}, Draws: {draws}")  
    
    # 重みパラメータの保存
    agent1.save_weights()  # agent1の重みを保存
    pygame.quit()  # Pygameを終了する


# In[13]:


if __name__ == "__main__":
    agent1 = DQNAgent(1,size=3)
    agent2 = RandomalfaAgent(-1)
    agent1.load_weights()
    trainDQN(agent1,agent2,nplay=20000)
    agent2 = QLAgent(-1,epsilon=0)
    agent2.load_weights()
    trainDQN(agent1,agent2,nplay=30000)


# In[14]:


agent1 = DQNAgent(1,size=3)
agent2 = HumanAgent(-1)
#progress(agent1,agent2,nplay=-1)


# In[ ]:




