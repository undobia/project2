# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *  # import numpy as np
from common.base_model import BaseModel


class BetterRnnlm(BaseModel):
    '''
     LSTM 계층을 2개 사용하고 각 층에 드롭아웃을 적용한 모델이다.
     아래 [1]에서 제안한 모델을 기초로 하였고, [2]와 [3]의 가중치 공유(weight tying)를 적용했다.

     [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
     [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
     [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
    '''
    '''
    2개 LSTM 계층:
    은닉층을 깊게 구성하여 더 복잡한 패턴 학습이 가능하도록 설계되었습니다.
    
    드롭아웃 (Dropout):
    드롭아웃을 각 LSTM 계층과 임베딩 후에 적용하여 과적합을 방지합니다. ([1] 논문 참고)
    
    가중치 공유 (Weight Tying):
    임베딩 가중치(embed_W)를 출력 계층의 가중치로 재사용합니다. ([2], [3] 논문 참고)
    이를 통해 모델의 학습 효율을 높이고, 학습 파라미터 수를 줄입니다.
    
    상태 유지형 RNN (Stateful RNN):
    TimeLSTM 계층에서 상태를 유지하도록 설정(stateful=True)하여 긴 시계열 데이터를 처리할 수 있습니다.
    '''
    def __init__(self, vocab_size=10000, wordvec_size=650,
                 hidden_size=650, dropout_ratio=0.5):
        """
        :param vocab_size: 어휘 크기.
        :param wordvec_size: 단어 임베딩 벡터 크기.
        :param hidden_size: LSTM 은닉 상태 크기.
        :param dropout_ratio: 드롭아웃 비율.
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')

        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')

        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')

        affine_b = np.zeros(V).astype('f')
        '''
        embed_W: 단어 임베딩 가중치.
        lstm_Wx1, lstm_Wh1, lstm_b1: 첫 번째 LSTM 계층 가중치 및 바이어스.
        lstm_Wx2, lstm_Wh2, lstm_b2: 두 번째 LSTM 계층 가중치 및 바이어스.
        affine_b: 출력 계층 바이어스.
        '''
        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)  # weight tying!!
        ]
        """
        TimeEmbedding: 입력 시퀀스를 임베딩 벡터로 변환.
        TimeLSTM: 두 LSTM 계층으로 시계열 처리.
        TimeDropout: 각 LSTM 계층 후 드롭아웃 적용.
        TimeAffine: 가중치 공유를 통해 출력 계층 정의 (embed_W.T 사용).
        """

        self.loss_layer = TimeSoftmaxWithLoss() #TimeSoftmaxWithLoss를 통해 예측과 실제 정답 간 손실 계산.
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]
        #모든 계층의 파라미터와 기울기를 하나의 리스트에 저장해 최적화에 활용.

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    """
    predict(xs, train_flg):
    입력 데이터 xs를 각 계층에 순차적으로 전달하여 출력 점수를 계산합니다.
    train_flg를 통해 학습/예측 모드를 구분하며, 드롭아웃 계층에 이를 반영합니다.
    """


    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss
    """
    forward(xs, ts, train_flg):
    손실 층(TimeSoftmaxWithLoss)을 추가하여 입력 데이터에 대한 손실 값을 반환합니다.
    """


    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    """
    역전파 (backward 메서드)
    손실로부터 시작해 각 계층을 역순으로 처리하며 기울기를 계산합니다.
    """


    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
    """
    상태 초기화 (reset_state 메서드)
    두 LSTM 계층의 상태를 초기화합니다. 이는 새로운 시퀀스를 학습하거나 평가할 때 필요합니다.
    """

    """
    GRU?
    """