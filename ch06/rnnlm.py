# coding: utf-8
import sys
sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        """
        vocab_size: 어휘 크기 (단어 집합의 크기, V).
        wordvec_size: 단어 임베딩 벡터의 크기 (D).
        hidden_size: LSTM 은닉 상태의 크기 (H).
        """
        rn = np.random.randn

        # 가중치 초기화
        embed_W = (rn(V, D) / 100).astype('f')  # embed_W: 단어 임베딩 층의 가중치. 값을 (V, D) 형태로 생성하고 작은 값으로 초기화합니다.
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        """
        lstm_Wx, lstm_Wh, lstm_b: LSTM 층의 입력 가중치, 은닉 상태 가중치, 바이어스.
        np.sqrt(D) 또는 np.sqrt(H)로 정규화해 폭발적인 기울기를 방지합니다.
        """
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        """
        affine_W, affine_b: 마지막 출력층의 가중치 및 바이어스.
        """

        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        """
        TimeEmbedding: 시계열 데이터에서 단어를 임베딩 벡터로 변환.
        TimeLSTM: LSTM 층을 사용하여 시계열 처리.
        TimeAffine: 완전 연결층으로 LSTM 출력을 어휘 크기 V에 맞게 변환.
        """
        self.loss_layer = TimeSoftmaxWithLoss() #TimeSoftmaxWithLoss: 소프트맥스 손실 계산.
        self.lstm_layer = self.layers[1]

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []    #self.params, self.grads는 각 층의 가중치와 기울기를 모아 효율적인 학습에 사용됩니다.
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):  #xs(입력 데이터)를 순차적으로 각 층에 통과시켜 출력 점수(softmax 이전의 raw scores)를 계산합니다.
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):  #입력 데이터 xs를 예측(predict) 메서드로 처리하고, 손실 층(TimeSoftmaxWithLoss)을 통해 손실 값을 계산합니다.
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1): #손실로부터 시작해 각 층의 기울기를 역순으로 계산합니다. 이를 통해 가중치가 갱신될 수 있도록 합니다.
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()
    """
    LSTM의 상태를 초기화합니다.
    상태유지형 RNN의 경우, 한 문장의 학습이 끝날 때 상태를 초기화해야 합니다.
    self.lstm_layer.reset_state()를 호출해 이를 수행합니다.
    """