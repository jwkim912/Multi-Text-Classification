import numpy as np
import tensorflow as tf
from tensorflow_addons.optimizers import RectifiedAdam
from transformers import TFBertForSequenceClassification, BertTokenizer

BEST_MODEL_NAME = './model/best_model.h5'
tokenizer = BertTokenizer.from_pretrained('./model')
# 입력 데이터(문장) 길이 제한
MAX_SEQ_LEN = 256


def convert_data(s):
    # BERT 입력으로 들어가는 token, mask, segment, target 저장용 리스트
    tokens, masks, segments, targets = [], [], [], []

    # token: 입력 문장 토큰화
    token = tokenizer.encode(s, truncation=True, padding='max_length', max_length=MAX_SEQ_LEN)

    # Mask: 토큰화한 문장 내 패딩이 아닌 경우 1, 패딩인 경우 0으로 초기화
    num_zeros = token.count(0)
    mask = [1] * (MAX_SEQ_LEN - num_zeros) + [0] * num_zeros

    # segment: 문장 전후관계 구분: 오직 한 문장이므로 모두 0으로 초기화
    segment = [0] * MAX_SEQ_LEN

    tokens.append(token)
    masks.append(mask)
    segments.append(segment)

    # numpy array로 저장
    tokens = np.array(tokens)
    masks = np.array(masks)
    segments = np.array(segments)

    return [tokens, masks, segments]


tf.keras.optimizers.RectifiedAdam = RectifiedAdam

s = '나는 요새 너무 힘들어요'
input = convert_data(s)

# 최고 성능의 모델 불러오기
sentiment_model_best = tf.keras.models.load_model(BEST_MODEL_NAME,
                                                  custom_objects={
                                                      'TFBertForSequenceClassification': TFBertForSequenceClassification})

# 모델이 예측한 라벨 도출
predicted_value = sentiment_model_best.predict(input)
predicted_label = np.argmax(predicted_value, axis=1)
