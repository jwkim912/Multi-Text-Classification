import tensorflow as tf
from transformers import TFBertForSequenceClassification

BEST_MODEL_NAME = './model/best_model.h5'
# 입력 데이터(문장) 길이 제한

saved_model_dir = './saved_model'

# 최고 성능의 모델 불러오기
sentiment_model_best = tf.keras.models.load_model(BEST_MODEL_NAME,
                                                  custom_objects={
                                                      'TFBertForSequenceClassification': TFBertForSequenceClassification})

# TF 서빙 모델로 변환
sentiment_model_best.save('./saved_model', save_format='tf')

# TFlite 모델로 변환
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open('tf/sentence_all/model.tflite', 'wb').write(tflite_model)