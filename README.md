## Summary
한국어 텍스트에 대한 6가지 레이블의 감정 분석(multi text classification)
- 테스트용으로 하드코딩 되어있음
- 데이터 미 포함

## Data
- 모델: [klue/bert-base](https://huggingface.co/klue/bert-base)
- 학습 데이터 : [AIHUB 감성 대화 말뭉치](https://aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=86)

## Run

### train & evaluation
- hugging face hub 로 부터 모델, 토크나이저 로드 
```shell
python sentiment_classifier.py
```
- local 에 저장 되어있는 모델, 토크나이저 로드 
```shell
python sentiment_classifier_local.py
```

### convert model
```shell
python convert_model.py
```

### predict
```shell
python predict.py
```

## Reference
- https://velog.io/@jaehyeong/Fine-tuning-Bert-using-Transformers-and-TensorFlow
- https://www.sunnyville.ai/fine-tuning-distilbert-multi-class-text-classification-using-transformers-and-tensorflow/