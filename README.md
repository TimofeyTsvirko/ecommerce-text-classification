py -3.11 -m venv .venv

.venv\Scripts\activate.bat

pip install -r requirements.txt

# Ecommerce Text Classification

## Суть задачи
Нужно классифицировать описание товара (product description) по одной из 4 категорий:

- Electronics  
- Household  
- Books  
- Clothing & Accessories

## EDA

**Формат:** CSV-файл с двумя колонками:
- Название класса (категория)
- Текст (название + описание товара)

**Количество примеров:** 50 425  
**Количество классов:** 4

### Дубликаты

В датасете было найдено 22622 (~45%) дубликатов - все они были удалены, но даже после удаления размер датасета достаточно большой

### Почти полное отсутствие пропусков

Только в одной строке был пропуск, поэтому мы просто удалили данную строку

### Большой хвост в распределении длин токенов

В дальнейшем мы обрезали неинформативную часть и оставляли первые 500 токенов

### Отсутствие влияния удаления стоп-слов и пунктуации

Даже после очистки токенов их последовательность важности (anova-f) почти не изменилась, поэтому скорее всего данная операция не нужна. В дальнейшем это подтверждается экспериментами с моделями.

### Небольшой дисбаланс классов
С распределением классов можно ознакомиться в таблице:
| Category                  | Percentage |
|---------------------------|------------|
| Household                 | 38.0%      |
| Books                     | 22.5%      |
| Clothing & Accessories    | 20.4%      |
| Electronics               | 19.1%      |

Т.к. имеется небольшой перевес в сторону класса Household, поэтому в дальнейшем мы будем опираться на подходящие для дисбаланса метрики и использовать дополнительные веса для обучения моделей.

## Baseline

Использовались разные вариации кодирования последовательностей и очистка:
1. CountVectorizer без очистки
2. CountVectorizer с очисткой
3. CountVectorizer с топ-100 важными токенами
4. TF-IDF без очистки
5. TF-IDF с очисткой

Самые лучшие результаты показала LogisticRegression с первой конфигурацией (CountVectorizer без очистки). Вот сравнительная таблица из данного эксперимента:

| Model              | F1 Score | Precision | Recall | Accuracy | Training Time (s) |
|--------------------|----------|-----------|--------|----------|-------------------|
| LogisticRegression | 0.975    | 0.976     | 0.975  | 0.975    | 117.633           |
| DecisionTree       | 0.612    | 0.829     | 0.581  | 0.644    | 12.712            |
| RandomForest       | 0.969    | 0.973     | 0.966  | 0.969    | 371.328           |
| LGBM               | 0.965    | 0.967     | 0.963  | 0.964    | 42.060            |

Это очень сильный результат для базовой модели - посмотрим, получится ли его улучшить с использованием нейросетей.

## Нейросети

### Criterion

Использовался CrossEntropyLoss с весами [0.6579, 1.1110, 1.2250, 1.3094]

### Optimizer

Использовался Adam

### Конфигурации моделей

#### RNN

```
RNNForCategoryClassification(
  (embedding): Embedding(31217, 256, padding_idx=0)
  (dropout): Dropout(p=0.2, inplace=False)
  (rnn): RNN(256, 64, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=128, out_features=4, bias=True)
)
```

#### CNN

```
CNNForCategoryClassification(
  (embedding): Embedding(31217, 512, padding_idx=0)
  (dropout): Dropout(p=0.2, inplace=False)
  (conv): Conv1d(512, 10, kernel_size=(3,), stride=(1,))
  (relu): ReLU()
  (global_max_pool): AdaptiveMaxPool1d(output_size=1)
  (fc): Linear(in_features=10, out_features=4, bias=True)
)
```

#### LSTM

```
LSTMForCategoryClassification(
  (embedding): Embedding(31217, 256, padding_idx=0)
  (dropout): Dropout(p=0.2, inplace=False)
  (lstm): LSTM(256, 64, batch_first=True, bidirectional=True)
  (fc): Linear(in_features=128, out_features=4, bias=True)
)
```

#### CNN + GloVe

```
GloveCNNForCategoryClassification(
  (embedding): Embedding(20003, 100, padding_idx=0)
  (dropout): Dropout(p=0.3, inplace=False)
  (conv): Conv1d(100, 10, kernel_size=(3,), stride=(1,))
  (relu): ReLU()
  (global_max_pool): AdaptiveMaxPool1d(output_size=1)
  (fc): Linear(in_features=10, out_features=4, bias=True)
)
```

Результаты экспериментов представлены в таблице

| Model                | F1 Score | Precision | Recall | Accuracy | Training Time (s) |
|----------------------|----------|-----------|--------|----------|-------------------|
| LogisticRegression   | 0.935    | 0.937     | 0.935  | 0.936    | 55.211            |
| RNN                  | 0.935    | 0.935     | 0.935  | 0.936    | 358.997           |
| CNN                  | 0.934    | 0.935     | 0.934  | 0.935    | 563.910           |
| LSTM                 | 0.934    | 0.933     | 0.936  | 0.935    | 572.373           |
| CNN with GloVe       | 0.915    | 0.912     | 0.919  | 0.915    | 159.950           |

В целом, использование нейросетей не улучшило качество предсказаний.

## DistilBERT

### Criterion

Такой же CrossEntropyLoss с весами

### Optimizer

Здесь уже вместо Adam я использовал привычный оптимайзер для трансформеров - AdamW

### Архитектура

```
DistilBertForCategoryClassification(
  (bert): DistilBertModel(
    (embeddings): Embeddings(
      (word_embeddings): Embedding(30522, 768, padding_idx=0)
      (position_embeddings): Embedding(512, 768)
      (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (transformer): Transformer(
      (layer): ModuleList(
        (0-5): 6 x TransformerBlock(
          (attention): DistilBertSelfAttention(
            (q_lin): Linear(in_features=768, out_features=768, bias=True)
            (k_lin): Linear(in_features=768, out_features=768, bias=True)
            (v_lin): Linear(in_features=768, out_features=768, bias=True)
            (out_lin): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (sa_layer_norm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (ffn): FFN(
            (dropout): Dropout(p=0.1, inplace=False)
            (lin1): Linear(in_features=768, out_features=3072, bias=True)
            (lin2): Linear(in_features=3072, out_features=768, bias=True)
            (activation): GELUActivation()
          )
...
    )
  )
  (dropout): Dropout(p=0.3, inplace=False)
  (fc): Linear(in_features=768, out_features=4, bias=True)
)
```

### Результаты

| Model              | F1 Score | Precision | Recall | Accuracy | Training Time (s) |
|--------------------|----------|-----------|--------|----------|-------------------|
| LogisticRegression | 0.935    | 0.937     | 0.935  | 0.936    | 62.236            |
| DistilBERT         | 0.947    | 0.946     | 0.949  | 0.947    | 1868.951          |

Здесь уже удалось достичь больше результатов и улучшить метрику F1 на 0.012 пунктов, хоть и с увеличением обучения на 30 минут

## Выводы

В результате проведения экспериментов самые лучшие результаты показала DistilBERT

| Model              | F1 Score | Precision | Recall | Accuracy | Training Time (s) |
|--------------------|----------|-----------|--------|----------|-------------------|
| DistilBERT         | 0.947    | 0.946     | 0.949  | 0.947    | 1868.951          |