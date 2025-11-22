# PumpkingLLM Pre-Alpha

Я всегда интересовался NLP задачами а так же LLM. Чтобы лучше понять как устроены LLM под капотом, а так же как их реализовывать, обучать и использовать я решил написать свою собственную LLM. Собственно в данном репозитории представлена самая первая версия данной LLM. 

## Aboat model

Главной целью данного репозитория была реализация с нуля класса GPT используя высокоуровневый фреймворк глубого обучения [PyTorch]("https://pytorch.org/"), класса Tokenizer использующий принцип [BPE]("https://en.wikipedia.org/wiki/Byte-pair_encoding"), а так же минимальное обучение модели и проверка ее работоспособности. По итогам у меня вышел минимальный пайплайн LLM, который успешно работает.

## File structure

```

PumpkingLLM-Pre-Alpha/
├── config/
|   ├── hyperparams.py      # гиперпараметры модели            
│   └── paths.py            # пути сохранения и загрузки токенизатора и модели
├── model/
│   ├── gpt.py              # собственный класс GPT  
│   ├── layers.py           # собственные классы DecoderBlock и PositionalEncoding
│   ├── tokenizer.py        # собственный BPE-tokenizer
│   └── utils.py            # маски для DecoderBlock
├── data/
│   ├── dataset.py          # датасет для GPT
│   └── prepare_data.py     # класс обучение Tokenizer
├── train.py                # класс обучения GPT
├── generate.py             # генерация текста 
├── testinggpt              # ноутбук с обучением и тестированием GPT
├── README.md
├── LICENSE
└── requirements.txt
```
