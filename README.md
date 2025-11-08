mychat/
├── model/
│   ├── gpt.py             # Твой GPT класс
│   ├── layers.py          # DecoderBlock и PosEnc
│   ├── tokenizer.py       # SimpleTokenizer
│   └── utils.py           # masks, helpers
├── data/
│   ├── dataset.py         # TextDataset
│   └── prepare_data.py
├── train.py               # цикл обучения
├── generate.py            # генерация текста
├── chat_cli.py            # CLI-интерфейс
└── requirements.txt


минимальный пайплайн на данный момент.

--написать потом про QKV attention и вприцнипе написать блок теории про attention

--написать про токенизатор что к чему и как я реализовал

--написать про train.py и generate.py