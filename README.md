# Adversarial_attacks_on_LLMs
Semester project done as part of the LTS4 lab at EPFL. 

In progress

Repository structure: 
````
adversarial_LLM/
│
├── config.ini
├── .gitignore
│
├── gen_adv_examples/
│   └── config_loader.py
│   └── data_processing.py
│   └── logger.py
│   └── main.py
│   └── model_loader.py
│   └── model_utils.py
│   └── response_generator.py
│   └── sentence_modification.py
│   └── sentiment_analysis.py
│   └── utils.py
│   └── logs/
│       └── ......
│       └── ......
├── datasets/
│   └── sst2
│       └── data
│            └── test-00000-of-00001.parquet
│            └── train-00000-of-00001.parquet
│            └── validation-00000-of-00001.parquet
│            └── sst2_fs_examples.pkl
│       └── .gitattributes
│       └── README.md
├── outputs/
│   └── final_modified_dataset/
│       └── ......
│       └── ......
│   └── json_info/
│       └── ......
│       └── ......
│   └── csv_info/
│       └── ......
│       └── ......
├── adv_samples/
│   └── .....parquet
│   └── .....parquet
│   └── .....parquet
│
└── ----
````
Adversarial attacks are performed on the SST2 validation dataset and sentiment analysis tasks using the Mistral 7B v02 instruct model. 
Parameters for attacks can be modified in config.ini file.
