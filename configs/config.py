QA_PATH = '/content/drive/MyDrive/ABC/KnowGPT/train_rand_split.jsonl'
CONCEPTNET_PATH = '/content/drive/MyDrive/ABC/KnowGPT/conceptnet_en.csv'
PKL_CCN_PATH = lambda sample: f'/content/drive/MyDrive/ABC/KnowGPT/ccn_samples_{sample}.pkl'
ENTITY_TO_VEC_PATH = f'/content/drive/MyDrive/ABC/KnowGPT/entity_to_vec.pkl'
TRIPLE_TO_VEC_PATH = f'/content/drive/MyDrive/ABC/KnowGPT/triple_to_vec.pkl'

EXTRACTED_TRAIN_PATH = '/content/drive/MyDrive/ABC/KnowGPT/extracted_train_paths.pkl'

MAB_TRAIN_PATH = '/content/drive/MyDrive/ABC/KnowGPT/mab_train_paths.pkl'

RESULT_TRAIN_PATH = f'/content/drive/MyDrive/ABC/KnowGPT/mab_training_log.csv'

USE_TRIPLE_POLICY = True
EMBEDDING_DIM = 384
HIDDEN_DIM = 256
NUM_EPOCHS = 5
REPEATS_PER_SAMPLE = 3
MODEL_NAME = "all-MiniLM-L6-v2"
sample_size = None