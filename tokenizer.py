import sentencepiece as spm
import datasets

import constants

# Train sentencepiece tokeniser 
def train(prefix):
    spm.SentencePieceTrainer.Train(
        input=f'{prefix}.txt',
        model_prefix=prefix,
        vocab_size=constants.VOCAB_SIZE,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )

# Implement tokeniser 
class Tokenizer:
    def __init__(self, prefix):
        model_path = f'{prefix}.model'
        self.sp = spm.SentencePieceProcessor(model_file=model_path)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, ids):
        return self.sp.decode_ids(ids)

    def get_vocab(self):
        return {self.sp.id_to_piece(id): id for id in range(self.sp.get_piece_size())}



if __name__ == '__main__':
    dataset = datasets.load_dataset("Clinton/Text-to-sql-v1")

    questions = [example['instruction'] for example in dataset['train']]
    answers = [example['response'] for example in dataset['train']]

    with open('questions.txt', 'w', encoding='utf-8') as f:
        for text in questions: f.write(text + '\n')

    with open('answers.txt', 'w', encoding='utf-8') as f:
        for text in answers: f.write(text + '\n')

    train('questions')
    train('answers')
