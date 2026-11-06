from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import Unigram

if __name__ == "__main__":
    tokenizer = Tokenizer(
        Unigram(
            vocab=[("[UNK]", 0.0)],
        )
    )

    tokenizer.train_from_iterator(load_dataset("BabyLM-community/babylm-xho", split="train")["text"])
