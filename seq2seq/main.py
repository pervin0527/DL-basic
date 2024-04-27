import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import spacy
import random
import datasets
import numpy as np

def tokenize_example(example, src_tokenizer, trg_tokenizer, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in src_tokenizer.tokenizer(example["en"])][:max_length]
    de_tokens = [token.text for token in trg_tokenizer.tokenizer(example["de"])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]

    return {"en_tokens": en_tokens, "de_tokens": de_tokens}

if __name__ == "__main__":
    dataset = datasets.load_dataset("bentrevett/multi30k")
    print(dataset)

    train_data, valid_data, test_data = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )

    print(train_data[0])

    en_tokenizer = spacy.load("en_core_web_sm")
    de_tokenizer = spacy.load("de_core_news_sm")

    string = "What a lovely day it is today!"

    result = [token.text for token in en_tokenizer.tokenizer(string)]
    print(result)

    max_length = 1000
    lower = True
    sos_token = "<sos>"
    eos_token = "<eos>"

    fn_kwargs = {
        "src_tokenizer": en_tokenizer,
        "trg_tokenizer": de_tokenizer,
        "max_length": max_length,
        "lower": lower,
        "sos_token": sos_token,
        "eos_token": eos_token,
    }

    train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

    print(train_data[0])