from typing import Tuple

import torch
from nltk.tag import StanfordPOSTagger
from transformers import BertTokenizer, BertForMaskedLM

from .consts import RELATIVE_PRONOUNS, MASK, ARTICLES

_jar = "./stanford-postagger-full-2018-10-16/stanford-postagger-3.9.2.jar"
_tagger = "./stanford-postagger-full-2018-10-16/models/spanish.tagger"

_pos_tagger = StanfordPOSTagger(_tagger, _jar, encoding="utf8")
_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
_model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")


def predictor(text):
    text = mask_sentence(text)
    masked_indexes = []
    tokenized_text = _tokenizer.tokenize(text)
    indexed_tokens = _tokenizer.convert_tokens_to_ids(tokenized_text)

    index = 0
    for tup in zip(tokenized_text, indexed_tokens):
        if tup[0] == "[MASK]":
            masked_indexes.append(index)
        index += 1

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        outputs = _model(tokens_tensor, token_type_ids=segments_tensors)
        predictions = outputs[0]

    predicted_tokens = []
    for masked_index in masked_indexes:
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_tokens.append(_tokenizer.convert_ids_to_tokens([predicted_index])[0])

    c = 0
    texto_final = []
    print(text)
    print(predicted_tokens)
    for word in text.split():
        if word == "[CLS]" or word == "[SEP]":
            continue
        if word == "[MASK]":
            if predicted_tokens[c].lower() in ARTICLES:
                texto_final.append(predicted_tokens[c])
                c += 1
            else:
                c += 1
                continue
        else:
            texto_final.append(word)

    return " ".join(texto_final)


def tag(sentence):
    sentence = sentence.split()
    return _pos_tagger.tag(sentence)


def mask_sentence(sentece):
    tagged_sentence = tag(sentece)
    prev = [("", "")]
    masked_sentence = ["[CLS]"]

    for tagged_word in tagged_sentence:

        # SI ANTES DE ADV+ADJ NO HAY NI SUST NI ART PUEDE IR UN ART
        # Juan es el más listo
        if (
            is_adjective(tagged_word)
            and is_adverb(prev[-1])
            and not (is_determinant(prev[-2]) or is_noun(prev[-2]))
        ):
            masked_sentence.append(MASK)
            # print(1, tagged_word[0])

        # SI ANTES DE UN ADJ O DE UN ADV NO HAY NI SUST NI ART PUEDE IR UN ART
        # los listos y los guapos // han ganado los rojos
        # TODO(Xiang): cambiado "is_adverb(prev[-1])" por "is_adverb(tagged_word)"
        if (is_adjective(tagged_word) or is_adverb(tagged_word)) and not (
            is_determinant(prev[-1]) or is_noun(prev[-1])
        ):
            masked_sentence.append(MASK)
            # print(2, tagged_word[0])

        # ART-PRONREL(QUE, CUAL, CUALES) = PRON REL COMPUESTO (EL QUE, EL CUAL, LOS CUALES)
        # TODO(Xiang): .lower()
        if tagged_word[0].lower() in RELATIVE_PRONOUNS and not is_determinant(prev[-1]):
            masked_sentence.append(MASK)
            # print(3, tagged_word[0])

        # ANTES DE UN SUST PUEDE IR UN ART
        # TODO(Xiang): he comprobado si ya tiene un articulo
        if is_noun(tagged_word) and not is_determinant(prev[-1]):
            masked_sentence.append(MASK)
            # print(4, tagged_word[0])

        masked_sentence.append(tagged_word[0])
        prev.append(tagged_word)

    masked_sentence.append("[SEP]")
    return " ".join(masked_sentence)


# helper functions
def is_noun(tagged_word: Tuple[str, str]):
    return tagged_word[1].startswith("n")


def is_adjective(tagged_word: Tuple[str, str]):
    return tagged_word[1].startswith("a")


def is_adverb(tagged_word: Tuple[str, str]):
    return tagged_word[1].startswith("r")


def is_determinant(tagged_word: Tuple[str, str]):
    return tagged_word[1].startswith("d")
