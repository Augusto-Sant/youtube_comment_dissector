from transformers import AutoTokenizer, AutoModelForSequenceClassification


def pre_trained_model():
    model_url = f'cardiffnlp/twitter-roberta-base-sentiment'
    model = AutoModelForSequenceClassification.from_pretrained(model_url)
    return model


def model_tokenizer():
    model_url = f'cardiffnlp/twitter-roberta-base-sentiment'
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    return tokenizer
