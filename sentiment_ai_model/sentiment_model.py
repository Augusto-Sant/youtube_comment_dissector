from transformers import AutoTokenizer, AutoModelForSequenceClassification

def pre_trained_model():
    """
    Calls pre-trained model from URL.
    :return:
    """
    model_url = f'cardiffnlp/twitter-roberta-base-sentiment-latest'
    model = AutoModelForSequenceClassification.from_pretrained(model_url)
    return model


def model_tokenizer():
    """
    Calls tokenizer for the model.
    :return:
    """
    model_url = f'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(model_url)
    return tokenizer
