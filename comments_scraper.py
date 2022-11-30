from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait  # wait until
from selenium.webdriver.support import expected_conditions as EC  # conditions for wait
from time import sleep
import random
import pandas
from sentiment_ai_model import sentiment_model
##
from scipy.special import softmax


def activate_chrome_driver(path):
    """
    Activates Chrome driver for scraping based on the path to it.
    :param path:
    :return:
    """
    service = Service(executable_path=path)
    options = webdriver.ChromeOptions()
    options.add_argument("start-maximized")
    driver = webdriver.Chrome(service=service, options=options)
    return driver


class CommentsScraper:
    """
    Class that Scrapes comments on a determined Video
    """
    def __init__(self, url):
        self.url = url

    def scrape_comments(self, driver, c_loops=500):
        """
        Scrapes comments until end of the loop, c_loops.
        :param driver:
        :param c_loops:
        :return:
        """
        driver.get(self.url)
        sleep(5)
        driver.execute_script(f"window.scrollBy(0,{700})", "")
        sleep(2)
        comment_section = WebDriverWait(driver, timeout=10).until(lambda d: d.find_element(By.ID, 'sections'))
        y = 100
        for _ in range(c_loops):
            driver.execute_script(f"window.scrollBy(0,{y})", "")
            y = random.randint(700, 950)

        comments = comment_section.find_elements(By.ID, 'content-text')
        return comments


class CommentsSentimentAnalyser:
    """
    Class that analyses sentiment of data scraped using AI
    """

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model

    def parse_sentiment_comments(self, comments):
        """
        Processes the sentiment of each comment using a pre-trained AI.
        :param comments:
        :return:
        """
        d_info_comments = {}
        for i, comment in enumerate(comments):
            d_info_comments.update({i: []})
            text = comment.text
            encoded_text = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_text)

            scores = output[0][0].detach().numpy()
            scores = softmax(scores)
            scores_dict = {
                'neg': scores[0],
                'neu': scores[1],
                'pos': scores[2],
            }

            bigger_sentiment = max(scores_dict, key=scores_dict.get)
            if bigger_sentiment == 'pos':
                bigger_sentiment = 'Positive'
            elif bigger_sentiment == 'neu':
                bigger_sentiment = 'Neutral'
            else:
                bigger_sentiment = 'Negative'

            d_info_comments[i].append(comment.text)
            d_info_comments[i].append(bigger_sentiment)

        return d_info_comments

    def _color_sentiment(self, val):
        """
        Mapping to color each cell on dataframe based on sentiment.
        :param val:
        :return:
        """
        if val == 'Positive':
            background_color = 'green'
        elif val == 'Negative':
            background_color = 'red'
        else:
            background_color = '#2b2d42'
        return f'background-color: {background_color}'

    def make_formatted_comments_html(self, d_info_comments):
        """
        Formats a Dataframe into an HTML with sentiment of each comment.

        :param d_info_comments:
        :return:
        """
        df = pandas.DataFrame.from_dict(d_info_comments, orient='index', columns=['Comment', 'Sentiment'])
        stylized_df = df.style.set_properties(**{
            'background-color': '#2b2d42',
            'color': '#edf2f4',
            'border-radius': '12px',
            'padding': '10px',
            'white-space': 'pre-wrap',
        })
        stylized_df.applymap(self._color_sentiment)
        stylized_df.to_html('comments_sentiments.html')
