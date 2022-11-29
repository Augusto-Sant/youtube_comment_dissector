from comments_scraper import activate_chrome_driver, CommentsScraper, CommentsSentimentAnalyser
from sentiment_ai_model import sentiment_model


def main():
    url = 'https://www.youtube.com/watch?v=Ze1C1kyETi8&t=506s'

    path_to_driver = "C:/SeleniumDrivers/chromedriver.exe"
    driver = activate_chrome_driver(path_to_driver)

    comments_scraper = CommentsScraper(url)
    comments = comments_scraper.scrape_comments(driver)

    tokenizer = sentiment_model.model_tokenizer()
    model = sentiment_model.pre_trained_model()

    comments_analyser = CommentsSentimentAnalyser(tokenizer, model)
    d_comments_sentiments = comments_analyser.parse_sentiment_comments(comments)
    comments_analyser.make_formatted_comments_html(d_comments_sentiments)

    driver.quit()


if __name__ == '__main__':
    main()
