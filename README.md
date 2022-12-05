# Youtube Comment Dissector

Web scraping tool to take comments using Selenium and dissect as much information as possible from a given video 
comment section using AI and other techniques.

# Sentiment

Sentiment is calculated using the pre-trained AI by: <a>cardiffnlp/twitter-roberta-base-sentiment-latest</a>
the values are then used to create a Dataframe using pandas alongside other informations and then converted 
to a HTML file that can be visualised