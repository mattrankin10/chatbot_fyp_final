from google.cloud import language


def language_analysis(text):
    client = language.LanguageServiceClient()
    document = client.Document(text)
    sent_anal = document.analyze_sentiment()
    print(dir(sent_anal))
    sentiment = sent_anal.sentiment
    ent_anal = document.analyze_entities()
    entities = ent_anal.entities
    return sentiment, entities


example_text = 'matt is the best footballer in the world'
sentiment, entities = language_analysis(example_text)
print(sentiment.score, sentiment.magnitude)

for e in entities:
    print(e.name)