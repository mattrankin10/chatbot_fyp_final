# Imports the Google Cloud client library
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

# Instantiates a client
client = language.LanguageServiceClient()


def analyze_text(text_content):
    # The text to analyze
    document = types.Document(
        content=text_content,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document).document_sentiment
    response = client.analyze_entities(document)
    sent_score = sentiment.score
    sent_magnitude = sentiment.magnitude
    # this = client.annotate_text(document)

    print('Text: {}'.format(text_content))
    print('Sentiment: {}, {}'.format(sent_score, sentiment.magnitude))
    print(sent_score)

    for e in response.entities:
        print(u"Representative name for the entity: {}".format(e.name))
        print(u"Entity type: {}".format(enums.Entity.Type(e.type).name))
        # Get the salience score associated with the entity in the [0, 1.0] range
        print(u"Salience score: {}".format(e.salience))
        # Loop over the metadata associated with entity. For many known entities,
        # the metadata is a Wikipedia URL (wikipedia_url) and Knowledge Graph MID (mid).
        # Some entity types may have additional metadata, e.g. ADDRESS entities
        # may have metadata for the address street_name, postal_code, et al.
        for metadata_name, metadata_value in e.metadata.items():
            print(u"{}: {}".format(metadata_name, metadata_value))

        for mention in e.mentions:
            print(u"Mention text: {}".format(mention.text.content))
            # Get the mention type, e.g. PROPER for proper noun
            print(u"Mention type: {}".format(enums.EntityMention.Type(mention.type).name))

    return sent_score, sent_magnitude


# def get_sentiment(text_content):

def main():
    text_content = "Hi my name is matt"
    analyze_text(text_content)


if __name__ == "__main__":
    main()
