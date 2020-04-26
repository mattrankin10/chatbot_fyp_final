from google.cloud import translate
from itertools import islice

project_id = 'chatbotfyp-1'
"""Translating Text."""

client = translate.TranslationServiceClient()

parent = client.location_path(project_id, "global")


def translate(from_file, to_file):
    with open(from_file, 'r') as chinese_f:
        with open(to_file, 'w') as eng_f:
            lines = chinese_f.readlines()
            for line in lines:
        # Detail on supported types can be found here:
        # https://cloud.google.com/translate/docs/supported-formats
                response = client.translate_text(
                    parent=parent,
                    contents=[str(line)],
                    mime_type="text/plain",  # mime types: text/plain, text/html
                    source_language_code="zh-CN",
                    target_language_code="en",
                )
                # Display the translation for each input text provided
                for translation in response.translations:
                    eng = translation.translated_text
                    eng_f.write(eng)
    chinese_f.close()
    eng_f.close()

#translate('train.resp', 'train_eng.resp')
translate('test.post', 'test_eng.post')
