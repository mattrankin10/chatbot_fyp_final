from google.cloud import translate

project_id='chatbotfyp-1'
"""Translating Text."""

client = translate.TranslationServiceClient()

parent = client.location_path(project_id, "global")
chinese_post = []
with open('test.post', 'r') as f:
    lines = f.readlines()
    for line in lines:
        chinese_post.append(line)
# Detail on supported types can be found here:
# https://cloud.google.com/translate/docs/supported-formats
response = client.translate_text(
    parent=parent,
    contents=chinese_post,
    mime_type="text/plain",  # mime types: text/plain, text/html
    source_language_code="zh-CN",
    target_language_code="en",
)
# Display the translation for each input text provided
with open('eng.post', 'w') as file:
    for translation in response.translations:
        eng = translation.translated_text
        file.write(eng)

