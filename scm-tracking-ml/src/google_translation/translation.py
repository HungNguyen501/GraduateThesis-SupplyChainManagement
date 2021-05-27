import pandas as pd
import fasttext
import json
import requests

class TRANSLATION:
    def __init__(self):
        self.root_path = "google_translation/"
        self.fasttext_model = fasttext.load_model(self.root_path + 'lid.176.bin')
        with open(self.root_path + 'dictionary.txt', 'rb') as json_file:
           self.dictionary = json.load(json_file)

    def detect_language(self, text):
        score = self.fasttext_model.predict(text, k = 1)
        label = score[0][0].replace('__label__', '')
        return label

    def translate(self, text, target_lang = 'en', api_key = 'AIzaSyBvGF21hPuCszJev1sSBEn6UnWM9ij8mfI'):  
        url = f"https://translation.googleapis.com/language/translate/v2?q={text}&target={target_lang}&key={api_key}"
        payload={}
        headers = {}
        # Sent request to Google translate
        response = requests.request("POST", url, headers=headers, data=payload)
        #  Check 200 OK
        if response.status_code == 200:
            response_data = response.json()
        else:
            score = text
            print(f"Google transalte\nStatus code: {response.status_code}")
            return score
        # parse data to get score
        try:
            score = response_data['data']['translations'][0]['translatedText']
        except Exception as ex:
            score = text
            print(f"Google transalte\nException: {ex}")    

        return score
