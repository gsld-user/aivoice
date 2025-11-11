import azure.cognitiveservices.speech as speechsdk

speech_key = "AxtWt7DKFwDapzaiOHjMHLaGvBNqxMnW4HwtrrHzCVd6ywaHMjQ6JQQJ99BIAC3pKaRXJ3w3AAAYACOGI9e9"
region = "eastasia"

try:
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    print("SpeechConfig 建立成功！")
except Exception as e:
    print("錯誤:", e)
