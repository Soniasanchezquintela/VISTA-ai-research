# VISTA-ai-research
Collaborative repository to share papers and notes for the VISTA project
Biblography - keywords (keep adding): Visual Impairment
Object Location, Area Of Research, Visual System, Object Detection,,Impaired Individuals, User Privacy, Assistance Systems, Millions Of People Worldwide, Impaired People, Edge DevicesWearable SystemSystem Usability ScaleBlind IndividualsNASA Task Load IndexFemale ParticipantsHorizontal PlaneMale ParticipantsExperimental Trials

## Tools in this repo

- train_yolo.py: script to start training the Yolo11 model. When you execute this script, it will start downloading everything it needs.

- heic_to_jpg.sh: script to convert iPhone images in HEIC format to standard jpeg. How to use it:

```
./heic_to_jpg.sh <directory_with_heic_images> <quality>
```

This will convert all images in the specified directory with the same output name as the original one but with .jpg extension. By default, quality is set to 95%. It is recommended not to use low values since then details in the image are lost and this heavily impacts AI model's performance.

- test_one_image.py: inference with the Yolo11 object detector. You need to download the weight file sku110k_768_e20_pat5.pt to this same directory, which can be found in the shared [Google Drive folder](https://drive.google.com/drive/folders/1Looy2jQJs1j0C8SvU6CuQzLTPKX01EGI). **Remember you need first to create your virtual environment before you can run this program!**

## Create your virtual environment

The virtual environment contains all the packages you need to execute a Python program. For this project, you can create it this way:

```
python3 -m venv vista_env
source vista_env/bin/activate
pip install -r requirements.txt
```

You only need to this once. Afterwards, you only need to activate it:
```
source vista_env/bin/activate
```

## Text to Speech using piper-tts

File `speak.py`is a test script to easily launch an example of text to voice conversion using the piper-tts package.
To test it, you can simply:

```
python3 speak.py "Text to produce" <language>
```

For example, to convert a sentence in both Spanish and English:

```
python3 speak.py "Hay una botella de leche en el estante superior." es
python3 speak.py "I can see a milk bottle on the top shelf." en
```

Currently it only supports 2 languages: Spanish (es) and English (en).

## Completion Time, Object Detection Model, Object Tracking, Physical Demands, Single-board Computer, Temporal Demand, Constant Feedback, Bounding Box, Participants In Experiment, You Only Look Once, Headphones. 
#https://ieeexplore.ieee.org/abstract/document/10463430 (subscription needed - not accessible via UPF or UPC). Abstract: According to the World Health Organization, hundreds of millions of people worldwide are affected by visual impairments. This has profound personal effects, as our perception, cognition, learning, and daily activities are mediated through vision. In this study, we introduce a wearable visual assistance system designed for visually impaired and blind individuals to help with locating personal items, an essential daily activity. Our system enhances object localization by bridging advanced computer vision-based object detection with spatial sound feedback. We run our method locally on an edge device to protect user privacy. We conducted extensive experiments with 44 participants to study the effectiveness of our system. We evaluated our system using the System Usability Scale and the NASA Task Load Index questionnaires. The experimental results show that our visual assistant system reduces the average object localization time by 37% and improves the successful localization rate by 2.2 times. The positive feedback from the participants highlights the potential of our system to improve the quality of life of visually impaired and blind people. We have made our source code publicly available at https://github.com/IS2AI/visual_assistant to stimulate further research in this area. Published in: 2025 11th International Conference on Control, Automation and Robotics (ICCAR)
# Applications: Seeing AI (Microsoft): This free app uses the camera to recognize text, objects, barcodes, handwriting, and currency. It can also read subtitles.
          Envision App & Glasses: This app uses AI to speak written information in over 60 languages, describe surroundings, and identify objects. The "Ask Envision" feature uses           GPT-4 for more detail.
          Be My Eyes (with "Be My AI"): Users take a photo of their surroundings, which is then analyzed for a detailed description.
          EMVI: This new app supports reading, scanning, and object recognition.
          Gemini Live (with Video): Available on some phones, this enables users to ask questions in real-time about their environment.
          Aira: Connects users with professional agents via live video for assistance. It is now integrated with "Access AI" for automatic scene descriptions.
          Oko: An AI tool that helps visually impaired pedestrians navigate traffic lights.
          EchoVision Glasses: These smart glasses offer real-time descriptions of surroundings. 
