# You can find several models at https://alphacephei.com/vosk/models
import json
from vosk import Model,KaldiRecognizer,SetLogLevel
import pyaudio
import wave
import time
import whisper
import torch
from threading import Thread
from flask import Flask

app = Flask(__name__)

# Global variable to store the recording state
recording = False
waves = ""
status = False
model = ""

def listen():
    # Initialize PyAudio
    audio = pyaudio.PyAudio()
    # Audio recording parameters
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024

    global recording
    recording = True

    if recording:
        # Open an audio stream
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                            rate=RATE, input=True, frames_per_buffer=CHUNK)
        print('recording')

        frames = []
        while recording:
            audio_data = stream.read(CHUNK)
            frames.append(audio_data)
        stream.stop_stream()
        stream.close()
        audio.terminate()
        print('stopped Streaming')

        t1 = time.time()
        global waves

        t = time.localtime()
        current_time = time.strftime("%H_%M_%S", t)

        waves = str(current_time)+".wav"
        wave_file = wave.open(waves, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(b''.join(frames))
        t2 = time.time()
        print("Time to Save : ", t2 - t1)
        wave_file.close()

# Route to the index page
@app.route('/')
def index():
    return "Hello!!!!"

# Route to stop recording
@app.route('/stop')
def stop_recording():
    global recording
    recording = False

    global status
    status = False

    time.sleep(0.1)

    global waves
    print("name of the File",waves)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    t3 = time.time()


    # open audio file
    wf = wave.open("Old_Audio_Translations/15_23_15.wav", "rb")

    global model
    rec = KaldiRecognizer(model, wf.getframerate())

    # To store our results
    transcription = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            # Convert json output to dict
            result_dict = json.loads(rec.Result())
            # Extract text values and append them to transcription list
            transcription.append(result_dict.get("text", ""))

    # Get final bits of audio and flush the pipeline
    final_result = json.loads(rec.FinalResult())
    transcription.append(final_result.get("text", ""))

    # merge or join all list elements to one big string
    transcription_text = ' '.join(transcription)
    print(transcription_text)

    savin = waves+"_outputVOSK.txt"
    # with open(savin, 'w') as file:
    #     file.write(result["text"])
    t4 = time.time()
    print("Time to Decode : ", t4 - t3)
    with open(savin, 'w') as file:
        file.write(str(transcription_text))

    return {"transcription":transcription_text}

# Route to handle the audio recording
@app.route('/record')
def record():
    global status
    if status:
        print("Already Recording")
        return {"status":"Already Recording"}
    else:
        thread = Thread(target=listen)
        thread.daemon = True
        thread.start()
        status = True
        print("Thread Started")
        return {"status":"Recording"}


@app.route('/record_test')
def record_test():
    try:
        audioT = pyaudio.PyAudio()
        # Open an audio stream
        stream = audioT.open(format=pyaudio.paInt16, channels=1,
                             rate=44100, input=True, frames_per_buffer=1024)

        return "Test!!! Audio Device Found"
    except:
        return "No Audio Device Found"


if __name__ == '__main__':
    model = Model("vosk-model-en-us-0.22")
    print("Model Loaded")

    app.run(debug=True)