import os
import pyaudio
import wave
import time
from datetime import datetime,timedelta
import whisper
import torch
from threading import Thread
from flask import Flask

app = Flask(__name__)

# Global variable to store the recording state
recording = False
waves = ""
status = False
start_time = 0

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

        today = str(datetime.now().date())
        path = os.path.join(os.getcwd(), today)
        print(path)
        if os.path.exists(path):
            print("Directory Available")
        else:
            os.mkdir(path)
            print("Directory Created")
        print("Saved Audio Path", path)

        waves = str(current_time)+".wav"

        audiosave = os.path.join(path,waves)
        wave_file = wave.open(audiosave, 'wb')
        # wave_file = wave.open(waves, 'wb')
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
    today = datetime.now().date()
    print("Today",today)
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

    # Whisper Params
    model = whisper.load_model("base",device=DEVICE)
    try:
        # SAVING RESULTS
        today = str(datetime.now().date())
        path = os.path.join(os.getcwd(), today)
        print(path)
        if os.path.exists(path):
            print("Directory Available")
        else:
            os.mkdir(path)
            print("Directory Created")
        print("Saved Translation Path", path)
        wav_file = os.path.join(path,waves)

        #result = model.transcribe('f3.wav',word_timestamps=True)
        result = model.transcribe(wav_file,word_timestamps=True)
        t4 = time.time()
        print("Time to Decode : ", t4 - t3)

        file = waves+"_output.txt"
        savin = os.path.join(path,file)

        # with open(savin, 'w') as file:
        #     file.write(result["text"])

        print(result["text"])

        #Extracting Segment level TimeStamps
        content = ''
        contentlist = []
        for i in range(len(result["segments"])):
            localdict = {}
            # print(result["segments"][i]["text"],start_time+timedelta(seconds=result["segments"][i]['start']),
            #       start_time+timedelta(seconds=result["segments"][i]['end']))
            content += result["segments"][i]["text"]+" "+str(start_time+timedelta(seconds=result["segments"][i]['start']))+" "+\
                       str(start_time+timedelta(seconds=result["segments"][i]['end']))+"\n"
            localdict["sentence"] = result["segments"][i]["text"]
            localdict["start_time"] = str(start_time+timedelta(seconds=result["segments"][i]['start']))
            localdict["end_time"] = str(start_time+timedelta(seconds=result["segments"][i]['end']))
            # print(localdict)
            contentlist.append(localdict)

        with open(savin, 'w') as file:
            file.write(content)
        t5 = time.time()
        print("Time to Save : ", t5 - t3)
    except:
        print("No Audio File Recorded")
        return {"transcription":"---"}

    return {"transcription":contentlist}                    #Returns content with time stamps
    #return {"transcription":result["text"]}            #Returns only content

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

        global start_time
        start_time = datetime.now()

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
    app.run(debug=True)
