#include "MySpeechWorker.h"
//#include "whisper/whisper.h"
//#include "SpeechRecognitionWorker.h"

//General Log
//DEFINE_LOG_CATEGORY(SpeechRecognitionPlugin);

std::string to_timestamp(int64_t t) {
    int64_t sec = t / 100;
    int64_t msec = t - sec * 100;
    int64_t min = sec / 60;
    sec = sec - min * 60;

    char buf[32];
    snprintf(buf, sizeof(buf), "%02d:%02d.%03d", (int)min, (int)sec, (int)msec);

    return std::string(buf);
}

FMySpeechWorker::FMySpeechWorker() {
	UE_LOG(LogTemp, Warning, TEXT("MySpeechWorker Constructor"));
   
}

FMySpeechWorker::~FMySpeechWorker() {
	delete Thread;
	Thread = NULL;
}

void FMySpeechWorker::ShutDown() {
	Stop();
	Thread->WaitForCompletion();
	delete Thread;
}

void FMySpeechWorker::Stop() {
	ClientMessage(FString("Thread Stopped"));
}

bool FMySpeechWorker::StartThread(AMyActor* manager, FWhisperParams whisperParams) {
	UE_LOG(LogTemp, Warning, TEXT("Thread started"));
    //params = wparams;
	Manager = manager;
    params = whisperParams;
	FString threadName = FString("FSpeechRecognitionWorker:") + FString::FromInt(8);
	Thread = FRunnableThread::Create(this, *threadName, 0U, TPri_Highest);	
	return true;
}

void FMySpeechWorker::ClientMessage(FString text) {
	UE_LOG(LogTemp, Warning, TEXT("%s"), *text);
}


uint32 FMySpeechWorker::Run() {
	ClientMessage(FString("RUN IS CALLED"));

    params.keep_ms = std::min(params.keep_ms, params.step_ms);
    params.length_ms = std::max(params.length_ms, params.step_ms);

    //int WHISPER_SAMPLE_RATE = 16000;

    const int n_samples_step = (1e-3 * params.step_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_len = (1e-3 * params.length_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_keep = (1e-3 * params.keep_ms) * WHISPER_SAMPLE_RATE;
    const int n_samples_30s = (1e-3 * 30000.0) * WHISPER_SAMPLE_RATE;

    const bool use_vad = n_samples_step <= 0; // sliding window mode uses VAD

    const int n_new_line = !use_vad ? std::max(1, params.length_ms / params.step_ms - 1) : 1; // number of steps to print new line

    params.no_timestamps = !use_vad;
    params.no_context |= use_vad;
    params.max_tokens = 0;


    audio_async audio(params.length_ms);

    //int32 capture = 0;
    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
    //if (!audio.init(0, WHISPER_SAMPLE_RATE)) {
        // fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("AUDIO INIT FAILED %s"), __func__);
        //return 1;
    }

    bool resumes = audio.resume();
    
    //struct whisper_context* ctx = whisper_init_from_file(std::string(TCHAR_TO_UTF8(*(params.model))).c_str());
    struct whisper_context* ctx = whisper_init_from_file(params.model.c_str());
    //struct whisper_context* ctx = whisper_init_from_file(TCHAR_TO_ANSI(*params.model));
    if (ctx == nullptr)
    {
        UE_LOG(LogTemp, Warning, TEXT("WHISPER CONTEXT IS A NULL POINTER"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("WHISPER CONTEXT IS NOT A NULL POINTER"));
    }
    std::vector<float> pcmf32(n_samples_30s, 0.0f);
    std::vector<float> pcmf32_old;
    std::vector<float> pcmf32_new(n_samples_30s, 0.0f);
    std::vector<whisper_token> prompt_tokens;

   

    int n_iter = 0;

    bool is_running = true;

    //std::string test2 = std::string(TCHAR_TO_UTF8(*params.language));

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    UE_LOG(LogTemp, Warning, TEXT("[START SPEAKING]"));
    fflush(stdout);

    auto t_last = std::chrono::high_resolution_clock::now();
    const auto t_start = t_last;

	int counter = 0;
	while (counter < 10 && is_running) {
        is_running = sdl_poll_events();
		ClientMessage(FString("Inside while"));
		counter = counter +1;
        //UE_LOG(LogTemp, Warning, TEXT("%d"), params.capture_id);
		UE_LOG(LogTemp, Warning, TEXT("%s"), *FString::FromInt(counter));

        // process new audio

        if (!use_vad) {
            while (true) {
                audio.get(params.step_ms, pcmf32_new);

                if ((int)pcmf32_new.size() > 2 * n_samples_step) {
                    fprintf(stderr, "\n\n%s: WARNING: cannot process audio fast enough, dropping audio ...\n\n", __func__);
                    UE_LOG(LogTemp, Warning, TEXT("WARNING: cannot process audio fast enough, dropping audio"));
                    audio.clear();
                    continue;
                }

                if ((int)pcmf32_new.size() >= n_samples_step) {
                    UE_LOG(LogTemp, Warning, TEXT("Clearing AUDIO"));
                    audio.clear();
                    break;
                }
                //UE_LOG(LogTemp, Warning, TEXT("Getting AUDIO in while"));
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }

            const int n_samples_new = pcmf32_new.size();

            // take up to params.length_ms audio from previous iteration
            const int n_samples_take = std::min((int)pcmf32_old.size(), std::max(0, n_samples_keep + n_samples_len - n_samples_new));

            //printf("processing: take = %d, new = %d, old = %d\n", n_samples_take, n_samples_new, (int) pcmf32_old.size());

            pcmf32.resize(n_samples_new + n_samples_take);

            for (int i = 0; i < n_samples_take; i++) {
                pcmf32[i] = pcmf32_old[pcmf32_old.size() - n_samples_take + i];
            }

            memcpy(pcmf32.data() + n_samples_take, pcmf32_new.data(), n_samples_new * sizeof(float));

            pcmf32_old = pcmf32;
        }
        else {
            UE_LOG(LogTemp, Warning, TEXT("ELSE PART = USE VAD TRUE"));
            const auto t_now = std::chrono::high_resolution_clock::now();
            const auto t_diff = std::chrono::duration_cast<std::chrono::milliseconds>(t_now - t_last).count();

            if (t_diff < 2000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            audio.get(2000, pcmf32_new);

            if (FMySpeechWorker::vad_simple(pcmf32_new, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, false)) {
                audio.get(params.length_ms, pcmf32);
            }
            else {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));

                continue;
            }

            t_last = t_now;
        }

        // run the inference
        {
            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress = false;
            wparams.print_special = params.print_special;
            wparams.print_realtime = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate = params.translate;
            wparams.single_segment = !use_vad;
            wparams.max_tokens = params.max_tokens;
            wparams.language = params.language.c_str();
            wparams.n_threads = params.n_threads;

            wparams.audio_ctx = params.audio_ctx;
            wparams.speed_up = params.speed_up;

            // disable temperature fallback
            wparams.temperature_inc = -1.0f;

            wparams.prompt_tokens = params.no_context ? nullptr : prompt_tokens.data();
            wparams.prompt_n_tokens = params.no_context ? 0 : prompt_tokens.size();

            if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                //fprintf(stderr, "%s: failed to process audio\n", argv[0]);
                UE_LOG(LogTemp, Warning, TEXT("failed to process audio"));
                return 6;
            }
            else {
                UE_LOG(LogTemp, Warning, TEXT("ABLE to process audio"));
            }

            // print result;
            {
                if (!use_vad) {
                    printf("\33[2K\r");

                    // print long empty line to clear the previous line
                    printf("%s", std::string(100, ' ').c_str());
                    UE_LOG(LogTemp, Warning, TEXT("PRINTING new line useVAD=False"));
                    printf("\33[2K\r");
                }
                else {
                    const int64_t t1 = (t_last - t_start).count() / 1000000;
                    const int64_t t0 = std::max(0.0, t1 - pcmf32.size() * 1000.0 / WHISPER_SAMPLE_RATE);

                    printf("\n");
                    UE_LOG(LogTemp, Warning, TEXT("PRINTING Trasnscription useVAD=True"));

                    printf("### Transcription %d START | t0 = %d ms | t1 = %d ms\n", n_iter, (int)t0, (int)t1);
                    printf("\n");
                }

                const int n_segments = whisper_full_n_segments(ctx);
                UE_LOG(LogTemp, Warning, TEXT("Total Number of Segments %d"),n_segments);

                for (int i = 0; i < n_segments; ++i) {
                    const char* text = whisper_full_get_segment_text(ctx, i);

                    if (params.no_timestamps) {
                        printf("%s", text);
                        UE_LOG(LogTemp, Warning, TEXT("NO TIME STAMPS : %s"), *FString(text));
                        fflush(stdout);

                        if (params.fname_out.length() > 0) {
                            fout << text;
                        }
                    }
                    else {
                        const int64_t t0 = whisper_full_get_segment_t0(ctx, i);
                        const int64_t t1 = whisper_full_get_segment_t1(ctx, i);

                        printf("[%s --> %s]  %s\n", to_timestamp(t0).c_str(), to_timestamp(t1).c_str(), text);
                        UE_LOG(LogTemp, Warning, TEXT("ELSE TIMESTAMPS: %s"), *FString(text));

                        if (params.fname_out.length() > 0) {
                            fout << "[" << to_timestamp(t0) << " --> " << to_timestamp(t1) << "]  " << text << std::endl;
                            UE_LOG(LogTemp, Warning, TEXT("ELSE TIMESTAMPS: %s"), *FString(text));
                        }
                    }
                }

                if (params.fname_out.length() > 0) {
                    fout << std::endl;
                }

                if (use_vad) {
                    printf("\n");
                    printf("### Transcription %d END\n", n_iter);
                }
            }

            ++n_iter;

            if (!use_vad && (n_iter % n_new_line) == 0) {
                printf("\n");

                // keep part of the audio for next iteration to try to mitigate word boundary issues
                pcmf32_old = std::vector<float>(pcmf32.end() - n_samples_keep, pcmf32.end());

                // Add tokens of the last full length segment as the prompt
                if (!params.no_context) {
                    prompt_tokens.clear();

                    const int n_segments = whisper_full_n_segments(ctx);
                    for (int i = 0; i < n_segments; ++i) {
                        const int token_count = whisper_full_n_tokens(ctx, i);
                        for (int j = 0; j < token_count; ++j) {
                            prompt_tokens.push_back(whisper_full_get_token_id(ctx, i, j));
                        }
                    }
                }
            }
        }

        if (resumes)
        {
            {
                //whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                UE_LOG(LogTemp, Warning, TEXT("WHISPER HEADER WORKING"));
            }
        }
        else
        {
            UE_LOG(LogTemp, Warning, TEXT("NO AUDIO DEVICE ACTIVATED"));
        }
	}

    audio.pause();

	return 0;
}

///////////////////////////////////////////

bool sdl_poll_events() {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        switch (event.type) {
        case SDL_QUIT:
        {
            return false;
        } break;
        default:
            break;
        }
    }

    return true;
}

//
// SDL Audio capture
//
audio_async::audio_async(int len_ms) {
    m_len_ms = len_ms;

    m_running = false;

    UE_LOG(LogTemp, Warning, TEXT("AUDIO AYSNC INITIATED"));
}

audio_async::~audio_async() {
    if (m_dev_id_in) {
        SDL_CloseAudioDevice(m_dev_id_in);
    }
}

bool audio_async::init(int capture_id, int sample_rate) {
    SDL_LogSetPriority(SDL_LOG_CATEGORY_APPLICATION, SDL_LOG_PRIORITY_INFO);

    if (SDL_Init(SDL_INIT_AUDIO) < 0) {
        SDL_LogError(SDL_LOG_CATEGORY_APPLICATION, "Couldn't initialize SDL: %s\n", SDL_GetError());
        return false;
    }

    SDL_SetHintWithPriority(SDL_HINT_AUDIO_RESAMPLING_MODE, "medium", SDL_HINT_OVERRIDE);

    {
        int nDevices = SDL_GetNumAudioDevices(SDL_TRUE);
        //fprintf(stderr, "%s: found %d capture devices:\n", __func__, nDevices);
        for (int i = 0; i < nDevices; i++) {
            fprintf(stderr, "%s:    - Capture device #%d: '%s'\n", __func__, i, SDL_GetAudioDeviceName(i, SDL_TRUE));
        }
    }

    SDL_AudioSpec capture_spec_requested;
    SDL_AudioSpec capture_spec_obtained;

    SDL_zero(capture_spec_requested);
    SDL_zero(capture_spec_obtained);

    capture_spec_requested.freq = sample_rate;
    capture_spec_requested.format = AUDIO_F32;
    capture_spec_requested.channels = 1;
    capture_spec_requested.samples = 1024;
    capture_spec_requested.callback = [](void* userdata, uint8_t* stream, int len) {
        audio_async* audio = (audio_async*)userdata;
        audio->callback(stream, len);
    };
    capture_spec_requested.userdata = this;

    if (capture_id >= 0) {
        //fprintf(stderr, "%s: attempt to open capture device %d : '%s' ...\n", __func__, capture_id, SDL_GetAudioDeviceName(capture_id, SDL_TRUE));
        UE_LOG(LogTemp, Warning, TEXT("attempt to open capture device %d : '%s'"),0, SDL_GetAudioDeviceName(0, SDL_TRUE));
        m_dev_id_in = SDL_OpenAudioDevice(SDL_GetAudioDeviceName(capture_id, SDL_TRUE), SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    }
    else {
        //fprintf(stderr, "%s: attempt to open default capture device ...\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("attempt to open default capture device"));
        m_dev_id_in = SDL_OpenAudioDevice(nullptr, SDL_TRUE, &capture_spec_requested, &capture_spec_obtained, 0);
    }

    if (!m_dev_id_in) {
        //fprintf(stderr, "%s: couldn't open an audio device for capture: %s!\n", __func__, SDL_GetError());
        UE_LOG(LogTemp, Warning, TEXT("couldn't open an audio device for capturee"));
        m_dev_id_in = 0;

        return false;
    }
    else {
        //fprintf(stderr, "%s: obtained spec for input device (SDL Id = %d):\n", __func__, m_dev_id_in);
       // fprintf(stderr, "%s:     - sample rate:       %d\n", __func__, capture_spec_obtained.freq);
        //fprintf(stderr, "%s:     - format:            %d (required: %d)\n", __func__, capture_spec_obtained.format,
            //capture_spec_requested.format);
        //fprintf(stderr, "%s:     - channels:          %d (required: %d)\n", __func__, capture_spec_obtained.channels,
            //capture_spec_requested.channels);
        //fprintf(stderr, "%s:     - samples per frame: %d\n", __func__, capture_spec_obtained.samples);
    }

    m_sample_rate = capture_spec_obtained.freq;

    m_audio.resize((m_sample_rate * m_len_ms) / 1000);
    UE_LOG(LogTemp, Warning, TEXT("AUDIO DEVICE ACTIVATED"));
    return true;
}

bool audio_async::resume() {
    if (!m_dev_id_in) {
        //fprintf(stderr, "%s: no audio device to resume!\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("no audio device to resume!"));
        return false;
    }

    if (m_running) {
        //fprintf(stderr, "%s: already running!\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("already running!"));
        return false;
    }

    SDL_PauseAudioDevice(m_dev_id_in, 0);
    UE_LOG(LogTemp, Warning, TEXT("AUDIO DEVICE FOUND"));
    m_running = true;

    return true;
}

bool audio_async::pause() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to pause!\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("no audio device to pause!"));
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: already paused!\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("already paused!"));
        return false;
    }

    SDL_PauseAudioDevice(m_dev_id_in, 1);

    m_running = false;
    UE_LOG(LogTemp, Warning, TEXT("AUDIO DEVICE PAUSED"));
    return true;
}

bool audio_async::clear() {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to clear!\n", __func__);
        return false;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        m_audio_pos = 0;
        m_audio_len = 0;
    }

    return true;
}

// callback to be called by SDL
void audio_async::callback(uint8_t* stream, int len) {
    if (!m_running) {
        return;
    }

    const size_t n_samples = len / sizeof(float);

    m_audio_new.resize(n_samples);
    memcpy(m_audio_new.data(), stream, n_samples * sizeof(float));

    //fprintf(stderr, "%s: %zu samples, pos %zu, len %zu\n", __func__, n_samples, m_audio_pos, m_audio_len);

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (m_audio_pos + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - m_audio_pos;

            memcpy(&m_audio[m_audio_pos], stream, n0 * sizeof(float));
            memcpy(&m_audio[0], &stream[n0], (n_samples - n0) * sizeof(float));

            m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
            m_audio_len = m_audio.size();
        }
        else {
            memcpy(&m_audio[m_audio_pos], stream, n_samples * sizeof(float));

            m_audio_pos = (m_audio_pos + n_samples) % m_audio.size();
            m_audio_len = std::min(m_audio_len + n_samples, m_audio.size());
        }
    }
}

void audio_async::get(int ms, std::vector<float>& result) {
    if (!m_dev_id_in) {
        fprintf(stderr, "%s: no audio device to get audio from!\n", __func__);
        return;
    }

    if (!m_running) {
        fprintf(stderr, "%s: not running!\n", __func__);
        return;
    }

    result.clear();

    {
        std::lock_guard<std::mutex> lock(m_mutex);

        if (ms <= 0) {
            ms = m_len_ms;
        }

        size_t n_samples = (m_sample_rate * ms) / 1000;
        if (n_samples > m_audio_len) {
            n_samples = m_audio_len;
        }

        result.resize(n_samples);

        int s0 = m_audio_pos - n_samples;
        if (s0 < 0) {
            s0 += m_audio.size();
        }

        if (s0 + n_samples > m_audio.size()) {
            const size_t n0 = m_audio.size() - s0;

            memcpy(result.data(), &m_audio[s0], n0 * sizeof(float));
            memcpy(&result[n0], &m_audio[0], (n_samples - n0) * sizeof(float));
        }
        else {
            memcpy(result.data(), &m_audio[s0], n_samples * sizeof(float));
        }
    }
}


///////////////////////////


void FMySpeechWorker::high_pass_filter(std::vector<float>& data, float cutoff, float sample_rate) {
    const float rc = 1.0f / (2.0f * M_PI * cutoff);
    const float dt = 1.0f / sample_rate;
    const float alpha = dt / (rc + dt);

    float y = data[0];

    for (size_t i = 1; i < data.size(); i++) {
        y = alpha * (y + data[i] - data[i - 1]);
        data[i] = y;
    }
}

bool FMySpeechWorker::vad_simple(std::vector<float>& pcmf32, int sample_rate, int last_ms, float vad_thold, float freq_thold, bool verbose) {
    const int n_samples = pcmf32.size();
    const int n_samples_last = (sample_rate * last_ms) / 1000;

    if (n_samples_last >= n_samples) {
        // not enough samples - assume no speech
        return false;
    }

    if (freq_thold > 0.0f) {
        high_pass_filter(pcmf32, freq_thold, sample_rate);
    }

    float energy_all = 0.0f;
    float energy_last = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        energy_all += fabsf(pcmf32[i]);

        if (i >= n_samples - n_samples_last) {
            energy_last += fabsf(pcmf32[i]);
        }
    }

    energy_all /= n_samples;
    energy_last /= n_samples_last;

    if (verbose) {
        //fprintf(stderr, "%s: energy_all: %f, energy_last: %f, vad_thold: %f, freq_thold: %f\n", __func__, energy_all, energy_last, vad_thold, freq_thold);
    }

    if (energy_last > vad_thold * energy_all) {
        return false;
    }

    return true;
}