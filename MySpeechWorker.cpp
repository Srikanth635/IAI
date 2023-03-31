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

std::string transcribe(whisper_context* ctx, const FWhisperParams& params, const std::vector<float>& pcmf32, float& prob, int64_t& t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    prob = 0.0f;
    t_ms = 0;

    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

    wparams.print_progress = false;
    wparams.print_special = params.print_special;
    wparams.print_realtime = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate = params.translate;
    wparams.no_context = true;
    wparams.single_segment = true;
    wparams.max_tokens = params.max_tokens;
    wparams.language = params.language.c_str();
    wparams.n_threads = params.n_threads;

    wparams.audio_ctx = params.audio_ctx;
    wparams.speed_up = params.speed_up;

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    int prob_n = 0;
    std::string result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);

        result += text;

        const int n_tokens = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n_tokens; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            prob += token.p;
            ++prob_n;
        }
    }

    if (prob_n > 0) {
        prob /= prob_n;
    }

    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}

// compute similarity between two strings using Levenshtein distance
float similarity(const std::string& s0, const std::string& s1) {
    const size_t len0 = s0.size() + 1;
    const size_t len1 = s1.size() + 1;

    std::vector<int> col(len1, 0);
    std::vector<int> prevCol(len1, 0);

    for (size_t i = 0; i < len1; i++) {
        prevCol[i] = i;
    }

    for (size_t i = 0; i < len0; i++) {
        col[0] = i;
        for (size_t j = 1; j < len1; j++) {
            col[j] = std::min(std::min(1 + col[j - 1], 1 + prevCol[j]), prevCol[j - 1] + (s0[i - 1] == s1[j - 1] ? 0 : 1));
        }
        col.swap(prevCol);
    }

    const float dist = prevCol[len1 - 1];

    return 1.0f - (dist / std::max(s0.size(), s1.size()));
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

    struct whisper_context* ctx = whisper_init_from_file(params.model.c_str());

    audio_async audio(30*1000);

    if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
    //if (!audio.init(0, WHISPER_SAMPLE_RATE)) {
        // fprintf(stderr, "%s: audio.init() failed!\n", __func__);
        UE_LOG(LogTemp, Warning, TEXT("AUDIO INIT FAILED %s"), __func__);
        //return 1;
    }

    bool resumes = audio.resume();
    
    //struct whisper_context* ctx = whisper_init_from_file(std::string(TCHAR_TO_UTF8(*(params.model))).c_str());
    
    //struct whisper_context* ctx = whisper_init_from_file(TCHAR_TO_ANSI(*params.model));
    if (ctx == nullptr)
    {
        UE_LOG(LogTemp, Warning, TEXT("WHISPER CONTEXT IS A NULL POINTER"));
    }
    else
    {
        UE_LOG(LogTemp, Warning, TEXT("WHISPER CONTEXT IS NOT A NULL POINTER"));
    }
    
    // wait for 1 second to avoid any buffered noise
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    audio.clear();

    int  ret_val = 0;

    //----------------------GENERAL TRANSCRIPTION---------------------//

    bool is_running = true;
    bool have_prompt = false;
    bool ask_prompt = true;

    float prob0 = 0.0f;
    float prob = 0.0f;

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    const std::string k_prompt = "Okay Whisper";

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: general-purpose mode\n", __func__);

    UE_LOG(LogTemp, Warning, TEXT("[START SPEAKING]"));


	//int counter = 0;
	while (is_running) {
        is_running = sdl_poll_events();
		//ClientMessage(FString("Inside while"));
		//counter = counter +1;
        //UE_LOG(LogTemp, Warning, TEXT("%d"), params.capture_id);
		//UE_LOG(LogTemp, Warning, TEXT("%s"), *FString::FromInt(counter));


        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (ask_prompt) {
            fprintf(stdout, "\n");
            fprintf(stdout, "%s: Say the following phrase: '%s%s%s'\n", __func__, "\033[1m", k_prompt.c_str(), "\033[0m");
            UE_LOG(LogTemp, Warning, TEXT("Say the following phrase %s"), *FString(k_prompt.c_str()));
            fprintf(stdout, "\n");

            ask_prompt = false;
        }

        {
            audio.get(2000, pcmf32_cur);

            if (FMySpeechWorker::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, params.print_energy))
            {
                fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);
                UE_LOG(LogTemp, Warning, TEXT("Speech detected! Processing"));
                int64_t t_ms = 0;
                if (!have_prompt) {
                    // wait for activation phrase
                    audio.get(params.prompt_ms, pcmf32_cur);

                    const auto txt = transcribe(ctx, params, pcmf32_cur, prob0, t_ms);

                    fprintf(stdout, "%s: Heard '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", txt.c_str(), "\033[0m", (int)t_ms);
                    UE_LOG(LogTemp, Warning, TEXT("Heard %s"), *FString(txt.c_str()));

                    const float sim = similarity(txt, k_prompt);

                    if (txt.length() < 0.8 * k_prompt.length() || txt.length() > 1.2 * k_prompt.length() || sim < 0.8f) {
                        fprintf(stdout, "%s: WARNING: prompt not recognized, try again\n", __func__);
                        UE_LOG(LogTemp, Warning, TEXT("WARNING: prompt not recognized, try again"));
                        ask_prompt = true;
                    }
                    else {
                        fprintf(stdout, "\n");
                        fprintf(stdout, "%s: The prompt has been recognized!\n", __func__);
                        UE_LOG(LogTemp, Warning, TEXT("The prompt has been recognized!"));
                        fprintf(stdout, "%s: Waiting for voice commands ...\n", __func__);
                        UE_LOG(LogTemp, Warning, TEXT("Waiting for voice commands"));
                        fprintf(stdout, "\n");

                        // save the audio for the prompt
                        pcmf32_prompt = pcmf32_cur;
                        have_prompt = true;
                    }
                }
                else
                {
                    // we have heard the activation phrase, now detect the commands
                    audio.get(params.command_ms, pcmf32_cur);

                    // prepend the prompt audio
                    pcmf32_cur.insert(pcmf32_cur.begin(), pcmf32_prompt.begin(), pcmf32_prompt.end());

                    const auto txt = transcribe(ctx, params, pcmf32_cur, prob, t_ms);

                    prob = 100.0f * (prob - prob0);

                    // find the prompt in the text
                    float best_sim = 0.0f;
                    size_t best_len = 0;
                    for (int n = 0.8 * k_prompt.size(); n <= 1.2 * k_prompt.size(); ++n) {
                        const auto prompt = txt.substr(0, n);

                        const float sim = similarity(prompt, k_prompt);

                        //fprintf(stderr, "%s: prompt = '%s', sim = %f\n", __func__, prompt.c_str(), sim);

                        if (sim > best_sim) {
                            best_sim = sim;
                            best_len = n;
                        }
                    }

                    const std::string command = txt.substr(best_len);

                    fprintf(stdout, "%s: Command '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", command.c_str(), "\033[0m", (int)t_ms);
                    fprintf(stdout, "\n");
                }

                audio.clear();
            }

        }
        

        if (resumes)
        {
            {
                //whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
                //UE_LOG(LogTemp, Warning, TEXT("WHISPER HEADER WORKING"));
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