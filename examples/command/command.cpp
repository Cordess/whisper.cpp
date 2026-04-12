// Voice assistant example
//
// Speak short text commands to the microphone.
// This program will detect your voice command and convert them to text.
//
// ref: https://github.com/ggml-org/whisper.cpp/issues/171
//

#include "common-sdl.h"
#include "common.h"
#include "whisper.h"
#include "grammar-parser.h"
#include <chrono>
#include <sstream>
#include <cassert>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <map>
// Ensure the CPR library is included properly
#include <cpr/cpr.h>
#include <common-whisper.h>

// command-line parameters
struct whisper_params {
    int32_t n_threads  = std::fmin(4, (int32_t) std::thread::hardware_concurrency());
    int32_t prompt_ms  = 5000;
    int32_t command_ms = 8000;
    int32_t capture_id = -1;
    int32_t max_tokens = 32;
    int32_t audio_ctx  = 0;

    float vad_thold  = 0.6f;
    float freq_thold = 100.0f;

    float grammar_penalty = 100.0f;

    grammar_parser::parse_state grammar_parsed;

    bool translate     = false;
    bool print_special = false;
    bool print_energy  = false;
    bool no_timestamps = true;
    bool use_gpu       = true;
    bool flash_attn    = true;

    std::string language  = "en";
    std::string model     = "models/ggml-base.en.bin";
    std::string fname_out;
    std::string commands;
    std::string prompt;
    std::string context;
    std::string grammar;

    // A regular expression that matches tokens to suppress
    std::string suppress_regex;
};

void whisper_print_usage(int argc, char ** argv, const whisper_params & params);
void send_discord_webhook(const std::string& text);

static std::string WhisperFullCtxzuText(whisper_context* ctx, float& logprob_min, float& logprob_sum, int& n_tokens);

static bool whisper_params_parse(int argc, char ** argv, whisper_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
        else if (arg == "-t"     || arg == "--threads")       { params.n_threads     = std::stoi(argv[++i]); }
        else if (arg == "-pms"   || arg == "--prompt-ms")     { params.prompt_ms     = std::stoi(argv[++i]); }
        else if (arg == "-cms"   || arg == "--command-ms")    { params.command_ms    = std::stoi(argv[++i]); }
        else if (arg == "-c"     || arg == "--capture")       { params.capture_id    = std::stoi(argv[++i]); }
        else if (arg == "-mt"    || arg == "--max-tokens")    { params.max_tokens    = std::stoi(argv[++i]); }
        else if (arg == "-ac"    || arg == "--audio-ctx")     { params.audio_ctx     = std::stoi(argv[++i]); }
        else if (arg == "-vth"   || arg == "--vad-thold")     { params.vad_thold     = std::stof(argv[++i]); }
        else if (arg == "-fth"   || arg == "--freq-thold")    { params.freq_thold    = std::stof(argv[++i]); }
        else if (arg == "-tr"    || arg == "--translate")     { params.translate     = true; }
        else if (arg == "-ps"    || arg == "--print-special") { params.print_special = true; }
        else if (arg == "-pe"    || arg == "--print-energy")  { params.print_energy  = true; }
        else if (arg == "-ng"    || arg == "--no-gpu")        { params.use_gpu       = false; }
        else if (arg == "-fa"    || arg == "--flash-attn")    { params.flash_attn    = true; }
        else if (arg == "-nfa"   || arg == "--no-flash-attn") { params.flash_attn    = false; }
        else if (arg == "-l"     || arg == "--language")      { params.language      = argv[++i]; }
        else if (arg == "-m"     || arg == "--model")         { params.model         = argv[++i]; }
        else if (arg == "-f"     || arg == "--file")          { params.fname_out     = argv[++i]; }
        else if (arg == "-cmd"   || arg == "--commands")      { params.commands      = argv[++i]; }
        else if (arg == "-p"     || arg == "--prompt")        { params.prompt        = argv[++i]; }
        else if (arg == "-ctx"   || arg == "--context")       { params.context       = argv[++i]; }
        else if (                   arg == "--grammar")       { params.grammar       = argv[++i]; }
        else if (                   arg == "--grammar-penalty") { params.grammar_penalty = std::stof(argv[++i]); }
        else if (                   arg == "--suppress-regex") { params.suppress_regex = argv[++i]; }
        else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            whisper_print_usage(argc, argv, params);
            exit(0);
        }
    }

    return true;
}

void whisper_print_usage(int /*argc*/, char ** argv, const whisper_params & params) {
    fprintf(stderr, "\n");
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h,         --help           [default] show this help message and exit\n");
    fprintf(stderr, "  -t N,       --threads N      [%-7d] number of threads to use during computation\n", params.n_threads);
    fprintf(stderr, "  -pms N,     --prompt-ms N    [%-7d] prompt duration in milliseconds\n",             params.prompt_ms);
    fprintf(stderr, "  -cms N,     --command-ms N   [%-7d] command duration in milliseconds\n",            params.command_ms);
    fprintf(stderr, "  -c ID,      --capture ID     [%-7d] capture device ID\n",                           params.capture_id);
    fprintf(stderr, "  -mt N,      --max-tokens N   [%-7d] maximum number of tokens per audio chunk\n",    params.max_tokens);
    fprintf(stderr, "  -ac N,      --audio-ctx N    [%-7d] audio context size (0 - all)\n",                params.audio_ctx);
    fprintf(stderr, "  -vth N,     --vad-thold N    [%-7.2f] voice activity detection threshold\n",        params.vad_thold);
    fprintf(stderr, "  -fth N,     --freq-thold N   [%-7.2f] high-pass frequency cutoff\n",                params.freq_thold);
    fprintf(stderr, "  -tr,        --translate      [%-7s] translate from source language to english\n",   params.translate ? "true" : "false");
    fprintf(stderr, "  -ps,        --print-special  [%-7s] print special tokens\n",                        params.print_special ? "true" : "false");
    fprintf(stderr, "  -pe,        --print-energy   [%-7s] print sound energy (for debugging)\n",          params.print_energy ? "true" : "false");
    fprintf(stderr, "  -ng,        --no-gpu         [%-7s] disable GPU\n",                                 params.use_gpu ? "false" : "true");
    fprintf(stderr, "  -fa,        --flash-attn     [%-7s] enbale flash attention\n",                      params.flash_attn ? "true" : "false");
    fprintf(stderr, "  -nfa,       --no-flash-attn  [%-7s] disable flash attention\n",                     params.flash_attn ? "false" : "true");
    fprintf(stderr, "  -l LANG,    --language LANG  [%-7s] spoken language\n",                             params.language.c_str());
    fprintf(stderr, "  -m FNAME,   --model FNAME    [%-7s] model path\n",                                  params.model.c_str());
    fprintf(stderr, "  -f FNAME,   --file FNAME     [%-7s] text output file name\n",                       params.fname_out.c_str());
    fprintf(stderr, "  -cmd FNAME, --commands FNAME [%-7s] text file with allowed commands\n",             params.commands.c_str());
    fprintf(stderr, "  -p,         --prompt         [%-7s] the required activation prompt\n",              params.prompt.c_str());
    fprintf(stderr, "  -ctx,       --context        [%-7s] sample text to help the transcription\n",       params.context.c_str());
    fprintf(stderr, "  --grammar GRAMMAR            [%-7s] GBNF grammar to guide decoding\n",              params.grammar.c_str());
    fprintf(stderr, "  --grammar-penalty N          [%-7.1f] scales down logits of nongrammar tokens\n",   params.grammar_penalty);
    fprintf(stderr, "  --suppress-regex REGEX       [%-7s] regular expression matching tokens to suppress\n", params.suppress_regex.c_str());
    fprintf(stderr, "\n");
}

static std::string transcribe(
                 whisper_context * ctx,
            const whisper_params & params,
        const std::vector<float> & pcmf32,
               const std::string & grammar_rule,
                           float & logprob_min,
                           float & logprob_sum,
                             int & n_tokens,
                         int64_t & t_ms) {
    const auto t_start = std::chrono::high_resolution_clock::now();

    logprob_min = 0.0f;
    logprob_sum = 0.0f;
    n_tokens    = 0;
    t_ms = 0;

    //whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

    wparams.print_progress   = false;
    wparams.print_special    = params.print_special;
    wparams.print_realtime   = false;
    wparams.print_timestamps = !params.no_timestamps;
    wparams.translate        = params.translate;
    wparams.no_context       = true;
    wparams.no_timestamps    = params.no_timestamps;
    wparams.single_segment   = true;
    wparams.max_tokens       = params.max_tokens;
    wparams.language         = params.language.c_str();
    wparams.n_threads        = params.n_threads;

    wparams.audio_ctx = params.audio_ctx;

    wparams.temperature     = 0.4f;
    wparams.temperature_inc = 1.0f;
    wparams.greedy.best_of  = 5;

    wparams.beam_search.beam_size = 5;

    wparams.initial_prompt = params.context.data();

    wparams.suppress_regex = params.suppress_regex.c_str();

    const auto & grammar_parsed = params.grammar_parsed;
    auto grammar_rules = grammar_parsed.c_rules();

    if (!params.grammar_parsed.rules.empty() && !grammar_rule.empty()) {
        if (grammar_parsed.symbol_ids.find(grammar_rule) == grammar_parsed.symbol_ids.end()) {
            fprintf(stderr, "%s: warning: grammar rule '%s' not found - skipping grammar sampling\n", __func__, grammar_rule.c_str());
        } else {
            wparams.grammar_rules   = grammar_rules.data();
            wparams.n_grammar_rules = grammar_rules.size();
            wparams.i_start_rule    = grammar_parsed.symbol_ids.at(grammar_rule);
            wparams.grammar_penalty = params.grammar_penalty;
        }
    }

    if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
        return "";
    }

    std::string result;
    result = WhisperFullCtxzuText(ctx, logprob_min, logprob_sum, n_tokens);

    //result = _result;
    const auto t_end = std::chrono::high_resolution_clock::now();
    t_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();

    return result;
}

static std::string WhisperFullCtxzuText(whisper_context* ctx, float& logprob_min, float& logprob_sum, int& n_tokens)
{
    std::string _result;

    const int n_segments = whisper_full_n_segments(ctx);
    for (int i = 0; i < n_segments; ++i) {
        const char* text = whisper_full_get_segment_text(ctx, i);

        _result += text;

        const int n = whisper_full_n_tokens(ctx, i);
        for (int j = 0; j < n; ++j) {
            const auto token = whisper_full_get_token_data(ctx, i, j);

            if (token.plog > 0.0f) exit(0);
            logprob_min = std::fmin(logprob_min, token.plog);
            logprob_sum += token.plog;
            ++n_tokens;
        }
    }
    return _result;
}

static std::vector<std::string> read_allowed_commands(const std::string & fname) {
    std::vector<std::string> allowed_commands;

    std::ifstream ifs(fname);
    if (!ifs.is_open()) {
        return allowed_commands;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        line = ::trim(line);
        if (line.empty()) {
            continue;
        }

        std::transform(line.begin(), line.end(),line.begin(), ::tolower);
        allowed_commands.push_back(std::move(line));
    }

    return allowed_commands;
}

static std::vector<std::string> get_words(const std::string &txt) {
    std::vector<std::string> words;

    std::istringstream iss(txt);
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

// command-list mode
// guide the transcription to match the most likely command from a provided list
static int process_command_list(struct whisper_context * ctx, audio_async &audio, const whisper_params &params, std::ofstream &fout) {
    fprintf(stderr, "\n");
    fprintf(stderr, "%s: guided mode\n", __func__);

    std::vector<std::string> allowed_commands = read_allowed_commands(params.commands);

    if (allowed_commands.empty()) {
        fprintf(stderr, "%s: error: failed to read allowed commands from '%s'\n", __func__, params.commands.c_str());
        return 2;
    }

    int max_len = 0;

    std::vector<std::vector<whisper_token>> allowed_tokens;

    for (const auto & cmd : allowed_commands) {
        whisper_token tokens[1024];
        allowed_tokens.emplace_back();

        for (int l = 0; l < (int) cmd.size(); ++l) {
            // NOTE: very important to add the whitespace !
            //       the reason is that the first decoded token starts with a whitespace too!
            std::string ss = std::string(" ") + cmd.substr(0, l + 1);

            const int n = whisper_tokenize(ctx, ss.c_str(), tokens, 1024);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to tokenize command '%s'\n", __func__, cmd.c_str());
                return 3;
            }

            if (n == 1) {
                allowed_tokens.back().push_back(tokens[0]);
            }
        }

        max_len = std::fmax(max_len, (int) cmd.size());
    }

    fprintf(stderr, "%s: allowed commands [ tokens ]:\n", __func__);
    fprintf(stderr, "\n");
    for (int i = 0; i < (int) allowed_commands.size(); ++i) {
        fprintf(stderr, "  - \033[1m%-*s\033[0m = [", max_len, allowed_commands[i].c_str());
        for (const auto & token : allowed_tokens[i]) {
            fprintf(stderr, " %5d", token);
        }
        fprintf(stderr, " ]\n");
    }

    std::string k_prompt = "select one from the available words: ";
    for (int i = 0; i < (int) allowed_commands.size(); ++i) {
        if (i > 0) {
            k_prompt += ", ";
        }
        k_prompt += allowed_commands[i];
    }
    k_prompt += ". selected word: ";

    // tokenize prompt
    std::vector<whisper_token> k_tokens;
    {
        k_tokens.resize(1024);
        const int n = whisper_tokenize(ctx, k_prompt.c_str(), k_tokens.data(), 1024);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to tokenize prompt '%s'\n", __func__, k_prompt.c_str());
            return 4;
        }
        k_tokens.resize(n);
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: prompt: '%s'\n", __func__, k_prompt.c_str());
    fprintf(stderr, "%s: tokens: [", __func__);
    for (const auto & token : k_tokens) {
        fprintf(stderr, " %d", token);
    }
    fprintf(stderr, " ]\n");

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: listening for a command ...\n", __func__);
    fprintf(stderr, "\n");

    bool is_running  = true;

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    // main loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        audio.get(2000, pcmf32_cur);

        if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, params.print_energy)) {
            fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

            const auto t_start = std::chrono::high_resolution_clock::now();

            whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);

            wparams.print_progress   = false;
            wparams.print_special    = params.print_special;
            wparams.print_realtime   = false;
            wparams.print_timestamps = !params.no_timestamps;
            wparams.translate        = params.translate;
            wparams.no_context       = true;
            wparams.single_segment   = true;
            wparams.max_tokens       = 1;
            wparams.language         = params.language.c_str();
            wparams.n_threads        = params.n_threads;

            wparams.audio_ctx        = params.audio_ctx;

            wparams.prompt_tokens    = k_tokens.data();
            wparams.prompt_n_tokens  = k_tokens.size();

            // run the transformer and a single decoding pass
            if (whisper_full(ctx, wparams, pcmf32_cur.data(), pcmf32_cur.size()) != 0) {
                fprintf(stderr, "%s: ERROR: whisper_full() failed\n", __func__);
                break;
            }

            // estimate command probability
            // NOTE: not optimal
            {
                const auto * logits = whisper_get_logits(ctx);

                std::vector<float> probs(whisper_n_vocab(ctx), 0.0f);

                // compute probs from logits via softmax
                {
                    float max = -1e9;
                    for (int i = 0; i < (int) probs.size(); ++i) {
                        max = std::fmax(max, logits[i]);
                    }

                    float sum = 0.0f;
                    for (int i = 0; i < (int) probs.size(); ++i) {
                        probs[i] = expf(logits[i] - max);
                        sum += probs[i];
                    }

                    for (int i = 0; i < (int) probs.size(); ++i) {
                        probs[i] /= sum;
                    }
                }

                std::vector<std::pair<float, int>> probs_id;

                double psum = 0.0;
                for (int i = 0; i < (int) allowed_commands.size(); ++i) {
                    probs_id.emplace_back(probs[allowed_tokens[i][0]], i);
                    for (int j = 1; j < (int) allowed_tokens[i].size(); ++j) {
                        probs_id.back().first += probs[allowed_tokens[i][j]];
                    }
                    probs_id.back().first /= allowed_tokens[i].size();
                    psum += probs_id.back().first;
                }

                // normalize
                for (auto & p : probs_id) {
                    p.first /= psum;
                }

                // sort descending
                {
                    using pair_type = decltype(probs_id)::value_type;
                    std::sort(probs_id.begin(), probs_id.end(), [](const pair_type & a, const pair_type & b) {
                        return a.first > b.first;
                    });
                }

                // print the commands and the respective probabilities
                {
                    fprintf(stdout, "\n");
                    for (const auto & cmd : probs_id) {
                        fprintf(stdout, "%s: %s%-*s%s = %f | ", __func__, "\033[1m", max_len, allowed_commands[cmd.second].c_str(), "\033[0m", cmd.first);
                        for (int token : allowed_tokens[cmd.second]) {
                            fprintf(stdout, "'%4s' %f ", whisper_token_to_str(ctx, token), probs[token]);
                        }
                        fprintf(stdout, "\n");
                    }
                }

                // best command
                {
                    const auto t_end = std::chrono::high_resolution_clock::now();

                    const float prob = probs_id[0].first;
                    const int index = probs_id[0].second;
                    const char * best_command = allowed_commands[index].c_str();

                    fprintf(stdout, "\n");
                    fprintf(stdout, "%s: detected command: %s%s%s | p = %f | t = %d ms\n", __func__,
                            "\033[1m", best_command, "\033[0m", prob,
                            (int) std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
                    fprintf(stdout, "\n");
                    //if(allowed_commands[index].compare("discord") == 0)
				    {
						float logprob_min0 = 0.0f;
						float logprob_sum0 = 0.0f;
						int   n_tokens0 = 0;
						int64_t t_ms = 0;
                        const auto txt = ::trim(::transcribe(ctx, params, pcmf32_cur, "prompt", logprob_min0, logprob_sum0, n_tokens0, t_ms));
                        const float p = 100.0f * std::exp(logprob_min0);
                        fprintf(stdout, "%s: Heard '%s%s%s', (t = %d ms, p = %.2f%%)\n", __func__, "\033[1m", txt.c_str(), "\033[0m", (int)t_ms, p);
						//add code to send a POST request to a discord webhook with cpr
                        send_discord_webhook(txt);
				    }
                }
            }

            audio.clear();
        }
    }

    return 0;
}

void send_discord_webhook(const std::string& text) {
    std::string url = "https://discord.com/api/webhooks/1415268710926520380/hQ46kU6fYGY6ycTiTdPeK3IvvnfwkXox-5Yt7xHxL4R9919qEStZSiPAc2SIwNyRvE5Z";
    // JSON-Body dynamisch mit Variable text erzeugen
    std::string payload =
        "{\n"
        "  \"content\": \"" + text + "\",\n"
        "  \"username\": \"Ignore-Me\"\n"
        "}";

    cpr::Response r = cpr::Post(
        cpr::Url{ url },
        cpr::Header{ {"Content-Type", "application/json"} },
        cpr::Body{ payload }
    );

    fprintf(stdout, "Status: %d\n", r.status_code);
    fprintf(stdout, "Body: %s\n", r.text.c_str());
}

// always-prompt mode
// transcribe the voice into text after valid prompt
static int always_prompt_transcription(struct whisper_context * ctx, audio_async & audio, const whisper_params & params, std::ofstream & fout) {
    bool is_running = true;
    bool ask_prompt = true;

    float logprob_min = 0.0f;
    float logprob_sum = 0.0f;
    int   n_tokens    = 0;

    std::vector<float> pcmf32_cur;

    const std::string k_prompt = params.prompt;

    const int k_prompt_length = get_words(k_prompt).size();

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: always-prompt mode\n", __func__);

    // main loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (ask_prompt) {
            fprintf(stdout, "\n");
            fprintf(stdout, "%s: The prompt is: '%s%s%s'\n", __func__, "\033[1m", k_prompt.c_str(), "\033[0m");
            fprintf(stdout, "\n");

            ask_prompt = false;
        }

        {
            audio.get(2000, pcmf32_cur);

            if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, params.print_energy)) {
                fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

                int64_t t_ms = 0;

                // detect the commands
                audio.get(params.command_ms, pcmf32_cur);

                const auto txt = ::trim(::transcribe(ctx, params, pcmf32_cur, "", logprob_min, logprob_sum, n_tokens, t_ms));

                const auto words = get_words(txt);

                std::string prompt;
                std::string command;

                for (int i = 0; i < (int) words.size(); ++i) {
                    if (i < k_prompt_length) {
                        prompt += words[i] + " ";
                    } else {
                        command += words[i] + " ";
                    }
                }

                const float sim = similarity(prompt, k_prompt);

                //debug
                //fprintf(stdout, "command size: %i\n", command_length);

                if ((sim > 0.7f) && (command.size() > 0)) {
                    fprintf(stdout, "%s: Command '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", command.c_str(), "\033[0m", (int) t_ms);
                    if (fout.is_open()) {
                        fout << command << std::endl;
                    }
                }

                fprintf(stdout, "\n");

                audio.clear();
            }
        }
    }

    return 0;
}

// general-purpose mode
// freely transcribe the voice into text
static int process_general_transcription(struct whisper_context * ctx, audio_async & audio, const whisper_params & params, std::ofstream & fout) {
    bool is_running  = true;
    bool have_prompt = false;
    bool ask_prompt  = true;

    float logprob_min0 = 0.0f;
    float logprob_min  = 0.0f;

    float logprob_sum0 = 0.0f;
    float logprob_sum  = 0.0f;

    int n_tokens0 = 0;
    int n_tokens  = 0;

    std::vector<float> pcmf32_cur;
    std::vector<float> pcmf32_prompt;

    std::string k_prompt = "Nachricht senden.";
    if (!params.prompt.empty()) {
        k_prompt = params.prompt;
    }

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: general-purpose mode\n", __func__);

    // main loop
    while (is_running) {
        // handle Ctrl + C
        is_running = sdl_poll_events();

        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (ask_prompt) {
            fprintf(stdout, "\n");
            fprintf(stdout, "%s: TEST CMAKE Say the following phrase: '%s%s%s'\n", __func__, "\033[1m", k_prompt.c_str(), "\033[0m");
            fprintf(stdout, "\n");

            ask_prompt = false;
        }

        {
            audio.get(2000, pcmf32_cur);

            if (::vad_simple(pcmf32_cur, WHISPER_SAMPLE_RATE, 1000, params.vad_thold, params.freq_thold, params.print_energy)) {
                fprintf(stdout, "%s: Speech detected! Processing ...\n", __func__);

                int64_t t_ms = 0;

                if (!have_prompt) {
                    // wait for activation phrase
                    audio.get(params.prompt_ms, pcmf32_cur);

                    const auto txt = ::trim(::transcribe(ctx, params, pcmf32_cur, "prompt", logprob_min0, logprob_sum0, n_tokens0, t_ms));

                    const float p = 100.0f * std::exp(logprob_min0);

                    fprintf(stdout, "%s: Heard '%s%s%s', (t = %d ms, p = %.2f%%)\n", __func__, "\033[1m", txt.c_str(), "\033[0m", (int) t_ms, p);

                    const float sim = similarity(txt, k_prompt);

                    if (txt.length() < 0.8*k_prompt.length() || txt.length() > 1.2*k_prompt.length() || sim < 0.8f) {
                        fprintf(stdout, "%s: WARNING: prompt not recognized, try again\n", __func__);
                        ask_prompt = true;
                    } else {
                        fprintf(stdout, "\n");
                        fprintf(stdout, "%s: The prompt has been recognized!\n", __func__);
                        fprintf(stdout, "%s: Waiting for voice commands ...\n", __func__);
                        fprintf(stdout, "\n");

                        // save the audio for the prompt
                        pcmf32_prompt = pcmf32_cur;
                        have_prompt = true;
                    }
                } else {
                    // we have heard the activation phrase, now detect the commands
                    audio.get(params.command_ms, pcmf32_cur);

                    //printf("len prompt:  %.4f\n", pcmf32_prompt.size() / (float) WHISPER_SAMPLE_RATE);
                    //printf("len command: %.4f\n", pcmf32_cur.size() / (float) WHISPER_SAMPLE_RATE);

                    // prepend 3 second of silence
                    pcmf32_cur.insert(pcmf32_cur.begin(), 3.0f*WHISPER_SAMPLE_RATE, 0.0f);

                    // prepend the prompt audio
                    pcmf32_cur.insert(pcmf32_cur.begin(), pcmf32_prompt.begin(), pcmf32_prompt.end());

                    const auto txt = ::trim(::transcribe(ctx, params, pcmf32_cur, "root", logprob_min, logprob_sum, n_tokens, t_ms));

                    //const float p = 100.0f * std::exp((logprob - logprob0) / (n_tokens - n_tokens0));
                    const float p = 100.0f * std::exp(logprob_min);

                    //fprintf(stdout, "%s: heard '%s'\n", __func__, txt.c_str());

                    // find the prompt in the text
                    float best_sim = 0.0f;
                    size_t best_len = 0;
                    for (size_t n = 0.8*k_prompt.size(); n <= 1.2*k_prompt.size(); ++n) {
                        if (n >= txt.size()) {
                            break;
                        }

                        const auto prompt = txt.substr(0, n);

                        const float sim = similarity(prompt, k_prompt);

                        //fprintf(stderr, "%s: prompt = '%s', sim = %f\n", __func__, prompt.c_str(), sim);

                        if (sim > best_sim) {
                            best_sim = sim;
                            best_len = n;
                        }
                    }

                    fprintf(stdout, "%s:   DEBUG: txt = '%s', prob = %.2f%%\n", __func__, txt.c_str(), p);
                    if (best_len == 0) {
                        fprintf(stdout, "%s: WARNING: command not recognized, try again\n", __func__);
                    } else {
                        // cut the prompt from the decoded text
                        const std::string command = ::trim(txt.substr(best_len));
                        fprintf(stdout, "%s: Command '%s%s%s', (t = %d ms)\n", __func__, "\033[1m", command.c_str(), "\033[0m", (int) t_ms);
                        send_discord_webhook(command.c_str());
                    }

                    fprintf(stdout, "\n");
                }

                audio.clear();
            }
        }
    }

    return 0;
}

// segmented transcription mode
// splits audio from WAV files into non-overlapping sequential chunks and transcribes each
// segment individually, inspired by the chunked approach in whisper-stream.
// Uses prompt token carry-forward between segments for context continuity without audio overlap.
// This produces better results for longer audio files by keeping each chunk within whisper's optimal window.
static int process_segmented_transcription_from_file(struct whisper_context* ctx, const whisper_params& params, std::ofstream& fout) {
    bool is_running = true;

    const int segment_ms = 10000;  // segment length in ms (each chunk sent to whisper)

    const int n_samples_segment = (int)((1e-3 * segment_ms) * WHISPER_SAMPLE_RATE);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: segmented transcription mode (file input)\n", __func__);
    fprintf(stderr, "%s: segment = %d ms (%d samples)\n", __func__, segment_ms, n_samples_segment);

    // Get WAV files from directory
    std::string wav_directory = "C:\\Users\\Cordess\\source\\repos\\Cordess\\AvnAudio\\AvnAudioSignalRDemo\\Server\\Files";
    if (wav_directory.empty()) {
        fprintf(stderr, "%s: ERROR: no directory path specified\n", __func__);
        return 1;
    }

    std::vector<std::string> wav_files;

#ifdef _WIN32
    WIN32_FIND_DATAA find_data;
    std::string search_path = wav_directory;
    if (search_path.back() != '\\' && search_path.back() != '/') {
        search_path += "\\";
    }
    search_path += "*.wav";

    HANDLE hFind = FindFirstFileA(search_path.c_str(), &find_data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                std::string filename = wav_directory;
                if (filename.back() != '\\' && filename.back() != '/') {
                    filename += "\\";
                }
                filename += find_data.cFileName;
                wav_files.push_back(filename);
            }
        } while (FindNextFileA(hFind, &find_data) != 0);
        FindClose(hFind);
    }
#else
    DIR* dir = opendir(wav_directory.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename(entry->d_name);
            if (filename.length() > 4) {
                std::string ext = filename.substr(filename.length() - 4);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".wav") {
                    std::string full_path = wav_directory;
                    if (full_path.back() != '/') {
                        full_path += "/";
                    }
                    full_path += filename;
                    wav_files.push_back(full_path);
                }
            }
        }
        closedir(dir);
    }
#endif

    if (wav_files.empty()) {
        fprintf(stderr, "%s: ERROR: no WAV files found in directory '%s'\n", __func__, wav_directory.c_str());
        return 1;
    }

    std::sort(wav_files.begin(), wav_files.end());

    fprintf(stderr, "%s: found %zu WAV files in '%s'\n", __func__, wav_files.size(), wav_directory.c_str());
    for (size_t i = 0; i < wav_files.size(); ++i) {
        fprintf(stderr, "  %3zu. %s\n", i + 1, wav_files[i].c_str());
    }
    fprintf(stderr, "\n");

    for (size_t file_idx = 0; file_idx < wav_files.size() && is_running; ++file_idx) {
        const std::string& wav_file = wav_files[file_idx];

        fprintf(stdout, "\n");
        fprintf(stdout, "%s: ======================================\n", __func__);
        fprintf(stdout, "%s: Processing file %zu/%zu\n", __func__, file_idx + 1, wav_files.size());
        fprintf(stdout, "%s: File: %s\n", __func__, wav_file.c_str());
        fprintf(stdout, "%s: ======================================\n", __func__);

        // Load the entire WAV file
        std::vector<float> pcmf32_all;
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(wav_file, pcmf32_all, pcmf32s, false)) {
            fprintf(stderr, "%s: ERROR: failed to load WAV file '%s'\n", __func__, wav_file.c_str());
            if (fout.is_open()) {
                fout << wav_file << " : [ERROR: Failed to load file]" << std::endl;
            }
            continue;
        }

        const int n_samples_total = (int)pcmf32_all.size();
        fprintf(stdout, "%s: Loaded %d samples (%.2f seconds)\n",
                __func__, n_samples_total, (float)n_samples_total / WHISPER_SAMPLE_RATE);

        // If the audio is short enough, transcribe it in one go (no segmentation needed)
        if (n_samples_total <= n_samples_segment) {
            fprintf(stdout, "%s: Audio is short enough for single-pass transcription\n", __func__);

            float logprob_min = 0.0f;
            float logprob_sum = 0.0f;
            int   n_tokens    = 0;
            int64_t t_ms      = 0;

            const auto txt = ::trim(::transcribe(ctx, params, pcmf32_all, "", logprob_min, logprob_sum, n_tokens, t_ms));

            fprintf(stdout, "%s: '%s%s%s' (t = %d ms)\n", __func__, "\033[1m", txt.c_str(), "\033[0m", (int)t_ms);

            if (fout.is_open()) {
                fout << wav_file << " : " << txt << std::endl;
                fout.flush();
            }
            continue;
        }

        // Split audio into non-overlapping sequential chunks (inspired by whisper-stream)
        const int n_chunks = (n_samples_total + n_samples_segment - 1) / n_samples_segment;
        fprintf(stdout, "%s: Splitting audio into %d chunks of %dms each\n",
                __func__, n_chunks, segment_ms);

        std::string full_transcription;
        int pos = 0; // current position in the audio (in samples)
        int n_iter = 0;

        while (pos < n_samples_total) {
            // Take the next chunk — no overlap with previous chunk
            const int n_samples_chunk = std::fmin(n_samples_segment, n_samples_total - pos);

            std::vector<float> pcmf32(pcmf32_all.begin() + pos, pcmf32_all.begin() + pos + n_samples_chunk);

            pos += n_samples_chunk;

            // Run whisper inference on this chunk
            {
                whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

                wparams.print_progress   = false;
                wparams.print_special    = params.print_special;
                wparams.print_realtime   = false;
                wparams.print_timestamps = !params.no_timestamps;
                wparams.translate        = params.translate;
                wparams.no_context       = true;
                wparams.no_timestamps    = true;
                wparams.single_segment   = false;
                wparams.max_tokens       = 0; // unlimited tokens per chunk (params.max_tokens=32 is too low for 10s segments)
                wparams.language         = params.language.c_str();
                wparams.n_threads        = params.n_threads;
                wparams.audio_ctx        = params.audio_ctx;
                wparams.temperature      = 0.4f;
                wparams.temperature_inc  = 1.0f;
                wparams.greedy.best_of   = 5;
                wparams.beam_search.beam_size = 5;
                wparams.initial_prompt   = params.context.data();
                wparams.suppress_regex   = params.suppress_regex.c_str();

                if (whisper_full(ctx, wparams, pcmf32.data(), pcmf32.size()) != 0) {
                    fprintf(stderr, "%s: WARNING: failed to transcribe chunk %d\n", __func__, n_iter);
                    ++n_iter;
                    continue;
                }

                // Extract text from all segments returned by whisper for this chunk
                const int n_segments = whisper_full_n_segments(ctx);
                fprintf(stdout, "%s: [chunk %d] whisper returned %d segments, audio samples = %d\n",
                        __func__, n_iter, n_segments, (int)pcmf32.size());

                for (int i = 0; i < n_segments; ++i) {
                    const char* text = whisper_full_get_segment_text(ctx, i);
                    fprintf(stdout, "%s: [chunk %d][seg %d] '%s'\n", __func__, n_iter, i, text);
                    full_transcription += text;
                }
            }

            fprintf(stdout, "%s: [chunk %d/%d] pos = %.2f / %.2f sec\n",
                    __func__, n_iter + 1, n_chunks,
                    (float)pos / WHISPER_SAMPLE_RATE,
                    (float)n_samples_total / WHISPER_SAMPLE_RATE);

            ++n_iter;
        }

        // Trim and output the full transcription for this file
        const auto txt = ::trim(full_transcription);

        fprintf(stdout, "\n%s: === Full transcription ===\n", __func__);
        fprintf(stdout, "%s: '%s%s%s'\n", __func__, "\033[1m", txt.c_str(), "\033[0m");
        fprintf(stdout, "%s: === End (%d segments) ===\n\n", __func__, n_iter);

        if (fout.is_open()) {
            fout << wav_file << " : " << txt << std::endl;
            fout.flush();
        }
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "%s: ======================================\n", __func__);
    fprintf(stdout, "%s: Completed processing %zu files\n", __func__, wav_files.size());
    fprintf(stdout, "%s: ======================================\n", __func__);

    return 0;
}

// general-purpose mode
// freely transcribe the voice of a file into text
static int process_general_transcription_from_file(struct whisper_context* ctx, const whisper_params& params, std::ofstream& fout) {
    bool is_running = true;

    float logprob_min = 0.0f;
    float logprob_sum = 0.0f;
    int n_tokens = 0;

    std::vector<float> pcmf32_cur;

    fprintf(stderr, "\n");
    fprintf(stderr, "%s: general-purpose mode (file input)\n", __func__);

    // Get the directory path from params.fname_out
    std::string wav_directory = "C:\\Users\\Cordess\\source\\repos\\Cordess\\AvnAudio\\AvnAudioSignalRDemo\\Server\\Files";
    if (wav_directory.empty()) {
        fprintf(stderr, "%s: ERROR: no directory path specified\n", __func__);
        return 1;
    }

    std::vector<std::string> wav_files;

    // Platform-specific directory listing for C++14
#ifdef _WIN32
    // Windows implementation using FindFirstFile/FindNextFile
    WIN32_FIND_DATAA find_data;
    std::string search_path = wav_directory;
    if (search_path.back() != '\\' && search_path.back() != '/') {
        search_path += "\\";
    }
    search_path += "*.wav";

    HANDLE hFind = FindFirstFileA(search_path.c_str(), &find_data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            if (!(find_data.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
                std::string filename = wav_directory;
                if (filename.back() != '\\' && filename.back() != '/') {
                    filename += "\\";
                }
                filename += find_data.cFileName;
                wav_files.push_back(filename);
            }
        } while (FindNextFileA(hFind, &find_data) != 0);
        FindClose(hFind);
    }
#else
    // POSIX implementation using dirent.h
    #include <dirent.h>
    DIR* dir = opendir(wav_directory.c_str());
    if (dir != nullptr) {
        struct dirent* entry;
        while ((entry = readdir(dir)) != nullptr) {
            std::string filename(entry->d_name);
            // Check if file ends with .wav or .WAV
            if (filename.length() > 4) {
                std::string ext = filename.substr(filename.length() - 4);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".wav") {
                    std::string full_path = wav_directory;
                    if (full_path.back() != '/') {
                        full_path += "/";
                    }
                    full_path += filename;
                    wav_files.push_back(full_path);
                }
            }
        }
        closedir(dir);
    }
#endif

    if (wav_files.empty()) {
        fprintf(stderr, "%s: ERROR: no WAV files found in directory '%s'\n", __func__, wav_directory.c_str());
        return 1;
    }

    // Sort files alphabetically for consistent processing order
    std::sort(wav_files.begin(), wav_files.end());

    fprintf(stderr, "%s: found %zu WAV files in '%s'\n", __func__, wav_files.size(), wav_directory.c_str());
    fprintf(stderr, "\n");
    fprintf(stderr, "%s: WAV files to process:\n", __func__);
    for (size_t i = 0; i < wav_files.size(); ++i) {
        fprintf(stderr, "  %3zu. %s\n", i + 1, wav_files[i].c_str());
    }
    fprintf(stderr, "\n");

    // Process each WAV file sequentially
    for (size_t file_idx = 0; file_idx < wav_files.size() && is_running; ++file_idx) {
        const std::string& wav_file = wav_files[file_idx];
        
        fprintf(stdout, "\n");
        fprintf(stdout, "%s: ======================================\n", __func__);
        fprintf(stdout, "%s: Processing file %zu/%zu\n", __func__, file_idx + 1, wav_files.size());
        fprintf(stdout, "%s: File: %s\n", __func__, wav_file.c_str());
        fprintf(stdout, "%s: ======================================\n", __func__);
        
        // Clear previous audio data
        pcmf32_cur.clear();
        
        // Load WAV file into pcmf32_cur
        std::vector<std::vector<float>> pcmf32s;
        if (!read_audio_data(wav_file, pcmf32_cur, pcmf32s, false)) {
            fprintf(stderr, "%s: ERROR: failed to load WAV file '%s'\n", __func__, wav_file.c_str());
            if (fout.is_open()) {
                fout << wav_file << " : [ERROR: Failed to load file]" << std::endl;
            }
            continue; // Skip to next file
        }

        fprintf(stdout, "%s: Loaded %zu samples (%.2f seconds)\n", 
                __func__, 
                pcmf32_cur.size(), 
                (float)pcmf32_cur.size() / WHISPER_SAMPLE_RATE);

        // Transcribe the loaded audio
        int64_t t_ms = 0;
        const auto txt = ::trim(::transcribe(ctx, params, pcmf32_cur, "", logprob_min, logprob_sum, n_tokens, t_ms));

        const float p = 100.0f * std::exp(logprob_min);

        fprintf(stdout, "%s: Transcription: '%s%s%s'\n", 
                __func__, "\033[1m", txt.c_str(), "\033[0m");
        fprintf(stdout, "%s: Time: %d ms, Probability: %.2f%%\n", 
                __func__, (int)t_ms, p);

        // Write to output file if specified
        if (fout.is_open()) {
            fout << wav_file << " : " << txt << std::endl;
            fout.flush(); // Ensure data is written immediately
        }

        // Handle Ctrl + C
        is_running = sdl_poll_events();
    }

    fprintf(stdout, "\n");
    fprintf(stdout, "%s: ======================================\n", __func__);
    fprintf(stdout, "%s: Completed processing %zu files\n", __func__, wav_files.size());
    fprintf(stdout, "%s: ======================================\n", __func__);

    return 0;
}

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    whisper_params params;

    if (whisper_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (whisper_lang_id(params.language.c_str()) == -1) {
        fprintf(stderr, "error: unknown language '%s'\n", params.language.c_str());
        whisper_print_usage(argc, argv, params);
        exit(0);
    }

    // whisper init

    struct whisper_context_params cparams = whisper_context_default_params();

    cparams.use_gpu    = params.use_gpu;
    cparams.flash_attn = params.flash_attn;

    struct whisper_context * ctx = whisper_init_from_file_with_params(params.model.c_str(), cparams);
    if (ctx == nullptr) {
        fprintf(stderr, "error: failed to initialize whisper context\n");
        return 2;
    }

    // print some info about the processing
    {
        fprintf(stderr, "\n");
        if (!whisper_is_multilingual(ctx)) {
            if (params.language != "en" || params.translate) {
                params.language = "en";
                params.translate = false;
                fprintf(stderr, "%s: WARNING: model is not multilingual, ignoring language and translation options\n", __func__);
            }
        }
        fprintf(stderr, "%s: processing, %d threads, lang = %s, task = %s, timestamps = %d ...\n",
                __func__,
                params.n_threads,
                params.language.c_str(),
                params.translate ? "translate" : "transcribe",
                params.no_timestamps ? 0 : 1);

        fprintf(stderr, "\n");
    }

    //// init audio

    //audio_async audio(30*1000);
    //if (!audio.init(params.capture_id, WHISPER_SAMPLE_RATE)) {
    //    fprintf(stderr, "%s: audio.init() failed!\n", __func__);
    //    return 1;
    //}

    //audio.resume();

    //// wait for 1 second to avoid any buffered noise
    //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //audio.clear();

    int  ret_val = 0;

    //if (!params.grammar.empty()) {
    //    auto & grammar = params.grammar_parsed;
    //    if (is_file_exist(params.grammar.c_str())) {
    //        // read grammar from file
    //        std::ifstream ifs(params.grammar.c_str());
    //        const std::string txt = std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    //        grammar = grammar_parser::parse(txt.c_str());
    //    } else {
    //        // read grammar from string
    //        grammar = grammar_parser::parse(params.grammar.c_str());
    //    }

    //    // will be empty (default) if there are parse errors
    //    if (grammar.rules.empty()) {
    //        ret_val = 1;
    //    } else {
    //        fprintf(stderr, "%s: grammar:\n", __func__);
    //        grammar_parser::print_grammar(stderr, grammar);
    //        fprintf(stderr, "\n");
    //    }
    //}

    std::ofstream fout;
    if (params.fname_out.length() > 0) {
        fout.open(params.fname_out);
        if (!fout.is_open()) {
            fprintf(stderr, "%s: failed to open output file '%s'!\n", __func__, params.fname_out.c_str());
            return 1;
        }
    }

    /*if (ret_val == 0) {
        if (!params.commands.empty()) {
            ret_val = process_command_list(ctx, audio, params, fout);
        } else if (!params.prompt.empty() && params.grammar_parsed.rules.empty()) {
            ret_val = always_prompt_transcription(ctx, audio, params, fout);
        } else {
            ret_val = process_general_transcription(ctx, audio, params, fout);
        }
    }*/
	ret_val = process_general_transcription_from_file(ctx, params, fout);

    //audio.pause();

    whisper_print_timings(ctx);
    whisper_free(ctx);

    return ret_val;
}
