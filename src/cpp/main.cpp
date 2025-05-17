#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include "json.hpp"
#include "piper.hpp"

using namespace std;
using json = nlohmann::json;

enum OutputType { OUTPUT_FILE, OUTPUT_DIRECTORY, OUTPUT_STDOUT, OUTPUT_RAW };

struct RunConfig {
  // Path to .onnx voice file
  filesystem::path modelPath;

  // Path to JSON voice config file
  filesystem::path modelConfigPath;

  // Type of output to produce.
  // Default is to write a WAV file in the current directory.
  OutputType outputType = OUTPUT_DIRECTORY;

  // Path for output
  optional<filesystem::path> outputPath = filesystem::path(".");

  // Numerical id of the default speaker (multi-speaker voices)
  optional<piper::SpeakerId> speakerId;

  // For multi-speaker models, the name of speaker to use (resolved to id)
  optional<std::string> speakerName;

  // Amount of noise to add during audio generation
  optional<float> noiseScale;

  // Speed of speaking (1 = normal, < 1 is faster, > 1 is slower)
  optional<float> lengthScale;

  // Variation in phoneme lengths
  optional<float> noiseW;

  // Seconds of silence to add after each sentence
  optional<float> sentenceSilenceSeconds;

  // Path to espeak-ng data directory (default is next to piper executable)
  optional<filesystem::path> eSpeakDataPath;

  // Path to libtashkeel ort model
  // https://github.com/mush42/libtashkeel/
  optional<filesystem::path> tashkeelModelPath;

  // stdin input is lines of JSON instead of text with format:
  // {
  //   "text": str,               (required)
  //   "speaker_id": int,         (optional)
  //   "speaker": str,            (optional)
  //   "output_file": str,        (optional)
  // }
  bool jsonInput = false;

  // Seconds of extra silence to insert after a single phoneme
  optional<std::map<piper::Phoneme, float>> phonemeSilenceSeconds;

  // true to use CUDA execution provider
  bool useCuda = false;

  // Output phoneme timing information as JSON to stdout
  bool jsonTiming = false;
};

void parseArgs(int argc, char *argv[], RunConfig &runConfig);
void rawOutputProc(vector<int16_t> &sharedAudioBuffer, mutex &mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished);

// ----------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  spdlog::set_default_logger(spdlog::stderr_color_st("piper"));

  RunConfig runConfig;
  parseArgs(argc, argv, runConfig);

#ifdef _WIN32
  // Required on Windows to show IPA symbols
  SetConsoleOutputCP(CP_UTF8);
#endif

  piper::PiperConfig piperConfig;
  piper::Voice voice;

  spdlog::debug("Loading voice from {} (config={})",
                runConfig.modelPath.string(),
                runConfig.modelConfigPath.string());

  auto startTime = chrono::steady_clock::now();
  loadVoice(piperConfig, runConfig.modelPath.string(),
            runConfig.modelConfigPath.string(), voice, runConfig.speakerId,
            runConfig.useCuda);
  auto endTime = chrono::steady_clock::now();
  spdlog::info("Loaded voice in {} second(s)",
               chrono::duration<double>(endTime - startTime).count());

  // Get the path to the piper executable so we can locate espeak-ng-data, etc.
  // next to it.
#ifdef _MSC_VER
  auto exePath = []() {
    wchar_t moduleFileName[MAX_PATH] = {0};
    GetModuleFileNameW(nullptr, moduleFileName, std::size(moduleFileName));
    return filesystem::path(moduleFileName);
  }();
#else
#ifdef __APPLE__
  auto exePath = []() {
    char moduleFileName[PATH_MAX] = {0};
    uint32_t moduleFileNameSize = std::size(moduleFileName);
    _NSGetExecutablePath(moduleFileName, &moduleFileNameSize);
    return filesystem::path(moduleFileName);
  }();
#else
  auto exePath = filesystem::canonical("/proc/self/exe");
#endif
#endif

  if (voice.phonemizeConfig.phonemeType == piper::eSpeakPhonemes) {
    spdlog::debug("Voice uses eSpeak phonemes ({})",
                  voice.phonemizeConfig.eSpeak.voice);

    if (runConfig.eSpeakDataPath) {
      // User provided path
      piperConfig.eSpeakDataPath = runConfig.eSpeakDataPath.value().string();
    } else {
      // Assume next to piper executable
      piperConfig.eSpeakDataPath =
          std::filesystem::absolute(
              exePath.parent_path().append("espeak-ng-data"))
              .string();

      spdlog::debug("espeak-ng-data directory is expected at {}",
                    piperConfig.eSpeakDataPath);
    }
  } else {
    // Not using eSpeak
    piperConfig.useESpeak = false;
  }

  // Enable libtashkeel for Arabic
  if (voice.phonemizeConfig.eSpeak.voice == "ar") {
    piperConfig.useTashkeel = true;
    if (runConfig.tashkeelModelPath) {
      // User provided path
      piperConfig.tashkeelModelPath =
          runConfig.tashkeelModelPath.value().string();
    } else {
      // Assume next to piper executable
      piperConfig.tashkeelModelPath =
          std::filesystem::absolute(
              exePath.parent_path().append("libtashkeel_model.ort"))
              .string();

      spdlog::debug("libtashkeel model is expected at {}",
                    piperConfig.tashkeelModelPath.value());
    }
  }

  piper::initialize(piperConfig);

  // Scales
  if (runConfig.noiseScale) {
    voice.synthesisConfig.noiseScale = runConfig.noiseScale.value();
  }

  if (runConfig.lengthScale) {
    voice.synthesisConfig.lengthScale = runConfig.lengthScale.value();
  }

  if (runConfig.noiseW) {
    voice.synthesisConfig.noiseW = runConfig.noiseW.value();
  }

  if (runConfig.sentenceSilenceSeconds) {
    voice.synthesisConfig.sentenceSilenceSeconds =
        runConfig.sentenceSilenceSeconds.value();
  }

  if (runConfig.phonemeSilenceSeconds) {
    if (!voice.synthesisConfig.phonemeSilenceSeconds) {
      // Overwrite
      voice.synthesisConfig.phonemeSilenceSeconds =
          runConfig.phonemeSilenceSeconds;
    } else {
      // Merge
      for (const auto &[phoneme, silenceSeconds] :
           *runConfig.phonemeSilenceSeconds) {
        voice.synthesisConfig.phonemeSilenceSeconds->try_emplace(
            phoneme, silenceSeconds);
      }
    }

  } // if phonemeSilenceSeconds

  if (runConfig.outputType == OUTPUT_DIRECTORY) {
    runConfig.outputPath = filesystem::absolute(runConfig.outputPath.value());
    spdlog::info("Output directory: {}", runConfig.outputPath.value().string());
  }

  string line;
  vector<int16_t> audioBuffer;
  piper::SynthesisResult result;
  while (getline(cin, line)) {
    auto outputType = runConfig.outputType;
    auto speakerId = voice.synthesisConfig.speakerId;
    std::optional<filesystem::path> maybeOutputPath = runConfig.outputPath;

    if (runConfig.jsonInput) {
      // Each line is a JSON object
      json lineRoot = json::parse(line);

      // Text is required
      line = lineRoot["text"].get<std::string>();

      if (lineRoot.contains("output_file")) {
        // Override output WAV file path
        outputType = OUTPUT_FILE;
        maybeOutputPath =
            filesystem::path(lineRoot["output_file"].get<std::string>());
      }

      if (lineRoot.contains("speaker_id")) {
        // Override speaker id
        voice.synthesisConfig.speakerId =
            lineRoot["speaker_id"].get<piper::SpeakerId>();
      } else if (lineRoot.contains("speaker")) {
        // Resolve to id using speaker id map
        auto speakerName = lineRoot["speaker"].get<std::string>();
        if ((voice.modelConfig.speakerIdMap) &&
            (voice.modelConfig.speakerIdMap->count(speakerName) > 0)) {
          voice.synthesisConfig.speakerId =
              (*voice.modelConfig.speakerIdMap)[speakerName];
        } else {
          spdlog::warn("No speaker named: {}", speakerName);
        }
      }
    }

    // Timestamp is used for path to output WAV file
    const auto now = chrono::system_clock::now();
    const auto timestamp =
        chrono::duration_cast<chrono::nanoseconds>(now.time_since_epoch())
            .count();

    if (outputType == OUTPUT_DIRECTORY) {
      // Generate path using timestamp
      stringstream outputName;
      outputName << timestamp << ".wav";
      filesystem::path outputPath = runConfig.outputPath.value();
      outputPath.append(outputName.str());

      // Output audio to automatically-named WAV file in a directory
      ofstream audioFile(outputPath.string(), ios::binary);
      piper::textToWavFile(piperConfig, voice, line, audioFile, result);
      cout << outputPath.string() << endl;
    } else if (outputType == OUTPUT_FILE) {
      if (!maybeOutputPath || maybeOutputPath->empty()) {
        throw runtime_error("No output path provided");
      }

      filesystem::path outputPath = maybeOutputPath.value();

      if (!runConfig.jsonInput) {
        // Read all of standard input before synthesizing.
        // Otherwise, we would overwrite the output file for each line.
        stringstream text;
        text << line;
        while (getline(cin, line)) {
          text << " " << line;
        }

        line = text.str();
      }

      // Output audio to WAV file
      ofstream audioFile(outputPath.string(), ios::binary);
      piper::textToWavFile(piperConfig, voice, line, audioFile, result);
      cout << outputPath.string() << endl;
    } else if (outputType == OUTPUT_STDOUT) {
      // Output WAV to stdout
      piper::textToWavFile(piperConfig, voice, line, cout, result);
    } else if (outputType == OUTPUT_RAW) {
      // Raw output to stdout
      bool audioReady = false;
      bool audioFinished = false;
      vector<int16_t> sharedAudioBuffer;
      mutex mutAudio;
      condition_variable cvAudio;

#ifdef _WIN32
      // Needed on Windows to avoid terminal conversions
      setmode(fileno(stdout), O_BINARY);
      setmode(fileno(stdin), O_BINARY);
#endif

      // Start thread to handle audio output
      thread rawOutputThread(rawOutputProc, ref(sharedAudioBuffer),
                             ref(mutAudio), ref(cvAudio), ref(audioReady),
                             ref(audioFinished));

      auto audioCallback = [&audioBuffer, &sharedAudioBuffer, &mutAudio,
                            &cvAudio, &audioReady]() {
        // Signal thread that audio is ready
        {
          unique_lock lockAudio(mutAudio);
          copy(audioBuffer.begin(), audioBuffer.end(),
               back_inserter(sharedAudioBuffer));
          audioReady = true;
          cvAudio.notify_one();
        }
      };

      piper::textToAudio(piperConfig, voice, line, audioBuffer, result,
                         audioCallback);

      // Signal thread that there is no more audio
      {
        unique_lock lockAudio(mutAudio);
        audioFinished = true;
        cvAudio.notify_one();
      }

      // Wait for audio output to finish
      spdlog::info("Waiting for audio to finish playing...");
      rawOutputThread.join();
    } else {
      // Generate audio in one go
      piper::textToAudio(piperConfig, voice, line, audioBuffer, result, NULL);
      
      // Debug prints
      std::cout << "After textToAudio: " << (runConfig.jsonTiming ? "jsonTiming=true" : "jsonTiming=false")
                << " phonemeLengths.size()=" << result.phonemeLengths.size() << std::endl;
    }
    
    // Debug prints
    std::cout << "Before jsonTiming condition: "
              << (runConfig.jsonTiming ? "jsonTiming=true" : "jsonTiming=false")
              << " phonemeLengths.size()=" << result.phonemeLengths.size() << std::endl;

    // Se jsonTiming estiver ativado, gerar JSON com os tempos dos fonemas
    if (runConfig.jsonTiming && !result.phonemeLengths.empty()) {
      // Print debug information
      std::cout << "Generating JSON timing data..." << std::endl;
      std::cout << "Phoneme lengths: " << result.phonemeLengths.size() << std::endl;
      
      // Calcular taxa de amostragem em segundos
      float sampleRate = static_cast<float>(voice.synthesisConfig.sampleRate);
      
      // Criar JSON com informações de timing dos fonemas
      json timingJson;
      timingJson["phoneme_count"] = result.phonemeLengths.size();
      timingJson["sample_rate"] = sampleRate;
      timingJson["audio_seconds"] = result.audioSeconds;
      timingJson["text"] = line;
      
      // Calcular timing em segundos para cada fonema
      float currentTime = 0.0f;
      
      timingJson["phonemes"] = json::array();
      for (size_t i = 0; i < result.phonemeLengths.size(); i++) {
        int phonemeLength = result.phonemeLengths[i];
        float phonemeDuration = static_cast<float>(phonemeLength) / sampleRate;
        
        json phonemeInfo;
        phonemeInfo["index"] = i;
        phonemeInfo["length_samples"] = phonemeLength;
        phonemeInfo["start_time"] = currentTime;
        phonemeInfo["duration"] = phonemeDuration;
        phonemeInfo["end_time"] = currentTime + phonemeDuration;
        
        timingJson["phonemes"].push_back(phonemeInfo);
        
        currentTime += phonemeDuration;
      }
      
      timingJson["total_duration"] = currentTime;
      
      // Exibir o JSON na saída padrão
      cout << timingJson.dump(2) << endl;
    }

    spdlog::info("Real-time factor: {} (infer={} sec, audio={} sec)",
                 result.realTimeFactor, result.inferSeconds,
                 result.audioSeconds);

    // Restore config (--json-input)
    voice.synthesisConfig.speakerId = speakerId;

    if (runConfig.jsonTiming && !result.phonemeLengths.empty()) {
      std::cout << "After textToAudio: " << (runConfig.jsonTiming ? "jsonTiming=true" : "jsonTiming=false")
                << " phonemeLengths.size()=" << result.phonemeLengths.size() << std::endl;
    }

    // Print debug information before the if condition
    std::cout << "Before jsonTiming condition: "
              << (runConfig.jsonTiming ? "jsonTiming=true" : "jsonTiming=false")
              << " phonemeLengths.size()=" << result.phonemeLengths.size() << std::endl;
  } // for each line

  piper::terminate(piperConfig);

  return EXIT_SUCCESS;
}

// ----------------------------------------------------------------------------

void rawOutputProc(vector<int16_t> &sharedAudioBuffer, mutex &mutAudio,
                   condition_variable &cvAudio, bool &audioReady,
                   bool &audioFinished) {
  vector<int16_t> internalAudioBuffer;
  while (true) {
    {
      unique_lock lockAudio{mutAudio};
      cvAudio.wait(lockAudio, [&audioReady] { return audioReady; });

      if (sharedAudioBuffer.empty() && audioFinished) {
        break;
      }

      copy(sharedAudioBuffer.begin(), sharedAudioBuffer.end(),
           back_inserter(internalAudioBuffer));

      sharedAudioBuffer.clear();

      if (!audioFinished) {
        audioReady = false;
      }
    }

    cout.write((const char *)internalAudioBuffer.data(),
               sizeof(int16_t) * internalAudioBuffer.size());
    cout.flush();
    internalAudioBuffer.clear();
  }

} // rawOutputProc

// ----------------------------------------------------------------------------

void printUsage(char *argv[]) {
  cout << endl;
  cout << "Usage: " << argv[0]
       << " --model /path/to/file.onnx [options]" << endl;
  cout << endl;
  cout << "Required Arguments:" << endl;
  cout << "  -m, --model    Path to onnx model file (-m)" << endl;
  cout << endl;
  cout << "Input/Output:" << endl;
  cout << "  -o,  --output_file    Path to output WAV file (default: "
          "output.wav)"
       << endl;
  cout << "  -od, --output_dir     Path to output directory (default: "
          "current)"
       << endl;
  cout << "  -r,  --output-raw     Output raw audio to stdout" << endl;
  cout << "  -j,  --json-input     Input lines are JSON objects" << endl;
  cout << "  -jt, --json_timing    Output phoneme timing information as JSON" << endl;
  cout << endl;
  cout << "Synthesis:" << endl;
  cerr << "   -s  NUM   --speaker     NUM   id of speaker (default: 0)" << endl;
  cerr << "   --noise_scale           NUM   generator noise (default: 0.667)"
       << endl;
  cerr << "   --length_scale          NUM   phoneme length (default: 1.0)"
       << endl;
  cerr << "   --noise_w               NUM   phoneme width noise (default: 0.8)"
       << endl;
  cerr << "   --sentence_silence      NUM   seconds of silence after each "
          "sentence (default: 0.2)"
       << endl;
  cerr << "   --espeak_data           DIR   path to espeak-ng data directory"
       << endl;
  cerr << "   --tashkeel_model        FILE  path to libtashkeel onnx model "
          "(arabic)"
       << endl;
  cerr << "   --use-cuda                    use CUDA execution provider"
       << endl;
  cerr << "   --debug                       print DEBUG messages to the console"
       << endl;
  cerr << "   -q       --quiet              disable logging" << endl;
  cerr << endl;
}

void ensureArg(int argc, char *argv[], int argi) {
  if ((argi + 1) >= argc) {
    printUsage(argv);
    exit(0);
  }
}

// Parse command-line arguments
void parseArgs(int argc, char *argv[], RunConfig &runConfig) {
  if (argc < 2) {
    printUsage(argv);
    exit(1);
  }

  for (int i = 1; i < argc; i++) {
    string arg = argv[i];
    if ((arg == "--help") || (arg == "-h")) {
      printUsage(argv);
      exit(0);
    } else if (arg == "--model" || arg == "-m") {
      ensureArg(argc, argv, i);
      runConfig.modelPath = filesystem::path(argv[++i]);

      if (!runConfig.modelConfigPath.has_filename()) {
        // Default config path is model path + .json
        runConfig.modelConfigPath =
            filesystem::path(string(runConfig.modelPath.string()) + ".json");
      }
    } else if (arg == "--model-config" || arg == "-mc") {
      ensureArg(argc, argv, i);
      runConfig.modelConfigPath = filesystem::path(argv[++i]);
    } else if (arg == "--output_file" || arg == "-o") {
      ensureArg(argc, argv, i);
      runConfig.outputType = OUTPUT_FILE;
      runConfig.outputPath = filesystem::path(argv[++i]);
    } else if (arg == "--output_dir" || arg == "-od") {
      ensureArg(argc, argv, i);
      runConfig.outputType = OUTPUT_DIRECTORY;
      runConfig.outputPath = filesystem::path(argv[++i]);
    } else if (arg == "--output-raw" || arg == "-r") {
      runConfig.outputType = OUTPUT_RAW;
    } else if (arg == "--cuda" || arg == "-c") {
      runConfig.useCuda = true;
    } else if (arg == "--json-input" || arg == "-j") {
      runConfig.jsonInput = true;
    } else if (arg == "--json_timing" || arg == "-jt") {
      runConfig.jsonTiming = true;
      std::cout << "JSON timing enabled" << std::endl;
    } else if (arg == "--speaker" || arg == "-s") {
      ensureArg(argc, argv, i);
      try {
        try {
          // Try to parse as integer speaker id
          runConfig.speakerId = stoi(argv[++i]);
        } catch (const exception &) {
          // Try to parse as string speaker name (will be resolved later)
          runConfig.speakerName = string(argv[i]);
          runConfig.speakerId = nullopt;
        }
      } catch (const exception &e) {
        spdlog::error("--speaker requires an id or name");
        exit(1);
      }
    } else if (arg == "--noise_scale" || arg == "--noise-scale") {
      ensureArg(argc, argv, i);
      runConfig.noiseScale = stof(argv[++i]);
    } else if (arg == "--length_scale" || arg == "--length-scale") {
      ensureArg(argc, argv, i);
      runConfig.lengthScale = stof(argv[++i]);
    } else if (arg == "--noise_w" || arg == "--noise-w") {
      ensureArg(argc, argv, i);
      runConfig.noiseW = stof(argv[++i]);
    } else if (arg == "--sentence_silence" || arg == "--sentence-silence") {
      ensureArg(argc, argv, i);
      runConfig.sentenceSilenceSeconds = stof(argv[++i]);
    } else if (arg == "--phoneme_silence" || arg == "--phoneme-silence") {
      ensureArg(argc, argv, i);
      ensureArg(argc, argv, i + 1);
      auto phonemeStr = std::string(argv[++i]);
      if (!piper::isSingleCodepoint(phonemeStr)) {
        std::cerr << "Phoneme '" << phonemeStr
                  << "' is not a single codepoint (--phoneme_silence)"
                  << std::endl;
        exit(1);
      }

      if (!runConfig.phonemeSilenceSeconds) {
        runConfig.phonemeSilenceSeconds.emplace();
      }

      auto phoneme = piper::getCodepoint(phonemeStr);
      (*runConfig.phonemeSilenceSeconds)[phoneme] = stof(argv[++i]);
    } else if (arg == "--espeak_data" || arg == "--espeak-data") {
      ensureArg(argc, argv, i);
      runConfig.eSpeakDataPath = filesystem::path(argv[++i]);
    } else if (arg == "--tashkeel_model" || arg == "--tashkeel-model") {
      ensureArg(argc, argv, i);
      runConfig.tashkeelModelPath = filesystem::path(argv[++i]);
    } else if (arg == "--version") {
      std::cout << piper::getVersion() << std::endl;
      exit(0);
    } else if (arg == "--debug") {
      // Set DEBUG logging
      spdlog::set_level(spdlog::level::debug);
    } else if (arg == "-q" || arg == "--quiet") {
      // diable logging
      spdlog::set_level(spdlog::level::off);
    }
  }

  // Verify model file exists
  ifstream modelFile(runConfig.modelPath.c_str(), ios::binary);
  if (!modelFile.good()) {
    throw runtime_error("Model file doesn't exist");
  }

  if (!runConfig.modelConfigPath.has_filename()) {
    runConfig.modelConfigPath =
        filesystem::path(runConfig.modelPath.string() + ".json");
  }

  // Verify model config exists
  ifstream modelConfigFile(runConfig.modelConfigPath.c_str());
  if (!modelConfigFile.good()) {
    throw runtime_error("Model config doesn't exist");
  }
}
