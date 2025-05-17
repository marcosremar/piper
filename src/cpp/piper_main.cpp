#include <iostream>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

int main(int argc, char *argv[]) {
  // ... existing code ...
  
  bool jsonInput = false;
  bool outputRaw = false;
  bool useCuda = false;
  bool listPhonemes = false;
  bool showHelp = false;
  bool jsonTiming = false;

  // Parse command-line arguments
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if ((arg == "--model") || (arg == "-m")) {
      if (i + 1 < argc) {
        modelPath = argv[++i];
      }
    } else if ((arg == "--output_file") || (arg == "-o")) {
      if (i + 1 < argc) {
        outputPath = argv[++i];
      }
    } else if ((arg == "--output_dir") || (arg == "-od")) {
      if (i + 1 < argc) {
        outputDir = argv[++i];
      }
    } else if ((arg == "--speaker") || (arg == "-s")) {
      if (i + 1 < argc) {
        speakerIdStr = argv[++i];
      }
    } else if ((arg == "--noise_scale") || (arg == "-n")) {
      if (i + 1 < argc) {
        noiseScale = std::stof(argv[++i]);
      }
    } else if ((arg == "--length_scale") || (arg == "-l")) {
      if (i + 1 < argc) {
        lengthScale = std::stof(argv[++i]);
      }
    } else if ((arg == "--noise_w") || (arg == "-w")) {
      if (i + 1 < argc) {
        noiseW = std::stof(argv[++i]);
      }
    } else if ((arg == "--sentence_silence") || (arg == "-ss")) {
      if (i + 1 < argc) {
        sentenceSilence = std::stof(argv[++i]);
      }
    } else if ((arg == "--json-input") || (arg == "-j")) {
      jsonInput = true;
    } else if ((arg == "--output-raw") || (arg == "-r")) {
      outputRaw = true;
    } else if ((arg == "--cuda") || (arg == "-c")) {
      useCuda = true;
    } else if ((arg == "--list-phonemes") || (arg == "-lp")) {
      listPhonemes = true;
    } else if ((arg == "--help") || (arg == "-h")) {
      showHelp = true;
    } else if ((arg == "--json_timing") || (arg == "-jt")) {
      jsonTiming = true;
    } else if ((arg == "--debug")) {
      spdlog::set_level(spdlog::level::debug);
    } else if ((arg == "--version") || (arg == "-v")) {
      std::cout << "Piper version: " << VERSION << std::endl;
      return EXIT_SUCCESS;
    } else if ((arg == "--espeak_data") || (arg == "-ed")) {
      if (i + 1 < argc) {
        config.eSpeakDataPath = argv[++i];
      }
    }
  }

  // ... existing code ...

  // Após a síntese de texto e a geração do áudio, se jsonTiming for true, 
  // Exportar informações de timing como JSON
  if (jsonTiming && !synthResult.phonemeLengths.empty()) {
    // Calcular taxa de amostragem em segundos
    float sampleRate = static_cast<float>(synthesisConfig.sampleRate);
    
    // Criar JSON com informações de timing dos fonemas
    nlohmann::json timingJson;
    timingJson["phoneme_count"] = synthResult.phonemeLengths.size();
    timingJson["sample_rate"] = sampleRate;
    timingJson["audio_seconds"] = synthResult.audioSeconds;
    
    // Calcular timing em segundos para cada fonema
    std::vector<float> phonemeTimes;
    float currentTime = 0.0f;
    
    timingJson["phonemes"] = nlohmann::json::array();
    for (size_t i = 0; i < synthResult.phonemeLengths.size(); i++) {
      int phonemeLength = synthResult.phonemeLengths[i];
      float phonemeDuration = static_cast<float>(phonemeLength) / sampleRate;
      
      nlohmann::json phonemeInfo;
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
    std::cout << timingJson.dump(2) << std::endl;
  }

  // ... existing code ...
} 