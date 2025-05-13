import ArgumentParser
import Foundation
import HCBacktrace
import Honeycrisp
import Palindrone

@main struct Main: AsyncParsableCommand {

  struct State: Codable {
    let modelConfig: TransformerConfig
    let model: Trainable.State
  }

  // Data configuration
  @Option(name: .long, help: "Minimum length of context.") var minChunkLength: Int = 8
  @Option(name: .long, help: "Maximum length of context.") var maxChunkLength: Int = 63

  // Sampling parameters
  @Option(name: .long, help: "Sampling algorithm.") var algorithm: SampleMethod = .greedy
  @Option(name: .long, help: "If specified, force this sample length.") var length: Int? = nil

  // Saving
  @Option(name: .shortAndLong, help: "Path to save state.") var savePath: String = "state.plist"

  var tokenizer: Tokenizer {
    Tokenizer(maxBytes: maxChunkLength)
  }

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    do {
      Backend.defaultBackend = try MPSBackend(allocator: .bucket)

      print("loading from checkpoint: \(savePath) ...")
      let data = try Data(contentsOf: URL(fileURLWithPath: savePath))
      let decoder = PropertyListDecoder()
      let state = try decoder.decode(State.self, from: data)

      print("creating model...")
      let model = Transformer(config: state.modelConfig)
      try model.loadState(state.model)

      let sample = try await algorithm.sample(
        model: model, tokenizer: tokenizer, charCount: length, verbose: true)

      if let string = String(bytes: sample, encoding: .utf8) {
        print("Result: \(string)")
      } else {
        print("Invalid UTF-8 sequence: \(sample)")
      }

    } catch { print("FATAL ERROR: \(error)") }
  }

}
