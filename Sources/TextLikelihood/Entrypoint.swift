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

  // Saving
  @Option(name: .shortAndLong, help: "Path to save state.") var savePath: String = "state.plist"

  @Option(name: .shortAndLong, help: "Text to measure likelihood of.") var text: String

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

      print("building sequence")
      let textBytes = [UInt8](text.utf8)
      let seq = Tensor(
        data: [0, 256 + textBytes.count] + tokenizer.alternatingStartEnd(textBytes).map(Int.init)
      ).unsqueeze(axis: 0)
      let output = Tensor.withGrad(enabled: false) {
        let logits = model(seq)
        return logits[..., 1..<(textBytes.count + 1)].logSoftmax(axis: -1).gather(
          axis: -1, indices: seq[..., 2...].unsqueeze(axis: -1)
        ).sum()
      }
      print("log likelihood: \(try await output.item())")
    } catch { print("FATAL ERROR: \(error)") }
  }

}
