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

      var iterator = model.sampleStream(prefixes: Tensor(data: [0], shape: [1, 1]))
        .makeAsyncIterator()
      let charCount = try await iterator.next()!.ints()[0] - 256
      print("character count: \(charCount)")
      var allChars = [UInt8]()
      for i in 0..<charCount {
        let nextToken = try await iterator.next()!.ints()[0]
        if let scalar = UnicodeScalar(nextToken), scalar.isASCII {
          let asciiChar = Character(scalar)
          print("character \(i): '\(asciiChar)'")
        } else {
          print("character \(i): \(nextToken)")
        }
        allChars.append(UInt8(nextToken))
      }
      let decoded = tokenizer.inverseAlternating(allChars)
      if let string = String(bytes: decoded, encoding: .utf8) {
        print("Result: \(string)")
      } else {
        print("Invalid UTF-8 sequence: \(decoded)")
      }

    } catch { print("FATAL ERROR: \(error)") }
  }

}
