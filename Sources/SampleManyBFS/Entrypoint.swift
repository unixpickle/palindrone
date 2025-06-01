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
  @Option(name: .long, help: "Length of all samples.") var length: Int = 20

  // Saving
  @Option(name: .shortAndLong, help: "Path to load state.") var savePath: String = "state.plist"

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

      var cumulativeProb: Double = 0.0
      try await SampleMethod.bfs.bfsSampleMany(
        model: model, tokenizer: tokenizer, charCount: length
      ) { sample in
        let inputSeqs = Tensor(
          data: [0, length + 256] + tokenizer.alternatingStartEnd(sample).map(Int.init),
          shape: [1, length + 2]
        )
        let preds = Tensor.withGrad(enabled: false) {
          model(inputSeqs)[..., 1..<(inputSeqs.shape[1] - 1)]
        }
        let logProb = try await preds.logSoftmax(axis: -1).gather(
          axis: -1, indices: inputSeqs[..., 2...].unsqueeze(axis: -1)
        ).sum().item()
        cumulativeProb += exp(Double(logProb))
        print(
          "sample=\(String(bytes: sample, encoding: .utf8)!) logprob=\(logProb) cumprob=\(cumulativeProb)"
        )
        return true
      }
    } catch { print("FATAL ERROR: \(error)") }
  }

  func sampleAndPrint(_ allSamples: [[UInt8]], _ allLogProbs: [Float]) async throws {
    let probs = Tensor(data: allLogProbs).softmax(axis: 0)
    let sampleIdx = try await probs.multinomial(sampleCount: 1).ints()[0]
    let prob = try await probs[sampleIdx].item()
    let sample = allSamples[sampleIdx]
    if let string = String(bytes: sample, encoding: .utf8) {
      print("Result: \(string) (prob: \(prob))")
    } else {
      print("Invalid UTF-8 sequence: \(sample) (prob: \(prob))")
    }
  }

}
