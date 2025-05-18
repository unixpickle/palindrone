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
  @Option(name: .long, help: "Length of all samples.") var length: Int = 20
  @Option(name: .long, help: "Batch size to sample.") var batchSize: Int = 32

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

      var allSamples = [[UInt8]]()
      var allLogProbs = [Float]()

      while true {
        let batch = try await algorithm.sampleBatch(
          model: model, tokenizer: tokenizer, batchSize: batchSize, charCount: length
        )
        let inputSeqs = Tensor(
          data: batch.flatMap {
            [0, length + 256] + tokenizer.alternatingStartEnd($0.bytes).map(Int.init)
          },
          shape: [batchSize, length + 2])
        let preds = Tensor.withGrad(enabled: false) {
          model(inputSeqs)[..., 1..<(inputSeqs.shape[1] - 1)]
        }
        var logProbs = try await preds.logSoftmax(axis: -1).gather(
          axis: -1, indices: inputSeqs[..., 2...].unsqueeze(axis: -1)
        ).squeeze(axis: -1).sum(axis: 1).floats()

        // Importance sample, p(x) / p_sample(x)
        for (i, x) in batch.enumerated() {
          if let lp = x.logProb {
            logProbs[i] -= lp
          }
        }

        allLogProbs.append(contentsOf: logProbs)
        allSamples.append(contentsOf: batch.map { $0.bytes })

        try await sampleAndPrint(allSamples, allLogProbs)
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
