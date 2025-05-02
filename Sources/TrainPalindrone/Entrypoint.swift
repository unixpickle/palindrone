import ArgumentParser
import Foundation
import HCBacktrace
import Honeycrisp
import Palindrone

@main struct Main: AsyncParsableCommand {

  struct State: Codable {
    let modelConfig: TransformerConfig
    let model: Trainable.State
    let step: Int?
    let opt: Adam.State?
    let clipper: GradClipper.State?
    // TODO: dataset state here.
  }

  // Dataset configuration
  @Option(name: .long, help: "Dataset directory.") var datasetDir: String
  @Option(name: .long, help: "Batch size.") var batchSize: Int = 8
  @Option(name: .long, help: "Divide batches into microbatches.") var microbatch: Int? = nil

  // Model hyperparameters
  @Option(name: .long, help: "Transformer layers.") var depth: Int = 12
  @Option(name: .long, help: "MLP hidden size.") var width: Int = 512

  // Adam hyperparams
  @Option(name: .shortAndLong, help: "Learning rate.") var lr: Float = 0.01
  @Option(name: .long, help: "Adam beta1.") var beta1: Float = 0.9
  @Option(name: .long, help: "Adam beta2.") var beta2: Float = 0.99
  @Option(name: .long, help: "Adam weight decay.") var weightDecay: Float = 0.01

  // Saving
  @Option(name: .shortAndLong, help: "Path to save state.") var savePath: String = "state.plist"
  @Option(name: .long, help: "Steps between saves.") var saveInterval: Int = 100

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    do {
      Backend.defaultBackend = try MPSBackend(allocator: .bucket)
      // TODO: train here.
    } catch { print("FATAL ERROR: \(error)") }
  }

}
