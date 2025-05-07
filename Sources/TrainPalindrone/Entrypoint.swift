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
    let data: Dataset.State?
  }

  // Dataset configuration
  @Option(name: .long, help: "Dataset directory.") var datasetDir: String
  @Option(name: .long, help: "Batch size.") var batchSize: Int = 8
  @Option(name: .long, help: "Divide batches into microbatches.") var microbatch: Int? = nil
  @Option(name: .long, help: "Minimum length of context.") var minChunkLength: Int = 8
  @Option(name: .long, help: "Maximum length of context.") var maxChunkLength: Int = 63

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

  var tokenizer: Tokenizer {
    Tokenizer(maxBytes: maxChunkLength)
  }

  mutating func run() async {
    print("Command:", CommandLine.arguments.joined(separator: " "))

    do {
      Backend.defaultBackend = try MPSBackend(allocator: .bucket)

      print("creating dataset...")
      let dataset = try Dataset(
        directory: datasetDir,
        minChunkLength: minChunkLength,
        maxChunkLength: maxChunkLength
      )

      var model: Transformer
      var opt: Adam
      let clipper = GradClipper()

      var step: Int = 0

      if FileManager.default.fileExists(atPath: savePath) {
        print("loading from checkpoint: \(savePath) ...")
        let data = try Data(contentsOf: URL(fileURLWithPath: savePath))
        let decoder = PropertyListDecoder()
        let state = try decoder.decode(State.self, from: data)

        print("creating model...")
        model = Transformer(config: state.modelConfig)
        try model.loadState(state.model)

        print("creating optimizer...")
        opt = Adam(model.parameters, lr: lr)
        if let optState = state.opt { try opt.loadState(optState) }

        if let clipperState = state.clipper { clipper.state = clipperState }
        if let dataState = state.data { dataset.state = dataState }
        step = state.step ?? 0
      } else {
        print("creating model...")
        model = Transformer(
          config: TransformerConfig(
            vocabSize: tokenizer.vocabSize,
            tokenCount: tokenizer.inputCount,
            layerCount: depth,
            modelDim: width,
            headDim: 64
          )
        )

        print("creating optimizer...")
        opt = Adam(model.parameters, lr: lr)
      }

      while true {
        let batch = try await loadBatch(dataset: dataset)
        let bs = batch.inputs.shape[0]
        let microbatch = microbatch ?? bs
        var totalLoss: Float = 0
        for i in stride(from: 0, to: bs, by: microbatch) {
          let microBs = min(microbatch, bs - i)
          let outputs = model(batch.inputs[i..<(i + microBs)])
          let losses = outputs.logSoftmax(axis: -1).gather(
            axis: -1,
            indices: batch.targets[i..<(i + microBs), ..., NewAxis()]
          )
          let loss = batch.masks[i..<(i + microBs)].when(isTrue: losses, isFalse: 0).sum()
          loss.backward()
          totalLoss += try await loss.item()
        }
        let normalizer = 1.0 / (try await batch.masks.cast(.float32).sum().item())
        let meanLoss = totalLoss * normalizer
        for (_, var p) in model.parameters {
          p.grad! = p.grad! * normalizer
        }
        let (norm, _) = try await clipper.clipGrads(model: model)
        opt.step()
        opt.clearGrads()
        print("step \(step): loss=\(meanLoss) grad_norm=\(norm)")
        step += 1
        if step % saveInterval == 0 {
          let state = State(
            modelConfig: model.config,
            model: try await model.state(),
            step: step,
            opt: try await opt.state(),
            clipper: clipper.state,
            data: dataset.state
          )
          let stateData = try PropertyListEncoder().encode(state)
          try stateData.write(to: URL(filePath: savePath), options: .atomic)
        }
      }

    } catch { print("FATAL ERROR: \(error)") }
  }

  func loadBatch(dataset: Dataset) async throws -> (inputs: Tensor, targets: Tensor, masks: Tensor)
  {
    var inputs = [Tensor]()
    var targets = [Tensor]()
    var masks = [Tensor]()
    for _ in 0..<batchSize {
      let sequence = try await dataset.next()
      let (input, target, mask) = tokenizer.tokenize(sequence: sequence)
      inputs.append(input)
      targets.append(target)
      masks.append(mask)
    }
    return (
      inputs: Tensor(stack: inputs), targets: Tensor(stack: targets), masks: Tensor(stack: masks)
    )
  }

}
