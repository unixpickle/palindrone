import ArgumentParser
import HCBacktrace
import Honeycrisp

public struct Sample {
  public let bytes: [UInt8]
  public let logProb: Float?
}

internal let ASCII_A = Int("a".first!.asciiValue!)
internal let ASCII_Z = Int("z".first!.asciiValue!)

public enum SampleMethod: String, ExpressibleByArgument, CaseIterable, Sendable {

  case random
  case rejection
  case greedy
  case greedyPair
  case bfs

  @recordCaller
  private func _sample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> Sample {
    switch self {
    case .random:
      try await randomSample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
    case .rejection:
      try await rejectionSample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
    case .greedy:
      try await greedySample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
    case .greedyPair:
      try await greedyPairSample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
    case .bfs:
      try await bfsSample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
    }
  }

  @recordCaller
  private func _sampleBatch(
    model: Transformer, tokenizer: Tokenizer, batchSize: Int, charCount: Int, verbose: Bool = false
  ) async throws -> [Sample] {
    switch self {
    case .greedy:
      return try await greedySampleBatch(
        model: model, tokenizer: tokenizer, batchSize: batchSize, charCount: charCount,
        verbose: verbose)
    default:
      var result = [Sample]()
      for _ in 0..<batchSize {
        result.append(
          try await sample(
            model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose))
      }
      return result
    }
  }

  @recordCaller
  private func _randomSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> Sample {
    var logProb: Float = 0.0
    let initialData =
      if let cc = charCount {
        [0, cc + 256]
      } else {
        [0]
      }
    let iterator = Sampler(
      model: model, prefixes: Tensor(data: [initialData])
    ).iterate()

    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await {
          let (output, lp) = iterator.next()!
          logProb += try await lp.item()
          return try await output.item(Int.self) - 256
        }()
      }

    var allChars = [UInt8]()
    for _ in 0..<charCount {
      let (sample, lp) = iterator.next()!
      logProb += try await lp.item()
      let nextToken = try await sample.item(Int.self)
      allChars.append(UInt8(min(0xff, nextToken)))
    }
    return Sample(bytes: tokenizer.inverseAlternating(allChars), logProb: logProb)
  }

  @recordCaller
  private func _rejectionSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false,
    batchSize: Int = 32
  ) async throws -> Sample {
    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await sampleCharCount(model: model, tokenizer: tokenizer).0
      }

    var attempt = 0
    while true {
      if verbose {
        print("sampling attempts starting at index #\(attempt*batchSize) ...")
      }

      let sampler = Sampler(
        model: model,
        prefixes: Tensor(data: [[0, charCount + 256]]).repeating(axis: 0, count: batchSize)
      )
      var samples = [[UInt8]](repeating: [], count: batchSize)
      for (sample, _) in sampler.iterate(count: charCount + 2, mask: ..<256) {
        for (i, x) in try await sample.ints().enumerated() {
          samples[i].append(UInt8(x))
        }
      }

      for (i, rawSample) in samples.enumerated() {
        let sample = tokenizer.inverseAlternating(rawSample)
        if sample.reversed() == sample {
          print("successfully found palindrome on attempt #\(attempt*batchSize + i + 1)")
          return Sample(bytes: sample, logProb: nil)
        }
      }

      attempt += 1
      if verbose {
        print("no palindrome after #\(attempt*batchSize) attempts")
      }
    }
  }

  @recordCaller
  private func _greedySample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> Sample {
    let sampler = Sampler(model: model, prefixes: Tensor(data: [[0]]))
    let countDist = sampler.predictNextLogits()
    let (charCount, lengthLogProb) =
      if let cc = charCount {
        (cc, Float(0.0))
      } else {
        try await {
          let sample = sampler.sampleTokens(logits: countDist, mask: 256...)
          let lp = countDist.logSoftmax(axis: -1).gather(axis: 1, indices: sample).flatten()
          return (try await sample.item(Int.self) - 256, try await lp.item())
        }()
      }
    var logProb = lengthLogProb

    if verbose {
      print("character count: \(charCount)")
    }
    sampler.sampled(tokens: Tensor(data: [[charCount + 256]]))
    var result = [UInt8]()
    var endResult = [UInt8]()
    for i in stride(from: 0, to: charCount, by: 2) {
      let logits = sampler.predictNextLogits()
      let firstSample = sampler.sampleTokens(logits: logits, mask: ..<256)
      logProb += try await logits.logSoftmax(axis: -1).gather(axis: 1, indices: firstSample).item()
      sampler.sampled(tokens: firstSample)
      let token = UInt8(try await firstSample.item(Int.self))
      result.append(token)
      if i + 1 < charCount {
        endResult.append(token)
        let _ = sampler.predictNextLogits()  // We don't actually care about the logits
        sampler.sampled(tokens: firstSample)
      }
      if verbose {
        let scalar = UnicodeScalar(token)
        if scalar.isASCII {
          let asciiChar = Character(scalar)
          print("character \(i): '\(asciiChar)'")
        } else {
          print("character \(i): \(token)")
        }
      }
    }
    return Sample(bytes: result + endResult.reversed(), logProb: logProb)
  }

  @recordCaller
  private func _greedySampleBatch(
    model: Transformer, tokenizer: Tokenizer, batchSize: Int, charCount: Int, verbose: Bool = false
  ) async throws -> [Sample] {
    let sampler = Sampler(
      model: model,
      prefixes: Tensor(data: [[0, charCount + 256]]).repeating(axis: 0, count: batchSize))
    var logProbs = [Float](repeating: 0.0, count: batchSize)
    var result = [[UInt8]](repeating: [], count: batchSize)
    var endResult = [[UInt8]](repeating: [], count: batchSize)
    for i in stride(from: 0, to: charCount, by: 2) {
      let logits = sampler.predictNextLogits()
      let firstSample = sampler.sampleTokens(logits: logits, mask: ..<256)
      let probs = try await logits.logSoftmax(axis: -1).gather(axis: -1, indices: firstSample)
        .floats()
      for (i, x) in probs.enumerated() {
        logProbs[i] += x
      }
      sampler.sampled(tokens: firstSample)
      for (j, token) in try await firstSample.ints().enumerated() {
        result[j].append(UInt8(token))
        if i + 1 < charCount {
          endResult[j].append(UInt8(token))
        }
        if verbose {
          if let scalar = UnicodeScalar(token), scalar.isASCII {
            let asciiChar = Character(scalar)
            print("sample \(j) character \(i): '\(asciiChar)'")
          } else {
            print("sample \(j) character \(i): \(token)")
          }
        }
      }
      if i + 1 < charCount {
        let _ = sampler.predictNextLogits()  // We don't actually care about the logits
        sampler.sampled(tokens: firstSample)
      }
    }
    let allSamples = zip(result, endResult).map { $0.0 + $0.1.reversed() }
    return zip(allSamples, logProbs).map { Sample(bytes: $0.0, logProb: $0.1) }
  }

  @recordCaller
  private func _greedyPairSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> Sample {
    let (charCount, lengthLogProb) =
      if let cc = charCount {
        (cc, Float(0.0))
      } else {
        try await sampleCharCount(model: model, tokenizer: tokenizer)
      }
    var logProb = lengthLogProb

    if verbose {
      print("character count: \(charCount)")
    }

    var samplePrefix = [0, charCount]
    for _ in 0..<(charCount / 2) {
      let prefixes: [[Int]] = (ASCII_A...ASCII_Z).map { samplePrefix + [$0] }
      let seq = Tensor(data: prefixes)
      let logits = Tensor.withGrad(enabled: false) {
        model(seq)[..., (-2)...]
      }

      // Get probability of producing each last token twice.
      let logProbs = logits.logSoftmax(axis: -1).gather(
        axis: 2, indices: seq[..., -1].reshape([seq.shape[0], 1, 1])
      ).reshape([seq.shape[0], 2]).sum(axis: 1).logSoftmax(axis: 0)

      let sample =
        try await logProbs.exp().multinomial(sampleCount: 1).item(Int.self)
      logProb +=
        try await logProbs[sample].item()
      samplePrefix.append(sample + ASCII_A)
      samplePrefix.append(sample + ASCII_A)
    }

    if charCount % 2 == 1 {
      let seq = Tensor(data: [samplePrefix])
      let logProbs = Tensor.withGrad(enabled: false) {
        model(seq)[..., ..., ..<256]
      }[0, -1].logSoftmax(axis: 0)
      let sample = try await logProbs.exp().multinomial(sampleCount: 1).item(Int.self)
      samplePrefix.append(sample)
      logProb += try await logProbs[sample].item()
    }

    return Sample(
      bytes: tokenizer.inverseAlternating(samplePrefix[2...].map { UInt8($0) }),
      logProb: logProb
    )
  }

  @recordCaller
  private func _bfsSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> Sample {
    var result: [UInt8]? = nil
    try await bfsSampleMany(
      model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose
    ) { x in
      result = x
      return false
    }
    if let result = result {
      return Sample(bytes: result, logProb: nil)
    } else {
      return Sample(bytes: [], logProb: nil)
    }
  }

  @recordCaller
  private func _bfsSampleMany(
    model: Transformer,
    tokenizer: Tokenizer,
    charCount: Int? = nil,
    verbose: Bool = false,
    cb: ([UInt8]) async throws -> Bool
  ) async throws {
    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await sampleCharCount(model: model, tokenizer: tokenizer).0
      }
    if verbose {
      print("character count: \(charCount)")
    }

    var nodes = PriorityQueue<[UInt8], Float>()
    nodes.push([], priority: 0.0)

    while let (nextPrefix, nextLL) = nodes.pop() {
      let nextNLL = -nextLL
      if nextPrefix.count == charCount {
        if !(try await cb(tokenizer.inverseAlternating(nextPrefix))) {
          return
        }
        continue
      }

      if verbose {
        print("expanding node of length \(nextPrefix.count) with NLL \(nextNLL)")
      }

      if nextPrefix.count % 2 == 1 && nextPrefix.count + 1 < charCount {
        // Lookahead to next token without an extra forward pass
        let prev = Int(nextPrefix.last!)
        let allLogits = model(
          Tensor(data: [[0, charCount + 256] + nextPrefix.map(Int.init) + [prev]])
        )
        let addNLL = try await -allLogits[..., -2].logSoftmax().floats()[prev]
        for (i, logProb) in try await allLogits[..., -1].logSoftmax().floats().enumerated() {
          if i < ASCII_A || i > ASCII_Z {
            continue
          }
          nodes.push(
            nextPrefix + [UInt8(prev), UInt8(i)],
            priority: -(nextNLL + addNLL - logProb)
          )
        }
      } else {
        let logits = model(
          Tensor(data: [[0, charCount + 256] + nextPrefix.map(Int.init)])
        )[..., -1]
        for (i, logProb) in try await logits.logSoftmax().floats().enumerated() {
          if i < ASCII_A || i > ASCII_Z {
            continue
          }

          // Enforce palindrome constraint.
          if nextPrefix.count % 2 == 1 {
            if i != Int(nextPrefix.last!) {
              continue
            }
          }

          nodes.push(
            nextPrefix + [UInt8(i)],
            priority: -(nextNLL - logProb)
          )
        }
      }
    }
  }

  @recordCaller
  private func _sampleCharCount(model: Transformer, tokenizer: Tokenizer) async throws -> (
    Int, Float
  ) {
    let sampler = Sampler(
      model: model, prefixes: Tensor(data: [[0]])
    )
    let logits = sampler.predictNextLogits()
    let sample = sampler.sampleTokens(logits: logits, mask: 256...)
    let logProb = logits.logSoftmax(axis: -1).gather(axis: 1, indices: sample)
    return (try await sample.item(Int.self) - 256, try await logProb.item())
  }

}

public class Sampler {
  public let model: Transformer
  public let kvCache: KVCache
  public let prefixLength: Int
  private var prevToken: Tensor
  private var needsSampled: Bool = false

  public init(model: Transformer, prefixes: Tensor) {
    self.model = model
    prefixLength = prefixes.shape[1]
    prevToken = prefixes
    kvCache = KVCache(batchSize: prefixes.shape[0], config: model.config)
  }

  @recordCaller
  private func _sampled(tokens: Tensor) {
    #alwaysAssert(needsSampled, "you must call predictNextLogits() before calling sampled()")
    #alwaysAssert(tokens.shape == [prevToken.shape[0], 1])
    #alwaysAssert(tokens.dtype == .int64)
    prevToken = tokens
    needsSampled = false
  }

  @recordCaller
  private func predictNextLogits() -> Tensor {
    #alwaysAssert(!needsSampled, "you must call sampled() before calling predictNextLogits again")
    needsSampled = true
    return Tensor.withGrad(enabled: false) {
      // Without asDependency, we may allocate fp16 parameters many
      // times at once since the internal cast() in the model doesn't
      // depend on any other result tensors.
      prevToken.asDependency {
        model(prevToken, kvCache: kvCache)[..., -1]
      }
    }
  }

  @recordCaller
  private func _sampleTokens(
    logits: Tensor, mask: some TensorIndex, generator: RandomGenerator? = nil
  ) -> Tensor {
    let useIndices = Tensor(data: 0..<logits.shape.last!)[mask]
    let logits = logits.gather(axis: 1, indices: useIndices)
    let indices = logits.softmax(axis: 1).multinomial(sampleCount: 1)
    return useIndices.unsqueeze(axis: 0).repeating(
      axis: 0, count: indices.shape[0]
    ).gather(axis: 1, indices: indices)
  }

  @recordCaller
  private func _sampleTokens(
    logits: Tensor, generator: RandomGenerator? = nil
  ) -> Tensor {
    logits.softmax(axis: -1).multinomial(sampleCount: 1)
  }

  @recordCaller
  private func _iterate(
    count: Int? = nil,
    mask: (any TensorIndex)? = nil,
    generator: RandomGenerator? = nil
  ) -> AnyIterator<(tokens: Tensor, logProbs: Tensor)> {
    var remaining = (count ?? model.config.tokenCount) - prefixLength

    return AnyIterator { [self] () -> (Tensor, Tensor)? in
      guard remaining > 0 else { return nil }
      remaining -= 1

      let logits = predictNextLogits()
      let samples =
        if let mask = mask {
          sampleTokens(logits: logits, mask: mask, generator: generator)
        } else {
          sampleTokens(logits: logits, generator: generator)
        }
      let logProbs = logits.logSoftmax(axis: -1).gather(axis: 1, indices: samples)
      sampled(tokens: samples)
      return (tokens: samples, logProbs: logProbs)
    }
  }
}
