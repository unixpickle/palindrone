import ArgumentParser
import HCBacktrace
import Honeycrisp

public enum SampleMethod: String, ExpressibleByArgument, CaseIterable, Sendable {

  case random
  case rejection
  case greedy
  case bfs

  @recordCaller
  private func _sample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> [UInt8] {
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
    case .bfs:
      try await bfsSample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
    }
  }

  @recordCaller
  private func _randomSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> [UInt8] {
    let initialData =
      if let cc = charCount {
        [0, cc + 256]
      } else {
        [0]
      }
    var iterator = Sampler(
      model: model, prefixes: Tensor(data: initialData, shape: [1, initialData.count])
    ).sampleStream().makeAsyncIterator()

    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await iterator.next()!.ints()[0] - 256
      }

    if verbose {
      print("character count: \(charCount)")
    }
    var allChars = [UInt8]()
    for i in 0..<charCount {
      let nextToken = try await iterator.next()!.ints()[0]
      allChars.append(UInt8(nextToken))
      if verbose {
        if let scalar = UnicodeScalar(nextToken), scalar.isASCII {
          let asciiChar = Character(scalar)
          print("character \(i): '\(asciiChar)'")
        } else {
          print("character \(i): \(nextToken)")
        }
      }
    }
    return tokenizer.inverseAlternating(allChars)
  }

  @recordCaller
  private func _rejectionSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> [UInt8] {
    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await sampleCharCount(model: model, tokenizer: tokenizer)
      }
    var attempt = 1
    while true {
      if verbose {
        print("sampling attempt #\(attempt) ...")
      }
      let sample = try await randomSample(
        model: model, tokenizer: tokenizer, charCount: charCount, verbose: verbose)
      if sample.reversed() == sample {
        print("successfully found palindrome on attempt #\(attempt)")
        return sample
      }
      if verbose {
        print("attempt #\(attempt) did not yield palindrome")
      }
      attempt += 1
    }
  }

  @recordCaller
  private func _greedySample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> [UInt8] {
    let sampler = Sampler(model: model, prefixes: Tensor(data: [0], shape: [1, 1]))
    let countDist = sampler.predictNextLogits()
    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await sampler.sampleTokens(logits: countDist).ints()[0] - 256
      }
    if verbose {
      print("character count: \(charCount)")
    }
    sampler.sampled(tokens: Tensor(data: [charCount + 256], shape: [1, 1]))
    var result = [UInt8]()
    var endResult = [UInt8]()
    for i in stride(from: 0, to: charCount, by: 2) {
      let firstSample = sampler.sampleTokens(logits: sampler.predictNextLogits())
      sampler.sampled(tokens: firstSample)
      let token = UInt8(try await firstSample.ints()[0])
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
    return result + endResult.reversed()
  }

  @recordCaller
  private func _bfsSample(
    model: Transformer, tokenizer: Tokenizer, charCount: Int? = nil, verbose: Bool = false
  ) async throws -> [UInt8] {
    let charCount =
      if let cc = charCount {
        cc
      } else {
        try await sampleCharCount(model: model, tokenizer: tokenizer)
      }

    struct SearchNode {
      let prefix: [UInt8]
      let nll: Float
    }

    var nodes = [SearchNode(prefix: [], nll: 0.0)]
    while let nextNode = nodes.popLast() {
      print("popping \(nextNode.prefix) with nll \(nextNode.nll)")
      if nextNode.prefix.count == charCount {
        return tokenizer.inverseAlternating(nextNode.prefix)
      }
      let logits = model(
        Tensor(
          data: [0, charCount + 256] + nextNode.prefix.map(Int.init),
          shape: [1, nextNode.prefix.count + 2]))
      for (i, logProb) in try await logits.logSoftmax().floats().enumerated() {
        if i >= 0x80 {
          continue
        }

        // Enforce palindrome constraint.
        if nextNode.prefix.count % 2 == 1 {
          if i != Int(nextNode.prefix.last!) {
            continue
          }
        }

        nodes.append(SearchNode(prefix: nextNode.prefix + [UInt8(i)], nll: nextNode.nll - logProb))
      }
      nodes.sort { $0.nll > $1.nll }
    }
    return []
  }

  @recordCaller
  private func _sampleCharCount(model: Transformer, tokenizer: Tokenizer) async throws -> Int {
    var iterator = model.sampleStream(prefixes: Tensor(data: [0], shape: [1, 1]))
      .makeAsyncIterator()
    return try await iterator.next()!.ints()[0] - 256
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
    prefixLength = prefixes.shape[0]
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
  private func _sampleTokens(logits: Tensor, generator: RandomGenerator? = nil) -> Tensor {
    let gumbels = -(-Tensor(randLike: logits, generator: generator).log()).log()
    return (logits + gumbels).argmax(axis: -1).unsqueeze(axis: 1)
  }

  @recordCaller
  private func _sampleStream(count: Int? = nil, generator: RandomGenerator? = nil) -> AsyncStream<
    Tensor
  > {
    return AsyncStream { continuation in
      for _ in 0..<((count ?? model.config.tokenCount) - prefixLength) {
        if Task.isCancelled {
          return
        }
        let logits = predictNextLogits()
        let gumbels = -(-Tensor(randLike: logits, generator: generator).log()).log()
        let samples = (logits + gumbels).argmax(axis: -1).unsqueeze(axis: 1)
        sampled(tokens: samples)
        continuation.yield(samples)
      }
      continuation.finish()
    }
  }
}
