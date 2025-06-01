import HCBacktrace
import Honeycrisp

final public class Tokenizer: Sendable {
  public let maxBytes: Int

  public var inputCount: Int {
    maxBytes + 1
  }

  public var vocabSize: Int {
    maxBytes + 256 + 1
  }

  public init(maxBytes: Int) {
    self.maxBytes = maxBytes
  }

  @recordCaller
  private func _tokenize(sequence: [UInt8]) -> (inputs: Tensor, targets: Tensor, mask: Tensor) {
    #alwaysAssert(
      sequence.count <= maxBytes,
      "sequence length \(sequence) is greater than maximum \(maxBytes)")
    var fullSeq = [0, sequence.count + 256] + alternatingStartEnd(sequence).map(Int.init)
    while fullSeq.count < maxBytes + 2 {
      fullSeq.append(0)
    }
    let inputs = Tensor(data: fullSeq[..<(fullSeq.count - 1)])
    let targets = Tensor(data: fullSeq[1...])
    let mask = Tensor(
      data: Array(repeating: true, count: sequence.count + 1)
        + Array(repeating: false, count: inputCount - (sequence.count + 1)))
    precondition(inputs.shape == targets.shape)
    precondition(inputs.shape == mask.shape)
    return (inputs: inputs, targets: targets, mask: mask)
  }

  public func alternatingStartEnd(_ sequence: [UInt8]) -> [UInt8] {
    var result = [UInt8]()
    for i in 0..<(sequence.count / 2) {
      result.append(sequence[i])
      result.append(sequence[sequence.count - (i + 1)])
    }
    if sequence.count % 2 != 0 {
      result.append(sequence[sequence.count / 2])
    }
    return result
  }

  public func inverseAlternating(_ sequence: [UInt8]) -> [UInt8] {
    var result = [UInt8]()
    for i in stride(from: 0, to: sequence.count, by: 2) {
      result.append(sequence[i])
    }
    for i in stride(from: 2 * (sequence.count / 2) - 1, to: 0, by: -2) {
      result.append(sequence[i])
    }
    return result
  }
}
