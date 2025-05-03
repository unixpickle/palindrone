import Foundation
import Honeycrisp

public class Dataset {

  public let directory: String
  public let minChunkLength: Int
  public let maxChunkLength: Int
  public let textFilenames: [String]
  public let currentEpoch: Int = 0
  public let currentDocOffset: Int = 0

  private var allTextData: [UInt8]? = nil
  private var docsInEpoch: [[UInt8]]? = nil

  public init(directory: String, minChunkLength: Int, maxChunkLength: Int) throws {
    self.directory = directory
    self.minChunkLength = minChunkLength
    self.maxChunkLength = maxChunkLength

    let fileManager = FileManager.default
    let files = try fileManager.contentsOfDirectory(
      at: URL(filePath: directory), includingPropertiesForKeys: nil)
    textFilenames =
      files
      .filter { $0.pathExtension == "txt" }
      .map { $0.deletingPathExtension().lastPathComponent }
  }

  private func epochDocs() async throws -> [[UInt8]] {
    if let docs = docsInEpoch {
      return docs
    }
    let data = try loadTextData()
    // TODO: randomly chunk it up in the given length range.
    try await CPUBackend.global.use {
      let rng = CPUBackend.global.createRandom()
      rng.seed(currentEpoch)
      let lengths = Tensor(
        randInt: [data.count / minChunkLength],
        in: Int64(minChunkLength)..<(Int64(maxChunkLength) + 1),
        generator: rng
      )
      let cumLength = lengths.cumulativeSum(axis: 0)
      let chunkCount = try await (cumLength > data.count).argmax().ints()[0]
      let lengthInts = try await lengths[..<chunkCount].ints()

      var chunks = [[UInt8]]()
      var offset = 0
      for length in lengthInts {
        chunks.append(Array(data[offset..<(offset + length)]))
        offset += length
      }
      docsInEpoch = chunks
    }
    return docsInEpoch!
  }

  private func loadTextData() throws -> [UInt8] {
    if let result = allTextData {
      return result
    }
    let fileURL = URL(filePath: directory)
    let allURLs = textFilenames.map { fileURL.appending(component: $0) }
    allTextData = try allURLs.map { url in
      try String(contentsOf: url, encoding: .utf8)
    }.joined(separator: "").lowercased().compactMap { char -> UInt8? in
      guard let ascii = char.asciiValue, ascii >= 97 && ascii <= 122 else {
        return nil
      }
      return ascii
    }
    return allTextData!
  }

}
