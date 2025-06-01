// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "Palindrone",
  platforms: [.macOS(.v13)],
  products: [
    .library(name: "Palindrone", targets: ["Palindrone"])
  ],
  dependencies: [
    .package(url: "https://github.com/unixpickle/honeycrisp", from: "0.0.31"),
    .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
  ],
  targets: [
    .target(
      name: "Palindrone",
      dependencies: [
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
        .product(name: "HCBacktrace", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "TrainPalindrone",
      dependencies: [
        "Palindrone", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "SamplePalindrone",
      dependencies: [
        "Palindrone", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "SampleMany",
      dependencies: [
        "Palindrone", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
    .executableTarget(
      name: "SampleManyBFS",
      dependencies: [
        "Palindrone", .product(name: "ArgumentParser", package: "swift-argument-parser"),
        .product(name: "Honeycrisp", package: "honeycrisp"),
      ]
    ),
  ]
)
