# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `voice_clone` now uploads the audio file contents (binary file handles) instead of
  path strings, fixing "corrupted" file errors from the API
  ([upstream issue #62](https://github.com/elevenlabs/elevenlabs-mcp/issues/62)).
- `server.json` version fields synced with the package version (were 0.9.1 / 0.8.1).

### Added
- Unit tests for all 27 MCP tool handlers with the ElevenLabs client mocked.
- Integration/e2e tests that start the MCP server over stdio and exercise tool
  listing and representative tool calls against a mocked backend.
- Lint (ruff check + format) step in CI; test matrix extended to Python 3.12 and 3.13.
- This changelog.

### Changed
- Loosened exact version pins (`fastapi`, `uvicorn`, `python-dotenv`, `httpx`,
  `fuzzywuzzy`, `sounddevice`, `soundfile`) to minimum-version constraints.
- Declared Python 3.13 support in package classifiers.

## [0.11.0]

Current published release. See the
[release notes](https://github.com/elevenlabs/elevenlabs-mcp/releases) for details.
