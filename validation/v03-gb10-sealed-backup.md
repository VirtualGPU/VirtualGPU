# GB10 v0.3 sealed report backup

The project owner approved one off-site backup of the GB10 capture.
A project operator working from one of the off-site machines ran
`tar -C <capture-root> -cf - reports | age -r <public-key>` as root on
Spark 1. The tar stream was encrypted in the pipeline. No plaintext
archive was written to disk.

The archive contains all 162 files under `reports`: 54 original timing
reports with mode `000` and 108 supporting files, including capture
receipts and non-timing host telemetry. The ciphertext is 87,081,928
bytes with SHA-256
`a804411aec6d1cbc27ad0a41892f817968a332d6fde55f953ef4f96c18e719cd`.
Copies are held on Spark 1 and on two project-owned off-site machines.

The age public key is
`age17pvv4kktj4z2def8fyrrq0d3slaejev7n46t4czvrr5mfv09rd5s8dxvjp`.
Only the project owner holds the private key. The capture operator,
release lead, and auditors lack the private key and cannot decrypt
the backup. Decryption requires authorization equivalent to formal
opening of the sealed reports.

## Deviation from the unopened requirement

The [capture procedure](v03-holdout-release-draft.md#capture-seal-and-retry)
requires sealed reports to remain unopened. The owner-approved backup
was a one-time deviation. The root process opened the sealed reports
and read their bytes to build the archive for encryption. No person
viewed the report contents, and no program parsed or inspected them
as report data.

The release lead independently verified that all three ciphertext
copies had the same SHA-256, that the original directory still
contained 162 files, and that all 54 original timing reports still
had mode `000`. No original file had been modified after the end of
capture at 2026-11-27T18:51:00Z.

## Recovery

Decrypt the backup only if originals are lost and authorization
equivalent to formal opening has been granted. Verify the ciphertext
SHA-256 before decryption. After extraction, verify each recovered
report against its published receipt SHA-256 before parsing any report.
The [GEMM seal](v03-gb10-gemm-seal.json) includes complete and
historical failed attempts. The [attention seal](v03-gb10-attention-seal.json)
contains complete attempts. The [mode inventory](v03-receipts/gb10-seal-modes.txt)
identifies all 54 original timing reports.
