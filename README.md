# Secure QR File Transfer

A Python utility to split any file into encrypted QR code images and reconstruct it from those images (or video frames). Useful for air-gapped transfers, offline backups, or any scenario where you need to move data through a camera.

---

## How It Works

1. **Generate** — The input file is read in chunks, each chunk is encrypted with AES-256-GCM, base64-encoded, and saved as a QR code image (Version 40, EC Level L). A leading QR code (sequence 0) stores the encryption salt.
2. **Recover** — QR images (or video frames) are scanned, ordered by their embedded sequence number, decrypted, and concatenated to reconstruct the original file byte-for-byte.

Each QR code holds up to **2,182 bytes** of plaintext data. Larger files simply produce more QR codes, numbered sequentially.

---

## Installation

Install dependencies via pip:

```bash
pip install "qrcode[pil]" pillow pyzbar opencv-python cryptography
```

> **Note:** `pyzbar` requires the `zbar` shared library. On macOS: `brew install zbar`. On Ubuntu/Debian: `sudo apt-get install libzbar0`.

---

## Usage

### Generate QR codes from a file

```bash
python secure_qr_file_transfer.py generate \
    /path/to/input/file.ext \
    /path/to/output/qr_folder \
    yourPassword
```

This produces a series of numbered PNG images in the output folder:

```
file_qr_000000.png   ← salt QR (sequence 0)
file_qr_000001.png
file_qr_000002.png
...
```

### Recover a file from QR codes

```bash
python secure_qr_file_transfer.py recover \
    /path/to/qr_folder \
    yourPassword \
    --out /path/to/output/recovered_file.ext
```

The `--out` flag is optional. If omitted, the file is saved as `reconstructed_<timestamp>.bin` in the current directory.

The source path can be:

- A **directory** of PNG/JPG/BMP images
- A **single image** file
- A **video file** (MP4, AVI, MOV, MKV) — frames are extracted and scanned automatically

---

## Security

| Property | Detail |
|---|---|
| Cipher | AES-256-GCM (authenticated encryption) |
| Key derivation | PBKDF2-HMAC-SHA256, 100,000 iterations |
| Salt | 16 bytes, randomly generated per session |
| Nonce | 12 bytes, randomly generated per chunk |
| Integrity | GCM authentication tag (16 bytes) verifies each chunk |

Each chunk is independently encrypted with a unique nonce. A wrong password or corrupted QR code will cause decryption to fail loudly rather than silently produce garbage output.

---

## Limitations & Notes

- **All QR codes are required.** If any data QR code (sequence ≥ 1) is missing or unreadable, recovery aborts. The salt QR (sequence 0) is also mandatory.
- **QR scanning quality matters.** Images should be well-lit and in focus. The tool tries `pyzbar` first and falls back to OpenCV's built-in detector.
- **Large files produce many QR codes.** At ~2 KB per chunk, a 1 MB file requires roughly 470 QR images.
- The tool does not currently support partial recovery or forward error correction across chunks.

---

## Example

```bash
# Split and encrypt a PDF
python secure_qr_file_transfer.py generate report.pdf ./qr_codes secret123

# Reconstruct it from the QR images
python secure_qr_file_transfer.py recover ./qr_codes secret123 --out report_recovered.pdf
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `cryptography` | AES-GCM encryption and PBKDF2 key derivation |
| `qrcode[pil]` | QR code generation |
| `pillow` | Image I/O |
| `pyzbar` | Fast QR code decoding (primary decoder) |
| `opencv-python` | Fallback QR decoder and video frame extraction |
