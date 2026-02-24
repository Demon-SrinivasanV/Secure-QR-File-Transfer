"""
Utility to split a file into encrypted QR code images and to reconstruct the file
from a collection of QR code images (or video frames).

Two main functions:
1. `chunk_and_generate(input_path, output_dir, password)` - splits the file, encrypts
   each chunk, encodes it into a QR code (Version 40, Model 2, EC level L) and saves
   the image. The file name contains the zero-padded sequence number.
2. `read_and_assemble(source_path, password, output_path=None)` - scans the provided
   path (directory, single image, or video file) for QR codes, orders them by the
   embedded sequence number, decrypts the payload and concatenates the original data.
   Missing or unreadable QR codes are reported.

Usage:
% > python qr_file_transfer.py generate \
      /path/to/file/file.ext \
      /path/to/QR \
      yourFavouritePassword
% > python qr_file_transfer.py recover \ 
      /path/to/QR \
      yourFavouritePassword \
      --out /path/to/Output/Folder/newFile.ext

Dependencies (install via pip):
    qrcode[pil]
    pillow
    pyzbar
    opencv-python
    cryptography
"""

import os
import struct
import base64
from pathlib import Path
from typing import List, Tuple, Optional

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import qrcode
from qrcode.constants import ERROR_CORRECT_L
from PIL import Image

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
except ImportError:
    pyzbar_decode = None
import cv2

# ---------------------------------------------------------------------------
# Encryption helpers
# ---------------------------------------------------------------------------

def _derive_key(password: str, salt: bytes, length: int = 32) -> bytes:
    # Derive a symmetric key from a password using PBKDF2.
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=length,
        salt=salt,
        iterations=100_000,
    )
    return kdf.derive(password.encode())

# ---------------------------------------------------------------------------
# QR constants
# ---------------------------------------------------------------------------

"""
Version 40, EC level L stores up to 2953 binary bytes.
Our payload is base64-encoded, so usable binary bytes = floor(2953 * 3/4) = 2214.
Subtract 32-byte header (seq 4B + nonce 12B + tag 16B) → 2182 bytes of plaintext per chunk.
"""
MAX_QR_BINARY_CAPACITY = 2953                                   # Version 40, EC level L (maximum possible)
MAX_PAYLOAD_BYTES = (MAX_QR_BINARY_CAPACITY * 3) // 4           # base64 overhead: 2214 bytes
HEADER_SIZE = 4 + 12 + 16                                       # seq(4) + nonce(12) + tag(16) = 32 bytes
MAX_PLAINTEXT_CHUNK = MAX_PAYLOAD_BYTES - HEADER_SIZE           # 2182 bytes

# ---------------------------------------------------------------------------
# Chunking / generation
# ---------------------------------------------------------------------------

def chunk_and_generate(input_path: str, output_dir: str, password: str) -> None:
    # Split *input_path* into encrypted chunks and save QR code images.
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    """
    Generate a random salt. Store it in QR seq=0.
    FIX: Encode the salt QR payload as base64 with a "SALT:" prefix (text marker).
         This ensures _save_qr always receives a plain ASCII string, not raw bytes,
         which avoids ambiguity in how the qrcode library encodes the data.
    """
    salt = os.urandom(16)
    salt_b64 = base64.b64encode(salt).decode("ascii")
    salt_payload_str = "SALT:" + salt_b64                       # e.g. "SALT:ABC123=="

    base_name = input_path.stem
    _save_qr(salt_payload_str, output_dir / f"{base_name}_qr_{0:06d}.png")

    # Derive AES‑GCM key from password and salt.
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    with input_path.open("rb") as f:
        seq = 1
        while True:
            chunk = f.read(MAX_PLAINTEXT_CHUNK)
            if not chunk:
                break
            nonce = os.urandom(12)
            ciphertext = aesgcm.encrypt(nonce, chunk, None)     # ciphertext || tag (16B)
            ct, tag = ciphertext[:-16], ciphertext[-16:]
            # Layout: [seq 4B][nonce 12B][tag 16B][ct]
            payload = struct.pack(">I", seq) + nonce + tag + ct
            """
            FIX: Encode as a plain ASCII base64 string (not bytes) for consistent
                 QR encoding. The qrcode library treats str as alphanumeric/byte
                 stream consistently across versions; passing raw bytes can cause
                 it to prepend length/encoding headers that pyzbar returns verbatim,
                 scrambling the subsequent decode step.
            """
            b64_str = base64.b64encode(payload).decode("ascii")
            _save_qr(b64_str, output_dir / f"{base_name}_qr_{seq:06d}.png")
            seq += 1
    print(f"Generated {seq} QR code images (1 salt + {seq-1} data) in {output_dir}")


def _save_qr(data: str, path: Path) -> None:
    """
    Create a QR code image from *data* (a plain ASCII string) and save to *path*.
    FIX: The parameter type is now explicitly `str` rather than `bytes`.
         Passing a Python `str` to qr.add_data() forces the library into its
         well-tested text/byte-stream path and avoids the raw-bytes ambiguity that
         caused the nonce/tag to be silently corrupted on some qrcode versions.
    """
    qr = qrcode.QRCode(
        version=40,
        error_correction=ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(path)

# ---------------------------------------------------------------------------
# Reading / assembly
# ---------------------------------------------------------------------------

def _extract_payload_from_qr(image: Image.Image) -> Optional[str]:
    """
    Decode a QR code from a Pillow image and return the payload as a string.

    FIX: Return type changed to `str`. Both pyzbar (bytes → decode utf-8) and
         OpenCV (already str) are normalised to str here so the caller always gets
         a consistent type regardless of which decoder ran.
    """
    if pyzbar_decode:
        decoded = pyzbar_decode(image)
        if decoded:
            # pyzbar returns bytes; decode to str for uniform handling.
            return decoded[0].data.decode("utf-8", errors="replace")

    # Fallback to OpenCV QRCodeDetector
    try:
        import numpy as np
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(cv_img)
        if data:
            return data                                         # OpenCV already returns str
    except Exception:
        pass
    return None


def _process_qr_file(path: Path) -> Tuple[Optional[int], Optional[bytes]]:
    """
    Return (seq, payload_bytes) for a QR image file.
    seq == 0  → payload is the raw 16-byte salt.
    seq >= 1  → payload is: nonce(12) + tag(16) + ciphertext.
    Returns (None, None) if unreadable or malformed.
    """
    try:
        img = Image.open(path)
    except Exception:
        return None, None

    raw_str = _extract_payload_from_qr(img)
    if raw_str is None:
        return None, None

    """
    FIX: Salt QR codes now use the text prefix "SALT:" followed by base64 salt.
         Previously the code stored raw bytes starting with b"SALT" which was
         ambiguous when the qrcode library re-encoded them.
    """
    if raw_str.startswith("SALT:"):
        try:
            salt = base64.b64decode(raw_str[5:])
        except Exception:
            return None, None
        return 0, salt

    # Data QR codes: the entire string is base64-encoded binary payload.
    try:
        decoded = base64.b64decode(raw_str)
    except Exception:
        return None, None

    # Minimum: 4 (seq) + 12 (nonce) + 16 (tag) = 32 bytes.
    if len(decoded) < 32:
        return None, None

    seq = struct.unpack(">I", decoded[:4])[0]
    # Return everything after the 4-byte sequence number.
    return seq, decoded[4:]


def read_and_assemble(source_path: str, password: str, output_path: Optional[str] = None) -> None:
    # Read QR codes from *source_path* (directory, image, or video) and rebuild file.
    src = Path(source_path)
    image_paths: List[Path] = []
    temp_paths: List[Path] = []                                 # track temp video-frame files for cleanup

    if src.is_dir():
        image_paths.extend(
            p for p in sorted(src.iterdir())
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
    elif src.is_file():
        if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            image_paths.append(src)
        elif src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            cap = cv2.VideoCapture(str(src))
            idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                temp_path = src.parent / f"__frame_{idx}.png"
                pil_img.save(temp_path)
                image_paths.append(temp_path)
                temp_paths.append(temp_path)
                idx += 1
            cap.release()
        else:
            raise ValueError("Unsupported file type for source_path")
    else:
        raise FileNotFoundError(f"{source_path} does not exist")

    # Parse each QR image.
    chunks: dict = {}
    salt: Optional[bytes] = None

    for p in image_paths:
        seq, payload = _process_qr_file(p)
        if seq is None:
            print(f"Unreadable QR in {p.name}")
            continue
        if seq == 0:
            salt = payload
            continue
        chunks[seq] = payload

    if salt is None:
        raise RuntimeError("Salt QR (sequence 0) not found – cannot derive key")

    if not chunks:
        raise RuntimeError("No data QR codes found")

    # Derive key.
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    max_seq = max(chunks.keys())

    # Detect missing sequence numbers.
    missing_seqs = [i for i in range(1, max_seq + 1) if i not in chunks]
    if missing_seqs:
        print("Missing QR code sequence numbers:", ", ".join(map(str, missing_seqs)))
        print("Aborting: missing QR code(s) prevent reconstruction.")
        return

    # Decrypt and assemble.
    assembled = bytearray()
    for i in range(1, max_seq + 1):
        payload = chunks[i]
        nonce = payload[:12]
        tag = payload[12:28]
        ct = payload[28:]
        try:
            plaintext = aesgcm.decrypt(nonce, ct + tag, None)
        except Exception as e:
            print(f"Decryption failed for chunk {i}: {e}")
            print("Aborting: decryption failed for one or more chunks.")
            return
        assembled.extend(plaintext)

    # Write output.
    if output_path is None:
        import time
        ts = int(time.time())
        output_path = f"reconstructed_{ts}.bin"

    with open(output_path, "wb") as out_f:
        out_f.write(assembled)
    print(f"Reconstructed file written to {output_path}")

    # Cleanup temporary video frames.
    for p in temp_paths:
        try:
            p.unlink()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="QR file splitter / assembler")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    gen = subparsers.add_parser("generate", help="Split file into QR images")
    gen.add_argument("input", help="Path to input file")
    gen.add_argument("outdir", help="Directory to store QR images")
    gen.add_argument("password", help="Encryption password")

    rec = subparsers.add_parser("recover", help="Reassemble file from QR images/video")
    rec.add_argument("source", help="Directory, image, or video containing QR codes")
    rec.add_argument("password", help="Encryption password used during generation")
    rec.add_argument("--out", help="Output file path", default=None)

    args = parser.parse_args()
    if args.cmd == "generate":
        chunk_and_generate(args.input, args.outdir, args.password)
    else:
        read_and_assemble(args.source, args.password, args.out)