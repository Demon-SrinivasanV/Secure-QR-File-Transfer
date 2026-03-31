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
% > python Secure_QR_File_Transfer.py generate \
      /path/to/file/file.ext \
      /path/to/QR \
      yourFavouritePassword
% > python Secure_QR_File_Transfer.py recover \ 
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
import numpy as np

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
    # Build an MP4 video from the generated QR images.
    qr_images = sorted(output_dir.glob(f"{base_name}_qr_*.png"))
    if qr_images:
        first = cv2.imread(str(qr_images[0]))
        h, w = first.shape[:2]
        video_path = output_dir / f"{base_name}_qr.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(video_path), fourcc, 1.0, (w, h))
        for img_path in qr_images:
            frame = cv2.imread(str(img_path))
            writer.write(frame)
        writer.release()
        print(f"Generated video: {video_path}")

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
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(cv_img)
        if data:
            return data                                         # OpenCV already returns str
    except Exception:
        pass
    return None


def _is_blurry(cv_frame, threshold: float = 50.0) -> bool:
    """Return True if the frame is too blurry to decode a QR code."""
    gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def _preprocess_frame(cv_frame):
    """
    Yield increasingly aggressive image preprocessing variants of a cv2 frame.
    Each variant is a Pillow image suitable for QR decoding.
    This helps decode QR codes from phone camera footage with perspective
    distortion, uneven lighting, or slight blur.
    """
    # 1. Original frame as-is
    yield Image.fromarray(cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB))

    gray = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2GRAY)

    # 2. Sharpened grayscale
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    yield Image.fromarray(sharpened)

    # 3. Adaptive threshold (handles uneven lighting)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 11
    )
    yield Image.fromarray(thresh)

    # 4. OTSU threshold on sharpened
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    yield Image.fromarray(otsu)


def _extract_payload_from_frame(cv_frame) -> Optional[str]:
    """
    Try to decode a QR code from a cv2 BGR frame using multiple preprocessing
    passes. Returns the payload string on first successful decode, else None.
    """
    for pil_img in _preprocess_frame(cv_frame):
        result = _extract_payload_from_qr(pil_img)
        if result is not None:
            return result
    return None


def _parse_qr_payload(raw_str: str) -> Tuple[Optional[int], Optional[bytes]]:
    """
    Parse a decoded QR string into (seq, payload_bytes).
    seq == 0  -> payload is the raw 16-byte salt.
    seq >= 1  -> payload is: nonce(12) + tag(16) + ciphertext.
    Returns (None, None) if malformed.
    """
    if raw_str.startswith("SALT:"):
        try:
            salt = base64.b64decode(raw_str[5:])
        except Exception:
            return None, None
        return 0, salt

    try:
        decoded = base64.b64decode(raw_str)
    except Exception:
        return None, None

    if len(decoded) < 32:
        return None, None

    seq = struct.unpack(">I", decoded[:4])[0]
    return seq, decoded[4:]


def _process_qr_file(path: Path) -> Tuple[Optional[int], Optional[bytes]]:
    """
    Return (seq, payload_bytes) for a QR image file.
    seq == 0  -> payload is the raw 16-byte salt.
    seq >= 1  -> payload is: nonce(12) + tag(16) + ciphertext.
    Returns (None, None) if unreadable or malformed.
    """
    try:
        img = Image.open(path)
    except Exception:
        return None, None

    raw_str = _extract_payload_from_qr(img)
    if raw_str is None:
        return None, None

    return _parse_qr_payload(raw_str)


def read_and_assemble(source_path: str, password: str, output_path: Optional[str] = None) -> None:
    # Read QR codes from *source_path* (directory, image, or video) and rebuild file.
    src = Path(source_path)
    chunks: dict = {}
    salt: Optional[bytes] = None
    is_video = False

    if src.is_dir():
        image_paths = sorted(
            p for p in src.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        for p in image_paths:
            seq, payload = _process_qr_file(p)
            if seq is None:
                print(f"Unreadable QR in {p.name}")
                continue
            if seq == 0:
                salt = payload
            else:
                chunks[seq] = payload

    elif src.is_file():
        if src.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            seq, payload = _process_qr_file(src)
            if seq is None:
                raise RuntimeError(f"Could not decode QR from {src}")
            if seq == 0:
                salt = payload
            else:
                chunks[seq] = payload

        elif src.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            is_video = True
            cap = cv2.VideoCapture(str(src))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            decoded_count = 0
            skipped_blur = 0
            skipped_dup = 0
            frame_idx = 0
            max_seq_seen = 0
            retry_positions: list = []

            # Downscale large frames for faster processing.
            # Version 40 QR codes (177x177 modules) need ~1800px to resolve.
            max_dim = 2000
            scale = 1.0
            if max(frame_w, frame_h) > max_dim:
                scale = max_dim / max(frame_w, frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)

            # Sample ~10 frames per second of video.
            sample_interval = max(1, int(fps // 10))

            print(f"Video: {total_frames} frames, {fps:.1f}fps, {frame_w}x{frame_h} "
                  f"-> {new_w}x{new_h}, sampling every {sample_interval} frames")

            # --- Pass 1: fast pyzbar-only scan -------------------------
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                if frame_idx % sample_interval != 0:
                    continue

                # Downscale for speed.
                if scale < 1.0:
                    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if _is_blurry(frame):
                    skipped_blur += 1
                    continue

                pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                raw_str = _extract_payload_from_qr(pil_img)

                if raw_str is None:
                    retry_positions.append(frame_idx)
                    continue

                seq, payload = _parse_qr_payload(raw_str)
                if seq is None:
                    continue

                if seq == 0:
                    if salt is None:
                        salt = payload
                        decoded_count += 1
                    else:
                        skipped_dup += 1
                elif seq not in chunks:
                    chunks[seq] = payload
                    decoded_count += 1
                    if seq > max_seq_seen:
                        max_seq_seen = seq
                else:
                    skipped_dup += 1

                # Progress every 20 new decodes.
                if decoded_count % 20 == 0 and decoded_count > 0:
                    print(f"  ... {decoded_count} unique QRs so far (frame {frame_idx}/{total_frames})")

            cap.release()
            print(f"Pass 1: {decoded_count} unique QRs, {skipped_blur} blurry, "
                  f"{skipped_dup} duplicates, {len(retry_positions)} undecoded")

            # --- Pass 2: retry failed frames with preprocessing --------
            expected_total = max_seq_seen + 1
            have_count = len(chunks) + (1 if salt else 0)
            missing = expected_total - have_count if expected_total > 0 else len(retry_positions)

            if missing > 0 and retry_positions:
                print(f"Pass 2: retrying {len(retry_positions)} frames ({missing} QRs missing)...")
                cap = cv2.VideoCapture(str(src))
                retry_set = set(retry_positions)
                frame_idx = 0
                pass2_decoded = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    if frame_idx not in retry_set:
                        continue

                    if scale < 1.0:
                        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    raw_str = _extract_payload_from_frame(frame)
                    if raw_str is None:
                        continue

                    seq, payload = _parse_qr_payload(raw_str)
                    if seq is None:
                        continue

                    if seq == 0 and salt is None:
                        salt = payload
                        pass2_decoded += 1
                    elif seq > 0 and seq not in chunks:
                        chunks[seq] = payload
                        pass2_decoded += 1
                        if seq > max_seq_seen:
                            max_seq_seen = seq

                    have_count = len(chunks) + (1 if salt else 0)
                    if max_seq_seen > 0 and have_count >= max_seq_seen + 1:
                        break

                cap.release()
                decoded_count += pass2_decoded
                print(f"Pass 2: recovered {pass2_decoded} additional QRs")

            print(f"Video total: {decoded_count} unique QRs decoded")
        else:
            raise ValueError("Unsupported file type for source_path")
    else:
        raise FileNotFoundError(f"{source_path} does not exist")

    if salt is None:
        raise RuntimeError("Salt QR (sequence 0) not found - cannot derive key")

    if not chunks:
        raise RuntimeError("No data QR codes found")

    # Derive key.
    key = _derive_key(password, salt)
    aesgcm = AESGCM(key)

    max_seq = max(chunks.keys())

    # Detect missing sequence numbers.
    missing_seqs = [i for i in range(1, max_seq + 1) if i not in chunks]
    if missing_seqs:
        print(f"Missing QR code sequence numbers ({len(missing_seqs)}): "
              + ", ".join(map(str, missing_seqs)))
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
