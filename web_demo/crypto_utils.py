import os, base64, json
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

KEY_DIR = Path(__file__).parent / "keys"

def _pad(data: bytes) -> bytes:
    pad_len = 16 - (len(data) % 16)
    return data + bytes([pad_len] * pad_len)

def _unpad(data: bytes) -> bytes:
    pad_len = data[-1]
    return data[:-pad_len]

def ensure_rsa_keys():
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    pub = KEY_DIR / "public.pem"
    priv = KEY_DIR / "private.pem"
    if pub.exists() and priv.exists():
        return
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    with open(pub, "wb") as f:
        f.write(key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo))
    with open(priv, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption()))

def _load_rsa_pub():
    with open(KEY_DIR / "public.pem", "rb") as f:
        return serialization.load_pem_public_key(f.read())

def _load_rsa_priv():
    with open(KEY_DIR / "private.pem", "rb") as f:
        return serialization.load_pem_private_key(f.read(), password=None)

def aes_encrypt(data: bytes) -> tuple[bytes, bytes, bytes]:
    key = os.urandom(32)
    iv = os.urandom(16)
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    encryptor = cipher.encryptor()
    ct = encryptor.update(_pad(data)) + encryptor.finalize()
    return key, iv, ct

def aes_decrypt(key: bytes, iv: bytes, ct: bytes) -> bytes:
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))
    decryptor = cipher.decryptor()
    return _unpad(decryptor.update(ct) + decryptor.finalize())

def hybrid_encrypt(data: bytes) -> dict:
    aes_key, iv, ct = aes_encrypt(data)
    rsa_pub = _load_rsa_pub()
    encrypted_key = rsa_pub.encrypt(
        aes_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None))
    return {
        "encrypted_key": base64.b64encode(encrypted_key).decode(),
        "iv": base64.b64encode(iv).decode(),
        "ciphertext": base64.b64encode(ct).decode()
    }

def hybrid_decrypt(package: dict) -> bytes:
    encrypted_key = base64.b64decode(package["encrypted_key"])
    iv = base64.b64decode(package["iv"])
    ct = base64.b64decode(package["ciphertext"])
    rsa_priv = _load_rsa_priv()
    aes_key = rsa_priv.decrypt(
        encrypted_key,
        padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()),
                     algorithm=hashes.SHA256(), label=None))
    return aes_decrypt(aes_key, iv, ct)