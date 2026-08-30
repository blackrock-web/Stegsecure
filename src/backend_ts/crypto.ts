import crypto from 'crypto';

const MAGIC_HEADER = Buffer.from('SSTG', 'utf-8'); // 4 bytes
const SALT_SIZE = 16;
const IV_SIZE = 12;
const TAG_SIZE = 16;
const KEY_LEN = 32;
const ITERATIONS = 100000;

/**
 * Encrypts secret text using AES-256-GCM + PBKDF2 key derivation.
 * Format: [MAGIC 4b][SALT 16b][IV 12b][PAYLOAD_LEN 4b][CIPHERTEXT][TAG 16b]
 */
export function encryptPayload(secretText: string, passphrase: string): Buffer {
  const salt = crypto.randomBytes(SALT_SIZE);
  const iv = crypto.randomBytes(IV_SIZE);

  const key = crypto.pbkdf2Sync(passphrase, salt, ITERATIONS, KEY_LEN, 'sha256');

  const cipher = crypto.createCipheriv('aes-256-gcm', key, iv);
  const textBuffer = Buffer.from(secretText, 'utf-8');
  
  const encrypted = Buffer.concat([cipher.update(textBuffer), cipher.final()]);
  const tag = cipher.getAuthTag();

  const lenBuf = Buffer.alloc(4);
  lenBuf.writeUInt32BE(encrypted.length, 0);

  return Buffer.concat([MAGIC_HEADER, salt, iv, lenBuf, encrypted, tag]);
}

/**
 * Decrypts binary payload using AES-256-GCM + PBKDF2.
 * Throws error if passphrase or magic header is invalid.
 */
export function decryptPayload(payloadBuffer: Buffer, passphrase: string): string {
  const headerLen = MAGIC_HEADER.length + SALT_SIZE + IV_SIZE + 4;
  if (payloadBuffer.length < headerLen + TAG_SIZE) {
    throw new Error('Payload too short or corrupted.');
  }

  const magic = payloadBuffer.subarray(0, 4);
  if (!magic.equals(MAGIC_HEADER)) {
    throw new Error('Invalid magic header — image does not contain valid steganographic message.');
  }

  const salt = payloadBuffer.subarray(4, 20);
  const iv = payloadBuffer.subarray(20, 32);
  const lenBuf = payloadBuffer.subarray(32, 36);
  const ciphertextLen = lenBuf.readUInt32BE(0);

  const totalExpected = 36 + ciphertextLen + TAG_SIZE;
  if (payloadBuffer.length < totalExpected) {
    throw new Error('Incomplete payload buffer.');
  }

  const ciphertext = payloadBuffer.subarray(36, 36 + ciphertextLen);
  const tag = payloadBuffer.subarray(36 + ciphertextLen, totalExpected);

  const key = crypto.pbkdf2Sync(passphrase, salt, ITERATIONS, KEY_LEN, 'sha256');

  const decipher = crypto.createDecipheriv('aes-256-gcm', key, iv);
  decipher.setAuthTag(tag);

  const decrypted = Buffer.concat([decipher.update(ciphertext), decipher.final()]);
  return decrypted.toString('utf-8');
}
