import numpy as np
from scipy.io import wavfile


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """
    Приводит фазу к интервалу [-pi, pi).
    """
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def bits_to_phases(bits: np.ndarray) -> np.ndarray:
    """
    0 -> +pi/2
    1 -> -pi/2
    """
    bits = np.asarray(bits, dtype=np.uint8)
    return np.where(bits == 0, np.pi / 2.0, -np.pi / 2.0)


def phases_to_bits(phases: np.ndarray) -> np.ndarray:
    """
    Обратное преобразование фаз в биты.
    """
    phases = np.asarray(phases)
    return (phases < 0.0).astype(np.uint8)


def normalize_audio_to_float(audio: np.ndarray):
    """
    Переводит PCM/float аудио в float64.
    Возвращает:
        audio_f   - float64 массив
        orig_dtype - исходный dtype
        scale     - коэффициент масштаба для обратного преобразования
    """
    orig_dtype = audio.dtype

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        scale = float(info.max)
        audio_f = audio.astype(np.float64) / scale
    elif np.issubdtype(orig_dtype, np.floating):
        scale = 1.0
        audio_f = audio.astype(np.float64)
    else:
        raise TypeError(f"Неподдерживаемый dtype аудио: {orig_dtype}")

    return audio_f, orig_dtype, scale


def denormalize_audio_from_float(audio_f: np.ndarray, orig_dtype: np.dtype, scale: float) -> np.ndarray:
    """
    Возвращает аудио в исходный формат.
    """
    audio_f = np.clip(audio_f, -1.0, 1.0)

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return np.round(audio_f * info.max).astype(orig_dtype)

    return audio_f.astype(orig_dtype)


def phase_coding_encode_channel(x: np.ndarray, bits: np.ndarray, n_segments: int) -> np.ndarray:
    """
    Фазовое кодирование для одного канала.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("Ожидался одномерный массив для одного канала.")

    if n_segments <= 0:
        raise ValueError("n_segments должно быть положительным.")

    n_samples = x.shape[0]
    segment_len = int(np.ceil(n_samples / n_segments))
    total_len = segment_len * n_segments
    pad_len = total_len - n_samples

    if pad_len > 0:
        x_pad = np.pad(x, (0, pad_len), mode="constant")
    else:
        x_pad = x.copy()

    segments = x_pad.reshape(n_segments, segment_len)

    spectra = np.fft.rfft(segments, axis=1)
    magnitudes = np.abs(spectra)
    phases = np.angle(spectra)

    n_bins = phases.shape[1]

    if segment_len % 2 == 0:
        max_embed_bits = max(0, n_bins - 2)
    else:
        max_embed_bits = max(0, n_bins - 1)

    if len(bits) > max_embed_bits:
        raise ValueError(
            f"Слишком длинное сообщение: {len(bits)} бит. "
            f"Максимум для данного сегмента: {max_embed_bits} бит."
        )

    msg_phases = bits_to_phases(bits)

    phases_tilde = np.empty_like(phases)
    phases_tilde[0] = phases[0].copy()

    m = len(bits)
    if m > 0:
        phases_tilde[0, 1:1 + m] = msg_phases

    for i in range(1, n_segments):
        delta_phi = wrap_phase(phases[i] - phases[i - 1])
        phases_tilde[i] = wrap_phase(phases_tilde[i - 1] + delta_phi)

    spectra_tilde = magnitudes * np.exp(1j * phases_tilde)
    x_tilde_segments = np.fft.irfft(spectra_tilde, n=segment_len, axis=1)

    return x_tilde_segments.reshape(-1)[:n_samples]


def phase_coding_decode_channel(x_stego: np.ndarray, n_segments: int, message_len: int) -> np.ndarray:
    """
    Извлечение сообщения из одного канала.
    """
    x_stego = np.asarray(x_stego, dtype=np.float64)
    if x_stego.ndim != 1:
        raise ValueError("Ожидался одномерный массив для одного канала.")

    if n_segments <= 0:
        raise ValueError("n_segments должно быть положительным.")

    n_samples = x_stego.shape[0]
    segment_len = int(np.ceil(n_samples / n_segments))
    total_len = segment_len * n_segments
    pad_len = total_len - n_samples

    if pad_len > 0:
        x_pad = np.pad(x_stego, (0, pad_len), mode="constant")
    else:
        x_pad = x_stego.copy()

    segments = x_pad.reshape(n_segments, segment_len)
    spectrum_0 = np.fft.rfft(segments[0])
    phase_0 = np.angle(spectrum_0)

    n_bins = phase_0.shape[0]
    if segment_len % 2 == 0:
        max_embed_bits = max(0, n_bins - 2)
    else:
        max_embed_bits = max(0, n_bins - 1)

    if message_len > max_embed_bits:
        raise ValueError(
            f"message_len={message_len} превышает допустимый максимум {max_embed_bits}."
        )

    encoded_phases = phase_0[1:1 + message_len]
    return phases_to_bits(encoded_phases)


def phase_coding_encode(audio: np.ndarray, bits, n_segments: int) -> np.ndarray:
    """
    Кодирование mono/stereo аудио.
    """
    bits = np.asarray(list(bits), dtype=np.uint8)

    if audio.ndim == 1:
        return phase_coding_encode_channel(audio, bits, n_segments)

    if audio.ndim != 2:
        raise ValueError("Поддерживаются только mono или multi-channel массивы аудио.")

    channels = []
    for ch in range(audio.shape[1]):
        channels.append(phase_coding_encode_channel(audio[:, ch], bits, n_segments))
    return np.stack(channels, axis=1)


def phase_coding_decode(audio_stego: np.ndarray, n_segments: int, message_len: int) -> np.ndarray:
    """
    Декодирование mono/stereo аудио.
    """
    if audio_stego.ndim == 1:
        return phase_coding_decode_channel(audio_stego, n_segments, message_len)

    if audio_stego.ndim != 2:
        raise ValueError("Поддерживаются только mono или multi-channel массивы аудио.")

    return phase_coding_decode_channel(audio_stego[:, 0], n_segments, message_len)


def write_wav(path: str, fs: int, audio_f: np.ndarray, orig_dtype: np.dtype, scale: float) -> None:
    """
    Запись WAV-файла.
    """
    audio_out = denormalize_audio_from_float(audio_f, orig_dtype, scale)
    wavfile.write(path, fs, audio_out)


def parse_bits(bit_string: str) -> np.ndarray:
    """
    Преобразует строку вида '1011001' в массив бит.
    """
    bit_string = bit_string.strip()
    if not bit_string:
        return np.array([], dtype=np.uint8)
    if any(c not in "01" for c in bit_string):
        raise ValueError("Строка битов должна содержать только символы 0 и 1.")
    return np.array([int(c) for c in bit_string], dtype=np.uint8)


if __name__ == "__main__":
    fs = 44100
    duration_sec = 3.0
    t = np.arange(int(fs * duration_sec)) / fs

    # Синтетический аудиосигнал: сумма гармоник + небольшой шум.
    audio = (
        0.45 * np.sin(2 * np.pi * 440.0 * t)
        + 0.25 * np.sin(2 * np.pi * 880.0 * t)
        + 0.15 * np.sin(2 * np.pi * 1760.0 * t)
        + 0.03 * np.random.default_rng(42).normal(size=t.shape[0])
    )

    # Ограничим амплитуду, чтобы не было клиппинга.
    audio = np.clip(audio, -1.0, 1.0)

    audio_f, orig_dtype, scale = normalize_audio_to_float(audio.astype(np.float32))

    message = "1011001110001011"
    bits = parse_bits(message)

    n_segments = 32

    stego = phase_coding_encode(audio_f, bits, n_segments=n_segments)
    recovered_bits = phase_coding_decode(stego, n_segments=n_segments, message_len=len(bits))

    recovered_message = "".join(map(str, recovered_bits.tolist()))

    print("Исходное сообщение :", message)
    print("Извлечённое сообщение:", recovered_message)
    print("Совпадение          :", message == recovered_message)

    # Сохранение файлов для проверки.
    write_wav("synthetic_original.wav", fs, audio_f, orig_dtype, scale)
    write_wav("synthetic_stego.wav", fs, stego, orig_dtype, scale)

    print("Файлы сохранены: synthetic_original.wav, synthetic_stego.wav")