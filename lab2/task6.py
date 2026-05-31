import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def wrap_phase(phi: np.ndarray) -> np.ndarray:
    """Оборачивает фазу в диапазон [-pi, pi].

    Args:
        phi (np.ndarray): Массив фаз.

    Returns:
        np.ndarray: Приведенные к интервалу [-pi, pi] фазы.
    """
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def bits_to_phases(bits: np.ndarray) -> np.ndarray:
    """Преобразует биты в соответствующие начальные фазы.

    0 -> +pi/2
    1 -> -pi/2

    Args:
        bits (np.ndarray): Массив битов.

    Returns:
        np.ndarray: Массив соответствующих фаз.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    return np.where(bits == 0, np.pi / 2.0, -np.pi / 2.0)


def phases_to_bits(phases: np.ndarray) -> np.ndarray:
    """Восстанавливает биты из значений фаз.

    Отрицательные фазы интерпретируются как 1, неотрицательные как 0.

    Args:
        phases (np.ndarray): Массив фаз.

    Returns:
        np.ndarray: Восстановленный массив битов.
    """
    phases = np.asarray(phases)
    return (phases < 0.0).astype(np.uint8)


def normalize_audio_to_float(audio: np.ndarray):
    """Нормализует аудиоданные в вещественный формат в диапазоне [-1.0, 1.0].

    Args:
        audio (np.ndarray): Исходный массив аудио.

    Returns:
        tuple: Кортеж, содержащий:
            - np.ndarray: Нормализованный массив типа float64.
            - np.dtype: Исходный тип данных.
            - float: Коэффициент масштабирования.

    Raises:
        TypeError: При неподдерживаемом типе данных.
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


def denormalize_audio_from_float(
    audio_f: np.ndarray, orig_dtype: np.dtype
) -> np.ndarray:
    """Возвращает нормализованное аудио к исходному типу данных.

    Args:
        audio_f (np.ndarray): Нормализованное аудио в формате float.
        orig_dtype (np.dtype): Исходный тип данных для восстановления.

    Returns:
        np.ndarray: Денормализованное аудио в исходном формате.
    """
    audio_f = np.clip(audio_f, -1.0, 1.0)

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        return np.round(audio_f * info.max).astype(orig_dtype)

    return audio_f.astype(orig_dtype)


def phase_coding_encode_channel(
    x: np.ndarray, bits: np.ndarray, n_segments: int
) -> np.ndarray:
    """Встраивает информацию в один канал аудио методом фазового кодирования.

    Args:
        x (np.ndarray): Одномерный массив аудиосигнала.
        bits (np.ndarray): Массив битов для скрытия.
        n_segments (int): На сколько сегментов делить сигнал.

    Returns:
        np.ndarray: Модифицированный аудиосигнал с внедренным сообщением.

    Raises:
        ValueError: При неверной размерности или слишком длинном сообщении.
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
        phases_tilde[0, 1 : 1 + m] = msg_phases

    for i in range(1, n_segments):
        delta_phi = wrap_phase(phases[i] - phases[i - 1])
        phases_tilde[i] = wrap_phase(phases_tilde[i - 1] + delta_phi)

    spectra_tilde = magnitudes * np.exp(1j * phases_tilde)
    x_tilde_segments = np.fft.irfft(spectra_tilde, n=segment_len, axis=1)

    return x_tilde_segments.reshape(-1)[:n_samples]


def phase_coding_decode_channel(
    x_stego: np.ndarray, n_segments: int, message_len: int
) -> np.ndarray:
    """Извлечение сообщения из одного канала (фазовое декодирование).

    Args:
        x_stego (np.ndarray): Аудиосигнал со скрытым сообщением.
        n_segments (int): Изначальное число сегментов.
        message_len (int): Длина извлекаемого сообщения в битах.

    Returns:
        np.ndarray: Извлеченный массив битов.

    Raises:
        ValueError: При запрошенной длине, превышающей допустимый максимум.
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

    encoded_phases = phase_0[1 : 1 + message_len]
    return phases_to_bits(encoded_phases)


def phase_coding_encode(audio: np.ndarray, bits, n_segments: int) -> np.ndarray:
    """Встраивает информацию в аудиосигнал (mono или multi-channel).

    Args:
        audio (np.ndarray): Массив аудиоданных (float).
        bits (iterable): Биты для встраивания.
        n_segments (int): Число сегментов разбиения.

    Returns:
        np.ndarray: Аудиомассив со встроенным сообщением.

    Raises:
        ValueError: Если аудиофайл не содержит 1 или несколько каналов (неверная размерность).
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


def phase_coding_decode(
    audio_stego: np.ndarray, n_segments: int, message_len: int
) -> np.ndarray:
    """Извлекает битовую последовательность из скрытого аудио (mono или multi-channel).

    Для стерео или многоканального аудио извлекает сообщение из первого канала.

    Args:
        audio_stego (np.ndarray): Аудио со встроенной информацией.
        n_segments (int): Количество сегментов.
        message_len (int): Длина сообщения.

    Returns:
        np.ndarray: Извлеченные биты.

    Raises:
        ValueError: При некорректной размерности массива аудио.
    """
    if audio_stego.ndim == 1:
        return phase_coding_decode_channel(audio_stego, n_segments, message_len)

    if audio_stego.ndim != 2:
        raise ValueError("Поддерживаются только mono или multi-channel массивы аудио.")

    return phase_coding_decode_channel(audio_stego[:, 0], n_segments, message_len)


def write_wav(
    path: str, fs: int, audio_f: np.ndarray, orig_dtype: np.dtype, scale: float
) -> None:
    """Сохраняет вещественный массив аудио в WAV файл с преобразованием к исходному формату.

    Args:
        path (str): Путь сохранения файла.
        fs (int): Частота дискретизации.
        audio_f (np.ndarray): Нормализованные аудиоданные.
        orig_dtype (np.dtype): Исходный тип данных (для денормализации).
        scale (float): Использованный масштаб (не используется напрямую, но для совместимости).
    """
    audio_out = denormalize_audio_from_float(audio_f, orig_dtype)
    wavfile.write(path, fs, audio_out)


def parse_bits(bit_string: str) -> np.ndarray:
    """Преобразует строковое представление битов в массив np.uint8.

    Args:
        bit_string (str): Строка из '0' и '1'.

    Returns:
        np.ndarray: Массив типа np.uint8.

    Raises:
        ValueError: Если строка содержит недопустимые символы.
    """
    bit_string = bit_string.strip()
    if not bit_string:
        return np.array([], dtype=np.uint8)
    if any(c not in "01" for c in bit_string):
        raise ValueError("Строка битов должна содержать только символы 0 и 1.")
    return np.array([int(c) for c in bit_string], dtype=np.uint8)


if __name__ == "__main__":
    fs, audio = wavfile.read("tune_filtered.wav")

    audio_f, orig_dtype, scale = normalize_audio_to_float(audio)

    message = "1011001110001011"
    bits = parse_bits(message)

    n_segments = 32

    stego = phase_coding_encode(audio_f, bits, n_segments=n_segments)
    recovered_bits = phase_coding_decode(
        stego, n_segments=n_segments, message_len=len(bits)
    )

    recovered_message = "".join(map(str, recovered_bits.tolist()))

    print("Исходное сообщение :", message)
    print("Извлечённое сообщение:", recovered_message)
    print("Совпадение          :", message == recovered_message)

    # Сохранение файлов для проверки.
    write_wav("tune_filtered_original.wav", fs, audio_f, orig_dtype, scale)
    write_wav("tune_filtered_stego.wav", fs, stego, orig_dtype, scale)

    print("Файлы сохранены: tune_filtered_original.wav, tune_filtered_stego.wav")

    # Построение графиков для визуализации стеганографии
    plt.figure(figsize=(12, 8))

    # 1. Сравнение формы сигнала (увеличенный фрагмент)
    plt.subplot(2, 1, 1)
    samples_to_plot = 1000
    plt.plot(audio_f[:samples_to_plot], label="Оригинал", alpha=0.7)
    plt.plot(stego[:samples_to_plot], label="Стего", alpha=0.7, linestyle="--")
    plt.title("Сравнение формы сигнала (первые 1000 сэмплов)")
    plt.xlabel("Сэмплы")
    plt.ylabel("Амплитуда")
    plt.legend()

    # 2. Сравнение фаз в первом сегменте (где спрятаны биты)
    n_samples = len(audio_f)
    segment_len = int(np.ceil(n_samples / n_segments))

    orig_segment0 = audio_f[:segment_len]
    stego_segment0 = stego[:segment_len]

    orig_phase0 = np.angle(np.fft.rfft(orig_segment0))
    stego_phase0 = np.angle(np.fft.rfft(stego_segment0))

    plt.subplot(2, 1, 2)
    x_bins = np.arange(1, len(bits) + 1)
    plt.plot(x_bins, orig_phase0[1 : len(bits) + 1], "o-", label="Исходная фаза")
    plt.plot(
        x_bins, stego_phase0[1 : len(bits) + 1], "s-", label="Фаза стего (сообщение)"
    )
    plt.title(f"Сравнение фазы для {len(bits)} частотных бинов")
    plt.xlabel("Индекс частотного бина")
    plt.ylabel("Фаза (радианы)")
    plt.legend()
    plt.xticks(x_bins)

    plt.tight_layout()
    plt.show()
