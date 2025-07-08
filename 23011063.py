import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import scipy.io.wavfile as wavfile
from scipy import signal
import os

# --- myConv Fonksiyonları ve Kullanımı --- 

def myConv_discrete(x, start_x, y, start_y):
    """İki ayrık zamanlı sinyalin konvolüsyonunu hesaplar."""
    n = len(x)
    m = len(y)
    conv_length = n + m - 1
    result = [0] * conv_length
    start_result = start_x + start_y

    for i in range(n):
        for j in range(m):
            result[i + j] += x[i] * y[j]

    return result, start_result

def plot_signals(x, start_x, y, start_y, my_conv_result, start_my_conv, np_conv_result, start_np_conv, dataset_num):
    """Sinyalleri ve konvolüsyon sonuçlarını çizer."""
    n = len(x)
    m = len(y)
    len_my_conv = len(my_conv_result)
    len_np_conv = len(np_conv_result)

    n_indices = np.arange(start_x, start_x + n)
    m_indices = np.arange(start_y, start_y + m)
    my_conv_indices = np.arange(start_my_conv, start_my_conv + len_my_conv)
    np_conv_indices = np.arange(start_np_conv, start_np_conv + len_np_conv)

    plt.figure(figsize=(12, 8))
    plt.suptitle(f'Veri Seti {dataset_num} Karşılaştırması')

    plt.subplot(2, 2, 1)
    plt.stem(n_indices, x, basefmt=" ")
    plt.title('x[n]')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.stem(m_indices, y, basefmt=" ")
    plt.title('y[m]')
    plt.xlabel('m')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.stem(my_conv_indices, my_conv_result, linefmt='g-', markerfmt='go', basefmt=" ")
    plt.title('myConv_discrete Sonucu')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.stem(np_conv_indices, np_conv_result, linefmt='r-', markerfmt='ro', basefmt=" ")
    plt.title('numpy.convolve Sonucu')
    plt.xlabel('n')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def run_myConv_example():
    """Kullanıcıdan girdi alarak myConv_discrete ve numpy.convolve karşılaştırmasını yapar."""
    print("--- Ayrık Zamanlı Konvolüsyon Örneği ---")
    for dataset_num in range(1, 2):
        print(f"\n--- Veri Seti {dataset_num} ---")
        print("Lütfen her dizi için en fazla 5 eleman giriniz.")

        while True:
            try:
                n = int(input("x dizisinin boyutunu giriniz (max 5): "))
                if 1 <= n <= 5: break
                else: print("Boyut 1 ile 5 arasında olmalıdır.")
            except ValueError: print("Geçersiz giriş. Lütfen bir sayı girin.")
        start_x = int(input("x dizisinin başlangıç indisini (n=0 noktası) giriniz: "))
        print("x dizisinin elemanlarını giriniz:")
        x = [float(input(f"x[{i+start_x}] = ")) for i in range(n)]

        while True:
            try:
                m = int(input("y dizisinin boyutunu giriniz (max 5): "))
                if 1 <= m <= 5: break
                else: print("Boyut 1 ile 5 arasında olmalıdır.")
            except ValueError: print("Geçersiz giriş. Lütfen bir sayı girin.")
        start_y = int(input("y dizisinin başlangıç indisini (m=0 noktası) giriniz: "))
        print("y dizisinin elemanlarını giriniz:")
        y = [float(input(f"y[{i+start_y}] = ")) for i in range(m)]

        my_conv_result, start_my_conv = myConv_discrete(x, start_x, y, start_y)
        np_conv_result = np.convolve(x, y)
        start_np_conv = start_x + start_y

        print("\n--- Vektörel Gösterim ---")
        print("x[n]:", [f"x[{i+start_x}]={val}" for i, val in enumerate(x)])
        print("y[m]:", [f"y[{i+start_y}]={val}" for i, val in enumerate(y)])
        print("myConv_discrete Sonucu:", [f"res[{i+start_my_conv}]={val}" for i, val in enumerate(my_conv_result)])
        print("numpy.convolve Sonucu:", [f"res[{i+start_np_conv}]={val}" for i, val in enumerate(np_conv_result)])

        print("\n--- Grafiksel Gösterim --- (Grafik penceresini kapatarak devam edebilirsiniz)")
        plot_signals(x, start_x, y, start_y, my_conv_result, start_my_conv, np_conv_result, start_np_conv, dataset_num)

# --- Ses Kayıt Fonksiyonları --- 

def record_audio(duration, filename, samplerate=44100):
    """Belirtilen sürede ses kaydı yapar ve dosyaya kaydeder."""
    print(f"\n{duration} saniyelik kayıt başlıyor... '{filename}'")
    print("Konuşmaya başlayın.")
    audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print(f"{duration} saniyelik kayıt tamamlandı.")
    wavfile.write(filename, samplerate, audio_data)
    print(f"Kayıt başarıyla '{filename}' olarak kaydedildi.")
    return filename, samplerate, audio_data

# --- Ses İşleme Fonksiyonları --- 

A_const = 0.5 # Sabit A değeri

def uygula_denklem(x, M, A=A_const):
    """Verilen fark denklemini uygular."""
    y = np.copy(x)
    for k in range(1, M + 1):
        delay = 400 * k
        delayed_x = np.zeros_like(x)
        if delay < len(x):
            delayed_x[delay:] = x[:len(x) - delay]
        y += A * k * delayed_x
    return y

def process_audio(input_filename, samplerate, M_values):
    """Ses dosyasını işler ve sonuçları kaydeder."""
    print(f"\n--- '{input_filename}' Ses Dosyası İşleniyor ---")
    
    try:
        _, x = wavfile.read(input_filename)
        x = x.astype(np.float32)
        if np.max(np.abs(x)) > 0:
             x = x / np.max(np.abs(x))
        else:
            print(f"Uyarı: '{input_filename}' boş veya sessiz.")
            return

    except Exception as e:
        print(f"Hata: '{input_filename}' okunurken sorun oluştu: {e}")
        return

    base_filename = os.path.splitext(input_filename)[0]

    processed_files = {}

    for M in M_values:
        print(f"M = {M} için işlemler yapılıyor...")
        
        # Sistem yanıtını oluştur (Konvolüsyon için)
        h = np.zeros(400 * M + 1)
        h[0] = 1
        for k in range(1, M + 1):
            h[400 * k] = A_const * k
        
        # Hazır konvolüsyon (SciPy)
        Y = signal.convolve(x, h, mode='full') # mode='full' varsayılan ve genellikle istenen
        # Normalize et (konvolüsyon sonrası genlik artabilir)
        if np.max(np.abs(Y)) > 0:
            Y = Y / np.max(np.abs(Y))
        output_filename_conv = f"{base_filename}_conv_M{M}.wav"
        wavfile.write(output_filename_conv, samplerate, Y.astype(np.float32))
        print(f"  Konvolüsyon sonucu kaydedildi: {output_filename_conv}")
        processed_files[f"Hazır Konv. (M={M})"] = (output_filename_conv, samplerate)

        Y_direct = uygula_denklem(x, M)
        if np.max(np.abs(Y_direct)) > 0:
            Y_direct = Y_direct / np.max(np.abs(Y_direct))
        output_filename_direct = f"{base_filename}_direct_M{M}.wav"
        wavfile.write(output_filename_direct, samplerate, Y_direct.astype(np.float32))
        print(f"  Doğrudan denklem sonucu kaydedildi: {output_filename_direct}")
        processed_files[f"MyConv (M={M})"] = (output_filename_direct, samplerate)

    return processed_files

# --- Seslendirme Fonksiyonları --- 

def play_sound(audio_data, sample_rate):
    """Ses verisini çalar."""
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Ses çalınırken hata: {e}")

def seslendirme_menu(original_files, processed_files_all):
    """Kullanıcının dinlemek istediği sesi seçmesini sağlar."""
    print("\n=== Seslendirme Menüsü ===")
    
    options = {}
    current_option = 1

    print("--- Orijinal Kayıtlar ---")
    for name, (filename, rate) in original_files.items():
        print(f"{current_option}. {name}")
        options[current_option] = (filename, rate)
        current_option += 1

    print("\n--- İşlenmiş Kayıtlar ---")
    for base_name, processed_dict in processed_files_all.items():
         print(f"  -- {base_name} için işlenmişler --")
         if not processed_dict: # Eğer işleme sırasında hata olduysa
             print("    (İşlenemedi veya hata oluştu)")
             continue
         for name, (filename, rate) in processed_dict.items():
             print(f"    {current_option}. {name}")
             options[current_option] = (filename, rate)
             current_option += 1

    print("\n0. Çıkış")

    while True:
        try:
            choice_str = input(f"\nLütfen dinlemek istediğiniz sesi seçin (0-{current_option - 1}): ")
            choice = int(choice_str)

            if choice == 0:
                break
            elif choice in options:
                filename, rate = options[choice]
                print(f"'{filename}' çalınıyor...")
                try:
                    sr, audio = wavfile.read(filename)
                    play_sound(audio, sr) 
                except FileNotFoundError:
                    print(f"Hata: '{filename}' bulunamadı.")
                except Exception as e:
                    print(f"Hata: '{filename}' okunurken veya çalınırken sorun oluştu: {e}")
            else:
                print("Geçersiz seçim!")
        except ValueError:
            print("Lütfen geçerli bir sayı girin!")
        except Exception as e:
            print(f"Beklenmedik bir hata oluştu: {e}")

# --- Ana Çalıştırma Bloğu --- 

if __name__ == "__main__":
    # 1. Ayrık Zamanlı Konvolüsyon Örneği
    run_myConv_example()

    # 2. Ses Kayıtları
    print("\n--- Ses Kayıt İşlemleri ---")
    original_audio_files = {}
    filename_5s, rate_5s, _ = record_audio(5, "kayit_5sn_birlesik.wav")
    original_audio_files["Orijinal 5sn"] = (filename_5s, rate_5s)
    
    filename_10s, rate_10s, _ = record_audio(10, "kayit_10sn_birlesik.wav")
    original_audio_files["Orijinal 10sn"] = (filename_10s, rate_10s)

    # 3. Ses İşleme
    print("\n--- Ses İşleme İşlemleri ---")
    M_values_audio = [3, 4, 5]
    all_processed_results = {}
    
    processed_5s = process_audio(filename_5s, rate_5s, M_values_audio)
    all_processed_results[filename_5s] = processed_5s
    
    processed_10s = process_audio(filename_10s, rate_10s, M_values_audio)
    all_processed_results[filename_10s] = processed_10s

    # 4. Seslendirme Menüsü
    seslendirme_menu(original_audio_files, all_processed_results)

    print("\nProgram tamamlandı.")