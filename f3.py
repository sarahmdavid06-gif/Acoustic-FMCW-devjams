import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# ACOUSTIC FMCW RADAR - ROUND 1 COMPLETE PROOF OF CONCEPT
# ============================================================
# One-file demonstration:
#
#   FMCW chirp
#       ↓
#   simulated echo
#       ↓
#   dechirp
#       ↓
#   Range FFT
#       ↓
#   Doppler FFT
#       ↓
#   2-D CA-CFAR
#       ↓
#   detected range + velocity
#       ↓
#   TTC
#       ↓
#   SAFE / CAUTION / COLLISION WARNING
#
# IMPORTANT:
# This is a SOFTWARE SIMULATION for Round 1.
# It does NOT yet read a physical ESP32/ADC.
#
# Current acoustic timing:
#   Tc = 20 ms
#   PRF = 50 Hz
#   unambiguous velocity ≈ +/- 0.107 m/s
#
# Therefore the DSP demo uses a small radial velocity such as
# -0.05 m/s. The vehicle scenario mode separately demonstrates
# realistic road speeds and TTC logic.
# ============================================================


# ============================================================
# RADAR PARAMETERS
# ============================================================

F0 = 35_000.0
F1 = 45_000.0
B = F1 - F0
TC = 0.020
FS = 96_000.0
C = 343.0

FC = (F0 + F1) / 2
LAMBDA = C / FC

SAMPLES_PER_CHIRP = int(round(FS * TC))

N_CHIRPS = 64
NFFT_RANGE = 2048
NFFT_DOPPLER = 64

# ============================================================
# CFAR PARAMETERS
# ============================================================

TRAIN_RANGE = 8
GUARD_RANGE = 3

TRAIN_DOPPLER = 4
GUARD_DOPPLER = 2

P_FALSE_ALARM = 1e-3

# ============================================================
# SAFETY PARAMETERS
# ============================================================

TTC_CAUTION = 3.0
TTC_WARNING = 1.5


# ============================================================
# UTILITY
# ============================================================

def ask_float(prompt, minimum=None):
    while True:
        try:
            value = float(input(prompt))

            if minimum is not None and value < minimum:
                print(f"Please enter a value >= {minimum}.")
                continue

            return value

        except ValueError:
            print("Please enter a number.")


# ============================================================
# FMCW CHIRP
# ============================================================

def make_chirp():

    t = np.arange(SAMPLES_PER_CHIRP) / FS

    k = B / TC

    phase = 2 * np.pi * (
        F0 * t +
        0.5 * k * t**2
    )

    tx = np.exp(1j * phase)

    return t, tx


# ============================================================
# SIMULATE ECHO
# ============================================================

def simulate_echo(
    tx,
    target_range,
    target_velocity,
    chirp_index,
    rng
):

    delay = 2 * target_range / C

    delay_samples = int(
        round(delay * FS)
    )

    rx = np.zeros(
        len(tx),
        dtype=complex
    )

    if 0 < delay_samples < len(tx):

        # Delayed copy of the transmitted chirp.
        rx[delay_samples:] = (
            tx[:-delay_samples]
        )

    # Doppler phase progression across chirps.
    phase_shift = (
        4 * np.pi *
        target_velocity *
        TC *
        chirp_index /
        LAMBDA
    )

    rx *= np.exp(
        1j * phase_shift
    )

    # Add complex receiver noise.
    noise = 0.015 * (
        rng.normal(size=len(tx))
        +
        1j * rng.normal(size=len(tx))
    )

    rx += noise

    return rx, delay, delay_samples


# ============================================================
# DECHIRP
# ============================================================

def dechirp(tx, rx):

    return tx * np.conj(rx)


# ============================================================
# RANGE-DOPPLER PROCESSING
# ============================================================

def build_range_doppler(
    target_range,
    target_velocity
):

    t, tx = make_chirp()

    rng = np.random.default_rng(1234)

    beat_matrix = np.zeros(
        (N_CHIRPS, SAMPLES_PER_CHIRP),
        dtype=complex
    )

    delay = None
    delay_samples = None

    # --------------------------------------------------------
    # Generate repeated chirps.
    # --------------------------------------------------------

    for m in range(N_CHIRPS):

        rx, delay, delay_samples = simulate_echo(
            tx,
            target_range,
            target_velocity,
            m,
            rng
        )

        beat_matrix[m] = dechirp(
            tx,
            rx
        )

    # --------------------------------------------------------
    # Fast-time FFT -> range.
    # --------------------------------------------------------

    fast_window = np.hanning(
        SAMPLES_PER_CHIRP
    )

    range_fft = np.fft.fft(
        beat_matrix * fast_window[None, :],
        n=NFFT_RANGE,
        axis=1
    )

    range_fft = range_fft[
        :, :NFFT_RANGE // 2
    ]

    beat_frequencies = (
        np.arange(NFFT_RANGE // 2)
        * FS /
        NFFT_RANGE
    )

    ranges = (
        beat_frequencies *
        C *
        TC /
        (2 * B)
    )

    # --------------------------------------------------------
    # Slow-time FFT -> Doppler.
    # --------------------------------------------------------

    slow_window = np.hanning(
        N_CHIRPS
    )

    windowed = (
        range_fft *
        slow_window[:, None]
    )

    doppler_fft = np.fft.fft(
        windowed,
        n=NFFT_DOPPLER,
        axis=0
    )

    doppler_fft = np.fft.fftshift(
        doppler_fft,
        axes=0
    )

    doppler_frequencies = np.fft.fftshift(
        np.fft.fftfreq(
            NFFT_DOPPLER,
            d=TC
        )
    )

    # With TX * conj(RX), invert the sign so approaching
    # targets are represented by negative velocity.
    velocities = (
        -doppler_frequencies *
        LAMBDA / 2
    )

    power = np.abs(
        doppler_fft
    ) ** 2

    power_db = 10 * np.log10(
        power + 1e-12
    )

    return (
        t,
        tx,
        beat_matrix,
        ranges,
        velocities,
        power,
        power_db,
        range_fft,
        delay,
        delay_samples
    )


# ============================================================
# 2-D CA-CFAR
# ============================================================

def ca_cfar_2d(power):

    rows, cols = power.shape

    detections = np.zeros(
        power.shape,
        dtype=bool
    )

    threshold_map = np.zeros_like(
        power
    )

    half_v = (
        TRAIN_DOPPLER +
        GUARD_DOPPLER
    )

    half_r = (
        TRAIN_RANGE +
        GUARD_RANGE
    )

    total_cells = (
        (2 * half_v + 1) *
        (2 * half_r + 1)
    )

    guard_cells = (
        (2 * GUARD_DOPPLER + 1) *
        (2 * GUARD_RANGE + 1)
    )

    training_cells = (
        total_cells -
        guard_cells
    )

    alpha = training_cells * (
        P_FALSE_ALARM **
        (-1.0 / training_cells)
        - 1
    )

    for v in range(
        half_v,
        rows - half_v
    ):

        for r in range(
            half_r,
            cols - half_r
        ):

            window = power[
                v - half_v:v + half_v + 1,
                r - half_r:r + half_r + 1
            ].copy()

            # Remove guard cells + CUT.
            center_v = half_v
            center_r = half_r

            window[
                center_v - GUARD_DOPPLER:
                center_v + GUARD_DOPPLER + 1,

                center_r - GUARD_RANGE:
                center_r + GUARD_RANGE + 1
            ] = np.nan

            noise_estimate = np.nanmean(
                window
            )

            threshold = (
                alpha *
                noise_estimate
            )

            threshold_map[v, r] = threshold

            if power[v, r] > threshold:
                detections[v, r] = True

    return detections, threshold_map


# ============================================================
# SELECT BEST DETECTION
# ============================================================

def select_best_detection(
    power,
    detections,
    ranges,
    velocities
):

    indices = np.argwhere(
        detections
    )

    if len(indices) == 0:
        return None

    values = power[
        detections
    ]

    best = np.argmax(
        values
    )

    v_index, r_index = (
        indices[best]
    )

    return (
        ranges[r_index],
        velocities[v_index],
        v_index,
        r_index,
        len(indices)
    )


# ============================================================
# TTC
# ============================================================

def calculate_ttc(
    distance,
    velocity
):

    # Negative velocity = approaching.
    if velocity < 0:

        closing_speed = abs(
            velocity
        )

        if closing_speed > 1e-9:
            return (
                distance / closing_speed,
                closing_speed
            )

    return np.inf, 0.0


def safety_status(ttc):

    if not np.isfinite(ttc):
        return "SAFE"

    if ttc < TTC_WARNING:
        return "COLLISION WARNING"

    if ttc < TTC_CAUTION:
        return "CAUTION"

    return "SAFE"


# ============================================================
# MODE 1: RADAR DSP DEMO
# ============================================================

def radar_dsp_demo():

    print()
    print("=" * 72)
    print("MODE 1 - ACOUSTIC FMCW RADAR DSP")
    print("=" * 72)

    print()
    print("Enter a target for the simulated radar.")
    print("Negative radial velocity = approaching.")
    print(
        f"Valid Doppler demonstration range is approximately "
        f"+/- {LAMBDA / (4 * TC):.3f} m/s."
    )
    print()

    target_range = ask_float(
        "Target range (m) [recommended 0.75]: ",
        0.1
    )

    target_velocity = ask_float(
        "Target radial velocity (m/s) [recommended -0.05]: "
    )

    vmax = LAMBDA / (4 * TC)

    if abs(target_velocity) >= vmax:

        print()
        print(
            f"WARNING: velocity is outside the current "
            f"unambiguous range of +/- {vmax:.3f} m/s."
        )
        print(
            "For the clean review demonstration, use "
            "approximately -0.05 m/s."
        )

    (
        t,
        tx,
        beat_matrix,
        ranges,
        velocities,
        power,
        power_db,
        range_fft,
        delay,
        delay_samples
    ) = build_range_doppler(
        target_range,
        target_velocity
    )

    # --------------------------------------------------------
    # CA-CFAR
    # --------------------------------------------------------

    print()
    print("Processing...")
    print("  [1/4] Dechirping              DONE")
    print("  [2/4] Range FFT               DONE")
    print("  [3/4] Doppler FFT             DONE")
    print("  [4/4] 2-D CA-CFAR             RUNNING")

    detections, threshold_map = ca_cfar_2d(
        power
    )

    print("  [4/4] 2-D CA-CFAR             DONE")

    result = select_best_detection(
        power,
        detections,
        ranges,
        velocities
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 72)
    print("RADAR RESULTS")
    print("=" * 72)

    print(
        f"True range             : "
        f"{target_range:.3f} m"
    )

    print(
        f"True radial velocity   : "
        f"{target_velocity:.4f} m/s"
    )

    print(
        f"Echo delay             : "
        f"{delay * 1000:.3f} ms"
    )

    print(
        f"Echo delay samples     : "
        f"{delay_samples}"
    )

    print(
        f"Data matrix            : "
        f"{beat_matrix.shape}"
    )

    if result is None:

        print()
        print("CFAR RESULT            : NO TARGET DETECTED")

    else:

        detected_range = result[0]
        detected_velocity = result[1]
        number_detections = result[4]

        ttc, closing_speed = calculate_ttc(
            detected_range,
            detected_velocity
        )

        status = safety_status(
            ttc
        )

        print()
        print(
            f"CFAR candidate cells   : "
            f"{number_detections}"
        )

        print(
            f"Selected target        : "
            f"{detected_range:.3f} m"
        )

        print(
            f"Detected velocity      : "
            f"{detected_velocity:.4f} m/s"
        )

        print(
            f"Range error            : "
            f"{abs(detected_range - target_range):.3f} m"
        )

        print(
            f"Velocity error         : "
            f"{abs(detected_velocity - target_velocity):.4f} m/s"
        )

        print(
            "Note: multiple adjacent CFAR cells can represent "
            "the same physical target."
        )

        print()

        if np.isfinite(ttc):

            print(
                f"Closing speed          : "
                f"{closing_speed:.4f} m/s"
            )

            print(
                f"TTC                    : "
                f"{ttc:.2f} s"
            )

        else:

            print(
                "TTC                    : N/A"
            )

        print(
            f"TTC warning threshold  : "
            f"{TTC_WARNING:.2f} s"
        )

        print(
            f"STATUS                 : "
            f"{status}"
        )

    print("=" * 72)

    # --------------------------------------------------------
    # Plot 1 - FMCW chirp
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    show = int(
        0.002 * FS
    )

    plt.plot(
        t[:show] * 1000,
        np.real(tx[:show])
    )

    plt.xlabel(
        "Time (ms)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "35–45 kHz Acoustic FMCW Chirp"
    )

    plt.grid()

    plt.tight_layout()

    # --------------------------------------------------------
    # Plot 2 - Range profile
    # --------------------------------------------------------

    range_profile = np.max(
        np.abs(range_fft),
        axis=0
    )

    plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        ranges,
        range_profile
    )

    plt.axvline(
        target_range,
        linestyle="--",
        label="True range"
    )

    if result is not None:

        plt.axvline(
            result[0],
            linestyle=":",
            label="CFAR detected range"
        )

    plt.xlim(
        0,
        min(2.0, ranges[-1])
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Magnitude"
    )

    plt.title(
        "Range FFT"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # Plot 3 - Range-Doppler map
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    # Limit display to useful short range.
    display_mask = (
        ranges <= 2.0
    )

    display_ranges = ranges[
        display_mask
    ]

    display_power = power_db[
        :, display_mask
    ]

    plt.imshow(
        display_power,
        aspect="auto",
        origin="lower",
        extent=[
            display_ranges[0],
            display_ranges[-1],
            velocities[0],
            velocities[-1]
        ]
    )

    plt.scatter(
        [target_range],
        [target_velocity],
        marker="x",
        s=120,
        label="True target"
    )

    if result is not None:

        plt.scatter(
            [result[0]],
            [result[1]],
            marker="o",
            facecolors="none",
            s=160,
            linewidths=2,
            label="CFAR selected target"
        )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Radial velocity (m/s)"
    )

    plt.title(
        "Range-Doppler Map"
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # Plot 4 - CA-CFAR result
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.imshow(
        display_power,
        aspect="auto",
        origin="lower",
        extent=[
            display_ranges[0],
            display_ranges[-1],
            velocities[0],
            velocities[-1]
        ]
    )

    detection_indices = np.argwhere(
        detections
    )

    if len(detection_indices):

        detection_ranges = ranges[
            detection_indices[:, 1]
        ]

        detection_velocities = velocities[
            detection_indices[:, 0]
        ]

        display_detection = (
            detection_ranges <= 2.0
        )

        # For presentation, show only CFAR candidates in a small
        # neighborhood around the strongest detected target. The full
        # CA-CFAR calculation is still performed on the entire map.
        if result is not None:
            near_target = (
                np.abs(
                    detection_ranges - result[0]
                ) <= 0.08
            ) & (
                np.abs(
                    detection_velocities - result[1]
                ) <= 0.015
            )
            display_detection &= near_target

        plt.scatter(
            detection_ranges[
                display_detection
            ],
            detection_velocities[
                display_detection
            ],
            marker="x",
            s=35,
            label="CA-CFAR candidates"
        )

    plt.scatter(
        [target_range],
        [target_velocity],
        marker="o",
        facecolors="none",
        s=160,
        linewidths=2,
        label="True target"
    )

    plt.xlabel(
        "Range (m)"
    )

    plt.ylabel(
        "Radial velocity (m/s)"
    )

    plt.title(
        "CA-CFAR Target Detection"
    )

    plt.colorbar(
        label="Power (dB)"
    )

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MODE 2: REALISTIC VEHICLE TTC SCENARIO
# ============================================================

def vehicle_scenario():

    print()
    print("=" * 72)
    print("MODE 2 - VEHICLE COLLISION SCENARIO")
    print("=" * 72)

    print()
    print("The radar is mounted on the front of your vehicle.")
    print()
    print("       YOUR VEHICLE                 OBSTACLE")
    print("           🚗  ------------------------ 🚙")
    print()
    print("This mode demonstrates vehicle-level TTC logic.")
    print("It is separate from the current low-speed acoustic")
    print("Doppler measurement limit of the DSP demo.")
    print()

    ego_speed_kmh = ask_float(
        "Your vehicle speed (km/h): ",
        0
    )

    obstacle_distance = ask_float(
        "Initial obstacle distance (m): ",
        0.1
    )

    obstacle_speed_kmh = ask_float(
        "Obstacle speed (km/h): ",
        0
    )

    obstacle_acceleration = ask_float(
        "Obstacle acceleration (m/s^2, 0 = constant): "
    )

    duration = ask_float(
        "Scenario duration (s): ",
        0.1
    )

    ego_speed = (
        ego_speed_kmh / 3.6
    )

    obstacle_speed = (
        obstacle_speed_kmh / 3.6
    )

    times = np.arange(
        0,
        duration + 0.05,
        0.05
    )

    distance = obstacle_distance

    distance_history = []
    closing_history = []
    ttc_history = []
    status_history = []

    warning_time = None

    for t in times:

        # Update obstacle speed.
        obstacle_speed += (
            obstacle_acceleration *
            0.05
        )

        obstacle_speed = max(
            0,
            obstacle_speed
        )

        # Positive closing speed means ego vehicle
        # is catching the obstacle.
        closing_speed = (
            ego_speed -
            obstacle_speed
        )

        distance -= (
            closing_speed *
            0.05
        )

        distance = max(
            0,
            distance
        )

        if (
            closing_speed > 0
            and distance > 0
        ):

            ttc = (
                distance /
                closing_speed
            )

        else:

            ttc = np.inf

        status = safety_status(
            ttc
        )

        if (
            warning_time is None
            and status == "COLLISION WARNING"
        ):

            warning_time = t

        distance_history.append(
            distance
        )

        closing_history.append(
            closing_speed
        )

        ttc_history.append(
            ttc
        )

        status_history.append(
            status
        )

    distance_history = np.array(
        distance_history
    )

    closing_history = np.array(
        closing_history
    )

    ttc_history = np.array(
        ttc_history
    )

    print()
    print("=" * 72)
    print("VEHICLE SCENARIO RESULTS")
    print("=" * 72)

    print(
        f"Initial gap            : "
        f"{obstacle_distance:.2f} m"
    )

    print(
        f"Ego speed              : "
        f"{ego_speed_kmh:.1f} km/h"
    )

    print(
        f"Obstacle speed         : "
        f"{obstacle_speed_kmh:.1f} km/h"
    )

    print(
        f"Obstacle acceleration  : "
        f"{obstacle_acceleration:.2f} m/s²"
    )

    print(
        f"Final distance         : "
        f"{distance_history[-1]:.2f} m"
    )

    print(
        f"Final closing speed    : "
        f"{closing_history[-1]:.2f} m/s"
    )

    if np.isfinite(ttc_history[-1]):

        print(
            f"Final TTC              : "
            f"{ttc_history[-1]:.2f} s"
        )

    else:

        print(
            "Final TTC              : N/A"
        )

    print(
        f"Final status           : "
        f"{status_history[-1]}"
    )

    if warning_time is not None:

        print(
            f"FIRST WARNING          : "
            f"{warning_time:.2f} s"
        )

    print("=" * 72)

    # --------------------------------------------------------
    # Distance plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    plt.plot(
        times,
        distance_history
    )

    plt.axhline(
        0,
        linestyle="--",
        label="Collision"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Distance to obstacle (m)"
    )

    plt.title(
        "Vehicle-to-Obstacle Distance"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    # --------------------------------------------------------
    # TTC plot
    # --------------------------------------------------------

    plt.figure(
        figsize=(8, 4)
    )

    plot_ttc = ttc_history.copy()

    plot_ttc[
        ~np.isfinite(plot_ttc)
    ] = np.nan

    plt.plot(
        times,
        plot_ttc
    )

    plt.axhline(
        TTC_CAUTION,
        linestyle="--",
        label="Caution = 3 s"
    )

    plt.axhline(
        TTC_WARNING,
        linestyle="--",
        label="Warning = 1.5 s"
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "TTC (s)"
    )

    plt.title(
        "Time-to-Collision"
    )

    plt.grid()

    plt.legend()

    plt.tight_layout()

    plt.show()


# ============================================================
# MAIN MENU
# ============================================================

def main():

    print()
    print("=" * 72)
    print("       ACOUSTIC FMCW RADAR - ROUND 1 DEMONSTRATOR")
    print("=" * 72)

    print()
    print("1. Radar DSP demo")
    print("   FMCW -> Range -> Doppler -> CA-CFAR -> TTC")
    print()
    print("2. Realistic vehicle scenario")
    print("   Vehicle motion -> closing speed -> TTC -> warning")
    print()

    choice = input(
        "Select mode [1/2]: "
    ).strip()

    if choice == "1":

        radar_dsp_demo()

    elif choice == "2":

        vehicle_scenario()

    else:

        print(
            "Invalid choice. Please run again and select 1 or 2."
        )


if __name__ == "__main__":
    main()
