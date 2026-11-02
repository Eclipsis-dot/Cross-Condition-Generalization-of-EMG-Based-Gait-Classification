# MOVEPORT — CROSS-CONDITION EMG CLASSIFICATION
#
# Research Question:
# How well does an EMG-based classifier trained to distinguish
# normal from abnormal gait under one experimental condition
# generalize to a different condition in previously unseen subjects?
#
# Subjects: 1–24
# Subject 25: EXCLUDED
# Modality: EMG ONLY
# Classifier: LDA
# Validation: LOSO
#
# Condition A:
#   treadmill_normal + treadmill_dragging
#
# Condition B:
#   treadmill_normal + treadmill_leghigh
#
# ================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ================================================================
# SETTINGS
# ================================================================

DATA_ROOT = "/content/MovePort"

RESULTS_DIR = "/content/MovePort/cross_condition_results"

os.makedirs(RESULTS_DIR, exist_ok=True)

SUBJECTS = list(range(1, 25))

NORMAL_CONDITION = "treadmill_normal"

CONDITION_A = "treadmill_dragging"

CONDITION_B = "treadmill_leghigh"


# ================================================================
# FEATURE EXTRACTION
# ================================================================

def extract_emg_features(window):

    """
    Extract 6 time-domain features from each EMG channel.

    Input:
        window = samples × 16 channels

    Output:
        96 features
        (16 channels × 6 features)
    """

    mav = np.mean(
        np.abs(window),
        axis=0
    )

    rms = np.sqrt(
        np.mean(window ** 2, axis=0)
    )

    var = np.var(
        window,
        axis=0
    )

    ssi = np.sum(
        window ** 2,
        axis=0
    )

    wl = np.sum(
        np.abs(np.diff(window, axis=0)),
        axis=0
    )

    zc = np.sum(
        (
            window[:-1] *
            window[1:] < 0
        )
        &
        (
            np.abs(
                window[:-1] -
                window[1:]
            ) > 1e-6
        ),
        axis=0
    )

    return np.hstack([
        mav,
        rms,
        var,
        ssi,
        wl,
        zc
    ])


# ================================================================
# FIND SUBJECT EMG FILES
# ================================================================

def find_condition_files(subject, condition):

    condition_path = os.path.join(
        DATA_ROOT,
        str(subject),
        str(subject),
        condition
    )

    if not os.path.exists(condition_path):

        return []

    files = []

    for file in os.listdir(condition_path):

        if (
            file.lower().endswith(".csv")
            and "emg" in file.lower()
        ):

            files.append(
                os.path.join(
                    condition_path,
                    file
                )
            )

    return sorted(files)


# ================================================================
# LOAD ONE EMG FILE
# ================================================================

def load_emg_file(file_path):

    df = pd.read_csv(
        file_path,
        header=None
    )

    # First column contains channel names
    # Remaining columns contain EMG samples

    data = df.iloc[:, 1:].apply(
        pd.to_numeric,
        errors="coerce"
    )

    data = data.to_numpy(
        dtype=float
    )

    # Original:
    # channels × samples
    #
    # Convert to:
    # samples × channels

    data = data.T

    return data


# ================================================================
# EXTRACT WINDOWS FROM A FILE
# ================================================================

def process_emg_file(
    file_path,
    label,
    subject
):

    data = load_emg_file(
        file_path
    )

    WINDOW_SIZE = 2000

    windows = []

    labels = []

    subjects = []

    n_windows = (
        data.shape[0]
        // WINDOW_SIZE
    )

    for i in range(n_windows):

        start = i * WINDOW_SIZE

        end = start + WINDOW_SIZE

        window = data[
            start:end
        ]

        if window.shape[0] != WINDOW_SIZE:
            continue

        features = extract_emg_features(
            window
        )

        windows.append(
            features
        )

        labels.append(
            label
        )

        subjects.append(
            subject
        )

    return (
        windows,
        labels,
        subjects
    )


# ================================================================
# BUILD A CONDITION DATASET
# ================================================================

def build_condition_dataset(
    abnormal_condition
):

    X = []

    y = []

    subjects = []

    print("\n")
    print("=" * 70)

    print(
        f"BUILDING DATASET:"
    )

    print(
        f"Normal + {abnormal_condition}"
    )

    print("=" * 70)


    for subject in SUBJECTS:

        # --------------------------------------------------------
        # NORMAL
        # --------------------------------------------------------

        normal_files = find_condition_files(
            subject,
            NORMAL_CONDITION
        )

        # --------------------------------------------------------
        # ABNORMAL
        # --------------------------------------------------------

        abnormal_files = find_condition_files(
            subject,
            abnormal_condition
        )


        print(
            f"Subject {subject:2d}: "
            f"Normal={len(normal_files):2d} | "
            f"Abnormal={len(abnormal_files):2d}"
        )


        # --------------------------------------------------------
        # PROCESS NORMAL
        # --------------------------------------------------------

        for file in normal_files:

            (
                features,
                labels,
                sub_ids
            ) = process_emg_file(
                file,
                "normal",
                subject
            )

            X.extend(features)

            y.extend(labels)

            subjects.extend(sub_ids)


        # --------------------------------------------------------
        # PROCESS ABNORMAL
        # --------------------------------------------------------

        for file in abnormal_files:

            (
                features,
                labels,
                sub_ids
            ) = process_emg_file(
                file,
                "abnormal",
                subject
            )

            X.extend(features)

            y.extend(labels)

            subjects.extend(sub_ids)


    X = np.asarray(X)

    y = np.asarray(y)

    subjects = np.asarray(subjects)


    print("\nDataset shape:", X.shape)

    print(
        "Classes:",
        np.unique(y)
    )

    print(
        "Class distribution:"
    )

    print(
        pd.Series(y).value_counts()
    )

    return X, y, subjects


# ================================================================
# PLOTTING HELPER
# ================================================================

def plot_loso_results(
    results_df,
    cm,
    train_condition,
    test_condition,
    output_dir
):

    safe_name = f"{train_condition}_TO_{test_condition}"

    # ------------------------------------------------------------
    # PER-SUBJECT ACCURACY BAR CHART
    # ------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        results_df["test_subject"].astype(str),
        results_df["accuracy"],
        color="steelblue"
    )

    ax.axhline(
        results_df["accuracy"].mean(),
        color="red",
        linestyle="--",
        label=f"Mean = {results_df['accuracy'].mean():.3f}"
    )

    ax.set_xlabel("Held-Out Subject")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"Per-Subject Accuracy\n"
        f"Train: {train_condition} → Test: {test_condition}"
    )
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{safe_name}_subject_accuracy.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)

    # ------------------------------------------------------------
    # CONFUSION MATRIX HEATMAP
    # ------------------------------------------------------------

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Normal", "Abnormal"]
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d", cmap="Blues")
    ax.set_title(
        f"Confusion Matrix\n"
        f"Train: {train_condition} → Test: {test_condition}"
    )
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_dir,
            f"{safe_name}_confusion_matrix.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)


# ================================================================
# CROSS-CONDITION LOSO FUNCTION
# ================================================================

def run_cross_condition_loso(
    X_train_all,
    y_train_all,
    subjects_train_all,

    X_test_all,
    y_test_all,
    subjects_test_all,

    train_condition,
    test_condition
):

    print("\n")
    print("=" * 70)

    print(
        f"TRAIN: {train_condition}"
    )

    print(
        f"TEST:  {test_condition}"
    )

    print("=" * 70)


    fold_results = []

    all_true = []

    all_pred = []


    # ============================================================
    # 24 SUBJECT LOSO
    # ============================================================

    for test_subject in SUBJECTS:

        print("\n")
        print(
            "-" * 70
        )

        print(
            f"HELD-OUT SUBJECT: "
            f"{test_subject}"
        )

        print(
            "-" * 70
        )


        # --------------------------------------------------------
        # TRAINING DATA
        #
        # IMPORTANT:
        # Remove held-out subject from training.
        # --------------------------------------------------------

        train_mask = (
            subjects_train_all
            != test_subject
        )

        test_mask = (
            subjects_test_all
            == test_subject
        )


        X_train = X_train_all[
            train_mask
        ]

        y_train = y_train_all[
            train_mask
        ]

        X_test = X_test_all[
            test_mask
        ]

        y_test = y_test_all[
            test_mask
        ]


        # --------------------------------------------------------
        # Skip if subject lacks data
        # --------------------------------------------------------

        if len(X_test) == 0:

            print(
                "No test data — skipping."
            )

            continue


        # --------------------------------------------------------
        # STANDARDIZATION
        #
        # Fit ONLY on training subjects.
        # --------------------------------------------------------

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        X_test_scaled = (
            scaler.transform(
                X_test
            )
        )


        # --------------------------------------------------------
        # TRAIN LDA
        # --------------------------------------------------------

        lda = LinearDiscriminantAnalysis()

        lda.fit(
            X_train_scaled,
            y_train
        )


        # --------------------------------------------------------
        # PREDICT
        # --------------------------------------------------------

        y_pred = lda.predict(
            X_test_scaled
        )


        # --------------------------------------------------------
        # METRICS
        # --------------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                y_pred
            )
        )

        macro_f1 = f1_score(
            y_test,
            y_pred,
            average="macro"
        )


        print(
            f"Accuracy: "
            f"{accuracy:.4f}"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy:.4f}"
        )

        print(
            f"Macro F1: "
            f"{macro_f1:.4f}"
        )


        fold_results.append({

            "train_condition":
                train_condition,

            "test_condition":
                test_condition,

            "test_subject":
                test_subject,

            "n_train_windows":
                len(X_train),

            "n_test_windows":
                len(X_test),

            "accuracy":
                accuracy,

            "balanced_accuracy":
                balanced_accuracy,

            "macro_f1":
                macro_f1
        })


        all_true.extend(
            y_test.tolist()
        )

        all_pred.extend(
            y_pred.tolist()
        )


    # ============================================================
    # RESULTS
    # ============================================================

    results_df = pd.DataFrame(
        fold_results
    )


    mean_accuracy = (
        results_df["accuracy"]
        .mean()
    )

    std_accuracy = (
        results_df["accuracy"]
        .std()
    )

    mean_balanced_accuracy = (
        results_df[
            "balanced_accuracy"
        ].mean()
    )

    std_balanced_accuracy = (
        results_df[
            "balanced_accuracy"
        ].std()
    )

    mean_macro_f1 = (
        results_df[
            "macro_f1"
        ].mean()
    )

    std_macro_f1 = (
        results_df[
            "macro_f1"
        ].std()
    )


    print("\n")
    print("=" * 70)

    print(
        f"FINAL RESULTS"
    )

    print(
        f"{train_condition}"
        f" → "
        f"{test_condition}"
    )

    print("=" * 70)

    print(
        f"Accuracy: "
        f"{mean_accuracy:.4f} ± "
        f"{std_accuracy:.4f}"
    )

    print(
        f"Balanced Accuracy: "
        f"{mean_balanced_accuracy:.4f} ± "
        f"{std_balanced_accuracy:.4f}"
    )

    print(
        f"Macro F1: "
        f"{mean_macro_f1:.4f} ± "
        f"{std_macro_f1:.4f}"
    )


    # ============================================================
    # CONFUSION MATRIX
    # ============================================================

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=[
            "normal",
            "abnormal"
        ]
    )


    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            cm,
            index=[
                "normal",
                "abnormal"
            ],
            columns=[
                "normal",
                "abnormal"
            ]
        )
    )


    # ============================================================
    # PLOTS
    # ============================================================

    plot_loso_results(
        results_df,
        cm,
        train_condition,
        test_condition,
        RESULTS_DIR
    )


    # ============================================================
    # SAVE RESULTS
    # ============================================================

    filename = (
        f"{train_condition}"
        f"_TO_"
        f"{test_condition}"
        "_LOSO.csv"
    )

    results_df.to_csv(
        os.path.join(
            RESULTS_DIR,
            filename
        ),
        index=False
    )


    return (
        results_df,
        cm
    )


# ================================================================
# BUILD CONDITION A
# ================================================================

X_A, y_A, subjects_A = build_condition_dataset(
    CONDITION_A
)


# ================================================================
# BUILD CONDITION B
# ================================================================

X_B, y_B, subjects_B = build_condition_dataset(
    CONDITION_B
)


# ================================================================
# EXPERIMENT 1
#
# NORMAL + DRAGGING
#              ↓
# NORMAL + LEG-HIGH
# ================================================================

results_A_to_B, cm_A_to_B = (
    run_cross_condition_loso(
        X_A,
        y_A,
        subjects_A,

        X_B,
        y_B,
        subjects_B,

        CONDITION_A,
        CONDITION_B
    )
)


# ================================================================
# EXPERIMENT 2
#
# NORMAL + LEG-HIGH
#              ↓
# NORMAL + DRAGGING
# ================================================================

results_B_to_A, cm_B_to_A = (
    run_cross_condition_loso(
        X_B,
        y_B,
        subjects_B,

        X_A,
        y_A,
        subjects_A,

        CONDITION_B,
        CONDITION_A
    )
)


# ================================================================
# SUMMARY COMPARISON CHART: A→B vs B→A
# ================================================================

summary_data = {
    f"{CONDITION_A}\n→ {CONDITION_B}": results_A_to_B,
    f"{CONDITION_B}\n→ {CONDITION_A}": results_B_to_A
}

metrics = ["accuracy", "balanced_accuracy", "macro_f1"]
labels = ["Accuracy", "Balanced Accuracy", "Macro F1"]

fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(summary_data))
width = 0.25

for i, (metric, label) in enumerate(zip(metrics, labels)):

    means = [
        df[metric].mean()
        for df in summary_data.values()
    ]

    stds = [
        df[metric].std()
        for df in summary_data.values()
    ]

    ax.bar(
        x + i * width,
        means,
        width,
        yerr=stds,
        capsize=4,
        label=label
    )

ax.set_xticks(x + width)
ax.set_xticklabels(summary_data.keys())
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
ax.set_title("Cross-Condition Generalization Summary")
ax.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULTS_DIR,
        "cross_condition_summary.png"
    ),
    dpi=300,
    bbox_inches="tight"
)
plt.show()
plt.close(fig)


print("\n")
print("=" * 70)
print("CROSS-CONDITION EXPERIMENT COMPLETE")
print("=" * 70)
print(f"\nAll results and figures saved to:\n{RESULTS_DIR}")
