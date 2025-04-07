import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import os # Added for directory creation

# 경로 설정
DATA_PATH = "E:/SpliceImageTextData/dataset/word_image_train_label2.h5"
MODEL_SAVE_PATH = "E:/SpliceImageTextData/model/unet_4channel_output_model.h5" # Modified name slightly for clarity
MODEL_SAVE_DIR = os.path.dirname(MODEL_SAVE_PATH) # Get directory path

# 데이터셋 분할 비율
TEST_SIZE = 0.1
BATCH_SIZE = 16 # You might need to adjust this based on your GPU memory
SEED = 42
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 4
LABEL_CHANNELS = 4

# 모델 저장 디렉토리 생성 (없으면)
if not os.path.exists(MODEL_SAVE_DIR):
    os.makedirs(MODEL_SAVE_DIR)
    print(f"Created directory: {MODEL_SAVE_DIR}")

# 전체 데이터 개수를 미리 얻기 위해 h5py 파일을 한 번 열어서 shape 확인
try:
    with h5py.File(DATA_PATH, 'r') as f:
        # Check if keys exist before accessing shape
        if 'korean_image' not in f:
             raise KeyError("Dataset 'korean_image' not found in HDF5 file.")
        if 'korean_label' not in f:
             raise KeyError("Dataset 'korean_label' not found in HDF5 file.")

        n_samples = f['korean_image'].shape[0]
        # Optional: Verify shapes of the first element if needed
        # img_shape = f['korean_image'].shape[1:]
        # lbl_shape = f['korean_label'].shape[1:]
        # print(f"HDF5 Image shape: {img_shape}, Label shape: {lbl_shape}")
        # assert img_shape == (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), f"Expected image shape {(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)} but got {img_shape}"
        # assert lbl_shape == (IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS), f"Expected label shape {(IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS)} but got {lbl_shape}"

except FileNotFoundError:
    print(f"Error: Data file not found at {DATA_PATH}")
    exit()
except KeyError as e:
    print(f"Error: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading HDF5 file: {e}")
    exit()


# 학습/검증 인덱스 생성
indices = np.arange(n_samples)
np.random.seed(SEED)
np.random.shuffle(indices)
split_idx = int(n_samples * (1 - TEST_SIZE))
train_indices = indices[:split_idx]
val_indices = indices[split_idx:]

print(f"Total samples: {n_samples}")
print(f"Training samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")

# generator 함수 정의 (학습용/검증용)
def data_generator(indices_list, mode='train'):
    """
    indices_list: 사용할 샘플의 인덱스 리스트
    mode: 'train' 또는 'val' (not currently used for different augmentation)
    """
    with h5py.File(DATA_PATH, 'r') as f:
        images = f['korean_image']
        labels = f['korean_label']
        for idx in indices_list:
            # 이미지: float32 캐스팅 후 0~1 정규화 (Input shape: (128, 128, 4))
            img = images[idx].astype('float32') / 255.0
            # 레이블: float32 캐스팅 후 0~1 정규화 (Label shape: (128, 128, 4))
            lab = labels[idx].astype('float32') / 255.0

            # Ensure shapes are correct before yielding
            if img.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS):
                 print(f"Warning: Unexpected image shape {img.shape} for index {idx}. Expected {(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)}. Skipping.")
                 continue
            if lab.shape != (IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS):
                 print(f"Warning: Unexpected label shape {lab.shape} for index {idx}. Expected {(IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS)}. Skipping.")
                 continue

            yield img, lab

# tf.data.Dataset.from_generator 사용 (학습용)
train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_indices, mode='train'),
    output_signature=(
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS), dtype=tf.float32)
    )
)
# Repeat(), Shuffle(), Batch(), Prefetch() are common optimizations
train_dataset = train_dataset.shuffle(buffer_size=len(train_indices) // 10).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# train_dataset = train_dataset.repeat().shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Use repeat for indefinite training loop if needed

# tf.data.Dataset.from_generator 사용 (검증용)
val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_indices, mode='val'),
    output_signature=(
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=tf.float32),
        tf.TensorSpec(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_CHANNELS), dtype=tf.float32)
    )
)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# val_dataset = val_dataset.repeat().batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE) # Use repeat for indefinite validation loop if needed


# 🔧 가중치가 적용된 binary crossentropy loss 함수
def weighted_binary_crossentropy(pos_weight=15.0):
    def loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        # 픽셀별 binary crossentropy 계산 (pos_weight가 양성에 적용됨)
        # Applies element-wise, so works for multi-channel output
        loss_val = - (pos_weight * y_true * tf.math.log(y_pred) +
                      (1 - y_true) * tf.math.log(1 - y_pred))
        # 평균 계산 (배치 내 모든 픽셀, 모든 채널에 대한 평균)
        return tf.reduce_mean(loss_val)
    return loss

# 📐 U-Net 구조 (Skip Connections 추가)
def build_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)):
    inputs = layers.Input(shape=input_shape)

    # Downsampling path (Encoder)
    # Block 1
    conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    # Block 2
    conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)


    # Bottleneck
    bottleneck = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(pool2)
    bottleneck = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(bottleneck)

    # Upsampling path (Decoder)
    # Block 4
    up4 = layers.UpSampling2D((2, 2))(bottleneck)
    concat4 = layers.Concatenate()([up4, conv2]) # Skip connection
    conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(concat4)
    conv4 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(conv4)

    # Block 5
    up5 = layers.UpSampling2D((2, 2))(conv4)
    concat5 = layers.Concatenate()([up5, conv1]) # Skip connection
    conv5 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(concat5)
    conv5 = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(conv5)

    # Output layer: 4 channels, sigmoid activation
    outputs = layers.Conv2D(LABEL_CHANNELS, (1, 1), activation='sigmoid')(conv5) # Output 4 channels

    model = models.Model(inputs, outputs)
    return model

# 모델 생성 및 컴파일
# Ensure input_shape matches the data generator output
model = build_model(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS))

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss=weighted_binary_crossentropy(pos_weight=15.0), # Keep weighted BCE if appropriate
    metrics=['accuracy'] # Accuracy might be misleading for segmentation, consider IoU or Dice later
)
model.summary()

# 콜백 설정: 검증 loss 기준 모델 저장 및 얼리 스탑핑
# Ensure MODEL_SAVE_PATH directory exists
checkpoint = callbacks.ModelCheckpoint(
    MODEL_SAVE_PATH,
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1 # Add verbosity
)
earlystop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5, # Increase patience if needed
    restore_best_weights=True,
    verbose=1 # Add verbosity
)

# 🚀 모델 훈련 (generator를 이용하여 배치 단위로 데이터 로딩)
print("Starting model training...")
# Calculate steps per epoch if not using repeat()
# steps_per_epoch = len(train_indices) // BATCH_SIZE
# validation_steps = len(val_indices) // BATCH_SIZE

history = model.fit(
    train_dataset,
    # steps_per_epoch=steps_per_epoch, # Use if dataset doesn't repeat indefinitely
    validation_data=val_dataset,
    # validation_steps=validation_steps, # Use if dataset doesn't repeat indefinitely
    epochs=10, # Adjust number of epochs
    callbacks=[checkpoint, earlystop]
)

# 훈련 종료 후 최종 모델 저장 (best weights might already be saved by checkpoint/earlystop)
# model.save(MODEL_SAVE_PATH) # Can save the final state, or rely on restore_best_weights=True
print("-----------------------------------------")
print("Model training finished.")
# Note: If EarlyStopping restored best weights, the saved model from the checkpoint callback *is* the best one.
# Calling model.save() again here would save the weights from the *last* epoch if training finished normally,
# or the restored best weights if EarlyStopping kicked in and restored them.
# It's generally safe to rely on the ModelCheckpoint with save_best_only=True and EarlyStopping with restore_best_weights=True.
print(f"Best model should be saved at: {MODEL_SAVE_PATH}")